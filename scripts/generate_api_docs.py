#!/usr/bin/env python3
"""
Auto-generate API documentation from Python docstrings.

This script scans docs/pkg_docs/*/index.md for files with 'import_from' in their
front matter, then generates API documentation from the specified Python modules.

Usage:
    python scripts/generate_api_docs.py           # Generate all docs
    python scripts/generate_api_docs.py --check   # Check if docs are up-to-date (for CI)
    python scripts/generate_api_docs.py --module preprocessing  # Generate for one module

Front matter fields:
    import_from: Python module path(s) to document (required for auto-generation)
                 Can be a single string or a list of strings for combined docs
    include_imported: If true, include re-exported functions (default: false)

Example front matter:
    ---
    layout: docs_with_sidebar
    title: Word Embeddings
    import_from:
      - qhchina.analytics.word2vec
      - qhchina.analytics.vectors
    ---

Requirements:
    pip install docstring_parser pyyaml
"""

import argparse
import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path so we can import qhchina
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    from docstring_parser import parse as parse_docstring
    from docstring_parser.common import DocstringStyle
except ImportError:
    print("Error: docstring_parser is required. Install with: pip install docstring_parser")
    sys.exit(1)


# Markers for auto-generated content
API_START_MARKER = "<!-- API-START -->"
API_END_MARKER = "<!-- API-END -->"


def discover_doc_files(docs_path: Path) -> Dict[str, Dict]:
    """
    Discover documentation files that have 'import_from' in their front matter.
    
    Scans docs/pkg_docs/*/index.md for files with import_from field.
    
    Args:
        docs_path: Path to the docs directory
        
    Returns:
        Dict mapping module name (folder name) to config dict with:
        - doc_path: Path to the markdown file
        - import_from: List of Python module paths to import
        - include_imported: Whether to include re-exported functions
    """
    discovered = {}
    pkg_docs_path = docs_path / 'pkg_docs'
    
    if not pkg_docs_path.exists():
        return discovered
    
    for module_dir in pkg_docs_path.iterdir():
        if not module_dir.is_dir():
            continue
            
        index_file = module_dir / 'index.md'
        if not index_file.exists():
            continue
        
        # Read and parse front matter
        content = index_file.read_text(encoding='utf-8')
        front_matter, _ = parse_front_matter(content)
        
        # Check if import_from is defined
        if 'import_from' not in front_matter:
            continue
        
        import_from = front_matter['import_from']
        
        # Normalize to list
        if isinstance(import_from, str):
            import_from = [import_from]
        
        discovered[module_dir.name] = {
            'doc_path': index_file,
            'import_from': import_from,
            'include_imported': front_matter.get('include_imported', False),
        }
    
    return discovered


def get_public_members(module: Any, include_imported: bool = False) -> List[Tuple[str, Any]]:
    """
    Get public functions and classes from a module using its __all__.
    
    Args:
        module: The imported module
        include_imported: If True, include functions/classes imported from other modules.
                         Set to True for modules that re-export from submodules.
    
    Returns:
        List of (name, object) tuples
    """
    members = []
    
    # Use __all__ if defined, otherwise get all public names
    if hasattr(module, '__all__'):
        names = module.__all__
    else:
        # Fallback: all public names (no leading underscore)
        names = [name for name in dir(module) if not name.startswith('_')]
    
    for name in names:
        if hasattr(module, name):
            obj = getattr(module, name)
            # Only include functions and classes
            if inspect.isfunction(obj) or inspect.isclass(obj):
                # Check if it's defined in this module (not imported)
                if hasattr(obj, '__module__') and obj.__module__ == module.__name__:
                    members.append((name, obj))
                elif include_imported:
                    # Include imported items only if flag is set
                    members.append((name, obj))
    
    return members


def format_signature(name: str, obj: Any) -> str:
    """Format a function or class signature."""
    try:
        sig = inspect.signature(obj)
        if inspect.isclass(obj):
            # For classes, get __init__ signature
            try:
                init_sig = inspect.signature(obj.__init__)
                # Remove 'self' parameter
                params = list(init_sig.parameters.values())[1:]
                sig_str = f"{name}({', '.join(str(p) for p in params)})"
            except (ValueError, TypeError):
                sig_str = f"{name}()"
        else:
            sig_str = f"{name}{sig}"
        return sig_str
    except (ValueError, TypeError):
        return f"{name}()"


def docstring_to_markdown(docstring: Optional[str], is_class: bool = False) -> str:
    """
    Convert a docstring to markdown format.
    
    Handles both Google and NumPy style docstrings via docstring_parser.
    """
    if not docstring:
        return ""
    
    # Parse the docstring (auto-detects style)
    parsed = parse_docstring(docstring)
    
    lines = []
    
    # Short description
    if parsed.short_description:
        lines.append(parsed.short_description)
        lines.append("")
    
    # Long description
    if parsed.long_description:
        lines.append(parsed.long_description)
        lines.append("")
    
    # Parameters/Args
    if parsed.params:
        lines.append("**Parameters:**")
        for param in parsed.params:
            type_str = f" ({param.type_name})" if param.type_name else ""
            desc = param.description or ""
            # Handle multi-line descriptions with proper indentation
            desc_lines = desc.split('\n')
            if len(desc_lines) > 1:
                lines.append(f"- `{param.arg_name}`{type_str}: {desc_lines[0]}")
                for dl in desc_lines[1:]:
                    lines.append(f"  {dl}")
            else:
                lines.append(f"- `{param.arg_name}`{type_str}: {desc}")
        lines.append("")
    
    # Returns
    if parsed.returns:
        lines.append("**Returns:**")
        ret = parsed.returns
        type_str = f" ({ret.type_name})" if ret.type_name else ""
        desc = ret.description or ""
        lines.append(f"{type_str} {desc}".strip())
        lines.append("")
    
    # Raises
    if parsed.raises:
        lines.append("**Raises:**")
        for exc in parsed.raises:
            desc = exc.description or ""
            lines.append(f"- `{exc.type_name}`: {desc}")
        lines.append("")
    
    # Examples
    if parsed.examples:
        lines.append("**Example:**")
        lines.append("```python")
        for example in parsed.examples:
            if example.description:
                lines.append(example.description)
        lines.append("```")
        lines.append("")
    
    return "\n".join(lines)


def generate_api_markdown(members: List[Tuple[str, Any]]) -> Tuple[str, List[Dict]]:
    """
    Generate markdown for API documentation.
    
    Args:
        members: List of (name, object) tuples
    
    Returns:
        Tuple of (markdown_content, functions_list_for_frontmatter)
    """
    lines = []
    functions_list = []
    
    # Separate classes and functions
    classes = [(n, o) for n, o in members if inspect.isclass(o)]
    functions = [(n, o) for n, o in members if inspect.isfunction(o)]
    
    # Document classes first
    for name, cls in classes:
        anchor = name.lower()
        functions_list.append({'name': f"{name}", 'anchor': anchor})
        
        lines.append(f'<h3 id="{anchor}">{name}</h3>')
        lines.append("")
        
        # Class signature
        sig = format_signature(name, cls)
        lines.append("```python")
        lines.append(sig)
        lines.append("```")
        lines.append("")
        
        # Class docstring
        if cls.__doc__:
            lines.append(docstring_to_markdown(cls.__doc__, is_class=True))
        
        # Document public methods
        public_methods = [
            (method_name, method) 
            for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not method_name.startswith('_') or method_name in ('__init__',)
        ]
        
        # Filter to only methods defined in this class (not inherited)
        own_methods = []
        for method_name, method in public_methods:
            if method_name in cls.__dict__:
                own_methods.append((method_name, method))
        
        if own_methods:
            for method_name, method in own_methods:
                if method_name == '__init__':
                    continue  # Already covered in class signature
                
                method_anchor = f"{anchor}-{method_name.lower()}"
                functions_list.append({'name': f"{name}.{method_name}()", 'anchor': method_anchor})
                
                lines.append(f'<h4 id="{method_anchor}">{name}.{method_name}()</h4>')
                lines.append("")
                
                # Method signature
                try:
                    sig = inspect.signature(method)
                    # Remove 'self' from signature display
                    params = list(sig.parameters.values())
                    if params and params[0].name == 'self':
                        params = params[1:]
                    sig_str = f"{method_name}({', '.join(str(p) for p in params)})"
                except (ValueError, TypeError):
                    sig_str = f"{method_name}()"
                
                lines.append("```python")
                lines.append(sig_str)
                lines.append("```")
                lines.append("")
                
                if method.__doc__:
                    lines.append(docstring_to_markdown(method.__doc__))
        
        lines.append("<br>")
        lines.append("")
    
    # Document functions
    for name, func in functions:
        anchor = name.lower()
        functions_list.append({'name': f"{name}()", 'anchor': anchor})
        
        lines.append(f'<h3 id="{anchor}">{name}()</h3>')
        lines.append("")
        
        # Function signature
        sig = format_signature(name, func)
        lines.append("```python")
        lines.append(sig)
        lines.append("```")
        lines.append("")
        
        # Function docstring
        if func.__doc__:
            lines.append(docstring_to_markdown(func.__doc__))
        
        lines.append("<br>")
        lines.append("")
    
    return "\n".join(lines), functions_list


def parse_front_matter(content: str) -> Tuple[Dict, str]:
    """
    Parse YAML front matter from markdown content.
    
    Returns:
        Tuple of (front_matter_dict, remaining_content)
    """
    if not content.startswith('---'):
        return {}, content
    
    # Find the closing ---
    end_match = re.search(r'\n---\n', content[3:])
    if not end_match:
        return {}, content
    
    end_pos = end_match.end() + 3
    front_matter_str = content[4:end_pos - 4]
    remaining = content[end_pos:]
    
    try:
        front_matter = yaml.safe_load(front_matter_str)
        return front_matter or {}, remaining
    except yaml.YAMLError:
        return {}, content


def build_front_matter(data: Dict) -> str:
    """Build YAML front matter string."""
    lines = ['---']
    
    # Preserve order: layout, title, permalink, then functions
    for key in ['layout', 'title', 'permalink']:
        if key in data:
            lines.append(f"{key}: {data[key]}")
    
    # Handle functions list specially for nicer formatting
    if 'functions' in data:
        lines.append("functions:")
        for func in data['functions']:
            lines.append(f"  - name: {func['name']}")
            lines.append(f"    anchor: {func['anchor']}")
    
    # Any other keys
    for key, value in data.items():
        if key not in ['layout', 'title', 'permalink', 'functions']:
            lines.append(f"{key}: {value}")
    
    lines.append('---')
    return '\n'.join(lines)


def update_doc_file(doc_path: Path, api_markdown: str, functions_list: List[Dict]) -> str:
    """
    Update a documentation file with new API content.
    
    Preserves content outside the API markers and updates the functions list in front matter.
    
    Returns:
        The updated file content
    """
    if not doc_path.exists():
        raise FileNotFoundError(f"Documentation file not found: {doc_path}")
    
    content = doc_path.read_text(encoding='utf-8')
    
    # Parse front matter
    front_matter, body = parse_front_matter(content)
    
    # Update functions list in front matter
    front_matter['functions'] = functions_list
    
    # Find and replace API section
    if API_START_MARKER in body and API_END_MARKER in body:
        start_idx = body.index(API_START_MARKER)
        end_idx = body.index(API_END_MARKER) + len(API_END_MARKER)
        
        new_body = (
            body[:start_idx] +
            API_START_MARKER + "\n\n" +
            api_markdown +
            "\n" + API_END_MARKER +
            body[end_idx:]
        )
    else:
        # No markers found - add them at the end with a note
        new_body = body + f"\n\n---\n\n## API Reference\n\n{API_START_MARKER}\n\n{api_markdown}\n{API_END_MARKER}\n"
    
    # Rebuild full content
    new_content = build_front_matter(front_matter) + "\n" + new_body
    
    return new_content


def generate_module_docs(module_name: str, config: Dict, dry_run: bool = False) -> bool:
    """
    Generate documentation for a single module.
    
    Args:
        module_name: Name of the module (folder name in pkg_docs)
        config: Config dict with doc_path, import_from, include_imported
        dry_run: If True, don't write changes
    
    Returns:
        True if documentation was updated, False if unchanged
    """
    print(f"Processing module: {module_name}")
    
    members = []
    include_imported = config.get('include_imported', False)
    
    # Import each module and collect public members
    for module_path in config['import_from']:
        try:
            module = importlib.import_module(module_path)
            module_members = get_public_members(module, include_imported=include_imported)
            members.extend(module_members)
            print(f"  Imported {module_path}: {len(module_members)} members")
        except ImportError as e:
            print(f"  Error importing {module_path}: {e}")
    
    if not members:
        print(f"  No public members found")
        return False
    
    print(f"  Total: {len(members)} public members")
    
    # Generate markdown
    api_markdown, functions_list = generate_api_markdown(members)
    
    # Update the doc file
    doc_path = config['doc_path']
    
    try:
        new_content = update_doc_file(doc_path, api_markdown, functions_list)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return False
    
    # Check if content changed
    old_content = doc_path.read_text(encoding='utf-8')
    if new_content == old_content:
        print(f"  Documentation is up-to-date")
        return False
    
    if dry_run:
        print(f"  Would update: {doc_path}")
        return True
    
    # Write the updated content
    doc_path.write_text(new_content, encoding='utf-8')
    print(f"  Updated: {doc_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate API documentation from docstrings")
    parser.add_argument('--check', action='store_true', 
                       help="Check if docs are up-to-date without modifying (for CI)")
    parser.add_argument('--module', type=str, 
                       help="Generate docs for a specific module only")
    parser.add_argument('--list-modules', action='store_true',
                       help="List available modules")
    args = parser.parse_args()
    
    root_path = Path(__file__).parent.parent
    docs_path = root_path / 'docs'
    
    # Discover all doc files with import_from in front matter
    discovered_modules = discover_doc_files(docs_path)
    
    if not discovered_modules:
        print("No documentation files found with 'import_from' in front matter.")
        print("Add 'import_from: module.path' to your doc file's front matter.")
        return
    
    if args.list_modules:
        print("Available modules (discovered from docs/pkg_docs/*/index.md):")
        for name, config in sorted(discovered_modules.items()):
            imports = ', '.join(config['import_from'])
            print(f"  - {name}: {imports}")
        return
    
    # Filter to specific module if requested
    if args.module:
        if args.module not in discovered_modules:
            print(f"Unknown module: {args.module}")
            print(f"Available: {', '.join(sorted(discovered_modules.keys()))}")
            sys.exit(1)
        modules_to_process = {args.module: discovered_modules[args.module]}
    else:
        modules_to_process = discovered_modules
    
    any_changed = False
    for module_name, config in modules_to_process.items():
        changed = generate_module_docs(module_name, config, dry_run=args.check)
        any_changed = any_changed or changed
    
    if args.check and any_changed:
        print("\nDocumentation is out of date! Run: python scripts/generate_api_docs.py")
        sys.exit(1)
    elif args.check:
        print("\nDocumentation is up-to-date.")
    else:
        print("\nDone!")


if __name__ == '__main__':
    main()
