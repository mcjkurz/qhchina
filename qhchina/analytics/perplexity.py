import logging
from typing import List, Tuple, Union, Optional, Any
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("qhchina.analytics.perplexity")


def calculate_perplexity_of_tokens(
    model: Any, 
    tokenizer: Any, 
    sequence: Union[str, List[str]], 
    context_size: int = 64, 
    verbose: bool = False, 
    device: Optional[Any] = None
) -> List[Tuple[str, int, float, bool]]:
    """
    Calculate the perplexity of each token in a sequence with consistent context size.
    
    Args:
        model: The language model
        tokenizer: The tokenizer corresponding to the model
        sequence: Input text sequence
        context_size: Number of tokens to use as context (default: 64)
        verbose: Whether to print debug information (default: False)
        device: Device to run computations on (default: cuda-else-cpu auto-detect).
                Can be a string ('cpu', 'cuda') or torch.device object.
        
    Returns:
        List of tuples (token, token_id, perplexity, has_full_context)
        where has_full_context is a boolean indicating if the token had the requested context_size
    
    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for calculate_perplexity_of_tokens(). "
            "Install it with: pip install torch"
        )
    
    model.eval()

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    token_perplexities = []

    # Tokenize the sequence
    if isinstance(sequence, str):
        # For string input
        ids = tokenizer.encode(sequence, add_special_tokens=False)
    else:
        # For pre-tokenized input
        ids = tokenizer.convert_tokens_to_ids(sequence)

    # We can only calculate perplexity for tokens that have at least some context
    if len(ids) <= 1:
        if verbose:
            logger.info("Sequence too short to calculate perplexity.")
        return []

    # Process each token
    for i in range(1, len(ids)):
        # Determine how much context we can use
        available_context = i
        has_full_context = available_context >= context_size
        
        # Use either the full requested context or all available context
        if has_full_context:
            context_start = i - context_size
        else:
            context_start = 0
            
        context_ids = ids[context_start:i]  # Context up to, but not including, the current token
        target_id = ids[i]  # Current target token

        # Prepare input for the model
        input_ids = torch.tensor([context_ids]).to(device)
        attention_mask = torch.tensor([[1] * len(context_ids)]).to(device)

        # Run the model to get logits
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get the logits corresponding to the target token
        relevant_logits = logits[0, -1, :]  # Logits for the last token in the context
        relevant_label = torch.tensor([target_id]).to(device)

        # Calculate cross-entropy loss for the target token
        loss = torch.nn.functional.cross_entropy(
            relevant_logits.view(1, -1), relevant_label.view(-1)
        )

        # Compute perplexity
        perplexity = torch.exp(loss).item()

        # Decode the token for readability
        token = tokenizer.decode([target_id])
        
        # Store result with context status
        token_perplexities.append((token, target_id, perplexity, has_full_context))

        if verbose:
            context_info = f"[Full context: {context_size}]" if has_full_context else f"[Partial context: {available_context}]"
            logger.info(f"Token: {token}, ID: {target_id}, Perplexity: {perplexity} {context_info}")
            
    return token_perplexities

def calculate_word_perplexity(
    model: Any, 
    tokenizer: Any, 
    context: str, 
    target_word: str, 
    device: Optional[Any] = None
) -> Tuple[float, List[Tuple[str, int, float]]]:
    """
    Calculate the perplexity of a target word given a context.
    
    Args:
        model: The language model
        tokenizer: The tokenizer corresponding to the model
        context: Context text (string) preceding the target word
        target_word: The word to calculate perplexity for
        device: Device to run computations on (default: cuda-else-cpu auto-detect).
                Can be a string ('cpu', 'cuda') or torch.device object.
        
    Returns:
        tuple: (perplexity, token_perplexities)
            - perplexity: The average perplexity of the target word
            - token_perplexities: List of tuples (token, token_id, perplexity) for each token in the target word
    
    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for calculate_word_perplexity(). "
            "Install it with: pip install torch"
        )
    
    model.eval()

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Tokenize the context
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    
    # Tokenize the target word
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)
    
    if len(target_ids) == 0:
        return (0.0, [])
    
    token_perplexities = []
    total_loss = 0.0
    
    # Process tokens in the target word one by one
    for i, target_id in enumerate(target_ids):
        # For each token in the target, we use all previous context plus any preceding tokens in the target
        current_context_ids = context_ids + target_ids[:i]
        
        # Skip if there's no context
        if len(current_context_ids) == 0:
            continue
            
        # Prepare input for the model
        input_ids = torch.tensor([current_context_ids]).to(device)
        attention_mask = torch.tensor([[1] * len(current_context_ids)]).to(device)

        # Run the model to get logits
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get the logits corresponding to the target token
        relevant_logits = logits[0, -1, :]  # Logits for the last token in the context
        relevant_label = torch.tensor([target_id]).to(device)

        # Calculate cross-entropy loss for the target token
        loss = torch.nn.functional.cross_entropy(
            relevant_logits.view(1, -1), relevant_label.view(-1)
        )
        
        # Add to total loss
        total_loss += loss.item()

        # Compute perplexity
        perplexity = torch.exp(loss).item()

        # Decode the token for readability
        token = tokenizer.decode([target_id])
        
        # Store result
        token_perplexities.append((token, target_id, perplexity))
    
    # Calculate average perplexity across all tokens in the target word
    avg_perplexity = torch.exp(torch.tensor(total_loss / len(target_ids))).item() if len(target_ids) > 0 else 0.0
    
    return (avg_perplexity, token_perplexities)

def visualize_perplexities(
    perplexities: List[float], 
    labels: List[str], 
    width: float = 14, 
    height: float = 3.5, 
    color: str = 'red', 
    filename: Optional[str] = None
) -> None:
    """
    Visualize perplexities with given labels.
    
    Args:
        perplexities: List of perplexity values (floats)
        labels: List of labels for x-axis ticks
        width: Figure width (default: 14)
        height: Figure height (default: 3.5)
        color: Line color (default: 'red')
        filename: File to save the plot to (default: None)
    """
    # Ensure perplexities and labels are lists
    perplexities = list(perplexities)
    labels = list(labels)
    
    # Ensure we have same number of labels as perplexities
    if len(labels) != len(perplexities):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of perplexities ({len(perplexities)})")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Plot positions (1-indexed)
    positions = np.arange(1, len(perplexities) + 1)
    
    # Plot the line with markers
    ax.plot(positions, perplexities, marker='o', linestyle='-', color=color)
    
    # Add value annotations above points
    for i, ppl in enumerate(perplexities):
        if not np.isnan(ppl):
            annotation_text = f"{round(ppl, 1)}"
            ax.annotate(annotation_text, (positions[i], ppl),
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=8)
    
    # Configure x-axis with the provided labels
    ax.set_xlabel('Character Sequence')
    ax.set_ylabel('Character Perplexity (log scale)')
    ax.set_xlim(0, len(perplexities) + 1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    ax.set_ylim(bottom=1, top=max(perplexities) * 10 if perplexities else 10)
    
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()