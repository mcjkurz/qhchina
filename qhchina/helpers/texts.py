def load_texts(filenames):
    """
    Loads text from a file or a list of files.

    Parameters:
    filenames (str or list): The filename or list of filenames to load text from.

    Returns:
    str or list: The text content of the file or a list of text contents if multiple files are provided.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    texts = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts