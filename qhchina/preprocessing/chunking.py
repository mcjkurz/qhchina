
def split_into_chunks(tokens, chunk_size):
    """
    Splits a list of tokens into chunks of equal length.
    
    Parameters:
    tokens (list): The list of tokens to be split.
    chunk_size (int): The size of each chunk.
    
    Returns:
    list: A list of chunks, where each chunk is a list of tokens.
    """
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
