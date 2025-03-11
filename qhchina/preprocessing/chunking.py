def split_into_chunks(tokens, chunk_size, overlap=0.0):
    """
    Splits a list of tokens into chunks with optional overlap between consecutive chunks.
    
    Parameters:
    tokens (list): The list of tokens to be split.
    chunk_size (int): The size of each chunk.
    overlap (float): The fraction of overlap between consecutive chunks (0.0 to 1.0).
                    Default is 0.0 (no overlap).
    
    Returns:
    list: A list of chunks, where each chunk is a list of tokens.
    
    Raises:
    ValueError: If overlap is not between 0 and 1.
    """
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be between 0 and 1")
        
    if not tokens:
        return []
        
    overlap_size = int(chunk_size * overlap)
    stride = chunk_size - overlap_size
    
    chunks = []
    for i in range(0, len(tokens) - chunk_size + 1, stride):
        chunks.append(tokens[i:i + chunk_size])
    
    # Handle the last chunk if there are remaining tokens
    if i + chunk_size < len(tokens):
        chunks.append(tokens[-chunk_size:])
        
    return chunks
