# qhchina/utils.py

def split_into_chunks(texts, chunk_size, overlap=0, only_full=True):

    if not isinstance(texts, list):
        texts = [texts]

    chunks = []
    for text in texts:
        curr_chunks = []
        text_len = len(text)
        step = int(chunk_size * (1 - overlap))  # Calculate step size based on overlap
        for i in range(0, text_len, step):
            chunk = text[i:i + chunk_size]
            if only_full and len(chunk) != chunk_size:
                continue
            curr_chunks.append(chunk)
        chunks.append(curr_chunks)
    
    return chunks