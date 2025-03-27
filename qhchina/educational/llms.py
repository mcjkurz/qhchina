import torch
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
import numpy as np
import pandas as pd

def predict_next_token(
    input_text: str, 
    model, 
    tokenizer, 
    k: int = 10, 
    device: str = 'cpu',
    result: Optional[str] = "df"
) -> Union[List[Tuple[str, float]], Optional[pd.DataFrame], None]:
    """
    Predicts the next token probabilities for a given input text.
    
    Args:
        input_text: The input text for which to predict the next token
        model: A causal language model (e.g., GPT-2, Llama, etc.)
        tokenizer: The tokenizer corresponding to the model
        k: Number of top tokens to return
        device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.)
        result: Output format, one of:
                - "list" or None: return list of (token, probability) tuples
                - "df": return pandas DataFrame with tokens and probabilities (default)
                - "text": print tokens nicely formatted, return None
                - "horizontal": show horizontal bar chart, return None
                - "vertical": show vertical bar chart, return None
        
    Returns:
        If result="list" or None: List of tuples containing (token_str, probability)
        If result="df": pandas DataFrame with columns 'token' and 'probability'
        Otherwise: None
    """
    # Set device and move model to it
    device = torch.device(device)
    model = model.to(device)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Encode the input text and move to device
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        logits = logits[:, -1, :]  # Get logits for the last position
    
    # Get top k tokens
    probs = torch.softmax(logits, dim=-1)
    top_k = torch.topk(probs, k=k)
    
    # Convert token IDs to tokens and clean up the output
    top_tokens = []
    for token_id, prob in zip(top_k.indices[0], top_k.values[0]):
        token_str = tokenizer.decode(token_id, clean_up_tokenization_spaces=True)
        # Handle special tokens and partial words
        if '▁' in token_str:  # For models like Llama
            token_str = token_str.replace('▁', ' ').strip()
        elif 'Ġ' in token_str:  # For GPT-2
            token_str = token_str.replace('Ġ', ' ').strip()
        top_tokens.append((token_str, prob.item()))
    
    # Handle the result parameter
    if result == "text":
        # Print tokens row by row
        print(f"\nTop {k} predictions for next token after: '{input_text}'")
        print('-' * 50)
        max_token_len = max(len(token) for token, _ in top_tokens)
        for i, (token, prob) in enumerate(top_tokens):
            # Format with token, probability and percentage
            print(f"{i+1:2d}. '{token}'{' ' * (max_token_len - len(token) + 2)}{prob:.6f} ({prob*100:.2f}%)")
        print('-' * 50)
        return None
    
    elif result in ["horizontal", "vertical"]:
        # Prepare for visualization
        tokens = [token for token, _ in top_tokens]
        probabilities = [prob for _, prob in top_tokens]
        
        plt.figure(figsize=(10, 6))
        
        if result == "horizontal":
            # Create horizontal bars
            # Sort in descending order for horizontal bars to have longest bar at top
            sorted_indices = np.argsort(probabilities)  # Descending order
            sorted_tokens = [tokens[i] for i in sorted_indices]
            sorted_probs = [probabilities[i] for i in sorted_indices]
            
            y_pos = np.arange(len(sorted_tokens))
            plt.barh(y_pos, sorted_probs, align='center')
            plt.yticks(y_pos, sorted_tokens)
            plt.xlabel('Probability')
            plt.title('Next Token Probabilities')
        else:  # vertical
            # Create vertical bars
            x_pos = np.arange(len(tokens))
            plt.bar(x_pos, probabilities, align='center')
            plt.xticks(x_pos, tokens, rotation=45, ha='right')
            plt.ylabel('Probability')
            plt.title('Next Token Probabilities')
        
        plt.tight_layout()
        plt.show()
        return None
    
    elif result == "df":
        # Return a pandas DataFrame
        tokens = [token for token, _ in top_tokens]
        probabilities = [prob for _, prob in top_tokens]
        return pd.DataFrame({'token': tokens, 'probability': probabilities})
    
    else:
        # Default: return list of tuples (for "list" or None)
        return top_tokens