import json
import os
from dateutil import parser
import numpy as np

def load_json(file_path):
    '''
    Load json file
    '''
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

def concatenate_entry(d):
    '''
    For all keys in a dictionary, if a value is a list, concatenate it.
    '''
    for key, value in d.items():
        if isinstance(value, list):  
            d[key] = ';'.join(map(str, value))  # Convert list to a string separated by ';'
    return d


def append_to_json(file_path, data):
    '''
    Append a dict or a list of dicts to a JSON file.
    '''
    try:
        if not os.path.exists(file_path):
            # Create an empty JSON file with an empty list if it does not exist yet
            with open(file_path, 'w') as file:
                json.dump([], file)
        #Open the existing file
        with open(file_path, 'r+') as file:
            file_data = json.load(file)
            if type(data)==list:
                for d in data:
                    if type(d)==dict:
                        file_data.append(concatenate_entry(d))
            else:
                file_data.append(concatenate_entry(data))
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")


def save_result(output,json_file_path):
    '''
    Save output results to a JSON file.
    '''
    try:    
        if type(output)==str:
            user_data = json.loads(output)
            append_to_json(json_file_path, user_data)
        else:
            append_to_json(json_file_path, output)
    except json.JSONDecodeError:
        #The output was not well formatted
        pass

def convert_to_date(date_str):
    '''
    Convert a string to a Date object
    '''
    try:
        return parser.parse(date_str)
    except (ValueError, TypeError):
        return None

def truncate_texts(texts, tokenizer, max_length=77):
    '''
    Truncate texts to the max length allowed by CLIP
    '''
    truncated_texts = []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        if len(tokenized_text) > max_length - 2:  # Subtract 2 to account for special tokens
            tokenized_text = tokenized_text[:max_length - 2]
        truncated_text = tokenizer.convert_tokens_to_string(tokenized_text)
        truncated_texts.append(truncated_text)
    return truncated_texts


def cosine_sim(embedding1, embedding2):
    '''
    Compute cosine similarity between two embeddings vectors
    '''
    dot_product = np.dot(embedding1, embedding2)
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    return dot_product / (norm_embedding1 * norm_embedding2)