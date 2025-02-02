import json
import os
from dateutil import parser
import numpy as np
from PIL import Image
from io import BytesIO
import requests as rq
from bs4 import BeautifulSoup as bs
from trafilatura import bare_extraction
import Levenshtein as lev

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


def download_image(url, file_path, max_size_mb=10):
    '''
    Download evidence images.
    '''
    try:
        # Send a GET request to the URL
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = rq.get(url, stream=True, timeout=(10,10),headers=headers)
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to download. Status code: {response.status_code}")
            return None
        # Check the content type to be an image
        if 'image' not in response.headers.get('Content-Type', ''):
            print("URL does not point to an image.")
            return None
        # Check the size of the image
        if int(response.headers.get('Content-Length', 0)) > max_size_mb * 1024 * 1024:
            print(f"Image is larger than {max_size_mb} MB.")
            return None
        # Read the image content
        image_data = response.content
        if not image_data:
            print("No image data received.")
            return None
        image = Image.open(BytesIO(image_data))
        image.verify()
        image = Image.open(BytesIO(image_data))
        # Save the image to a file
        image.save(file_path + '.png')
        print("Image downloaded and saved successfully.")
    except rq.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def compute_url_distance(url1,url2,threshold):
    distance = lev.distance(url1,url2)
    if distance < threshold:
        return True
    else:
        return False

def find_image_caption(soup, image_url,threshold=25):
    '''
    Retrieve the caption corresponding to an image url by searching the html in BeautifulSoup format.
    '''
    img_tag = None
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or img.get('data-original')
        if src and compute_url_distance(src, image_url, threshold):
            img_tag = img
            break
    if not img_tag:
        return "Image not found"
    figure = img_tag.find_parent('figure')
    if figure:
        figcaption = figure.find('figcaption')
        if figcaption:
            return figcaption.get_text().strip()
    for sibling in img_tag.find_next_siblings(['div', 'p','small']):
        if sibling.get_text().strip():
            return sibling.get_text().strip()
    title = img_tag.get('title')
    if title:
        return title.strip()
    # Strategy 4: Use the alt attribute of the image
    alt_text = img_tag.get('alt')
    if alt_text:
        return alt_text.strip()

    return "Caption not found"


def extract_info_trafilatura(page_url,image_url):
    try:
        headers= {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'} 
        response = rq.get(page_url, headers=headers, timeout=(10,10))
        if response.status_code == 200:
            #Extract content with Trafilatura
            result = bare_extraction(response.text,
                                   include_images=True,
                                   include_tables=False)
            #Remove unnecessary contente
            keys_to_keep = ['title','author','url',
                            'hostname','description','sitename',
                            'date','text','language','image','pagetype']
            result = {key: result[key] for key in keys_to_keep if key in result}
            result['image_url'] = image_url
            # Finding the image caption
            image_caption = []
            soup = bs(response.text, 'html.parser')
            for img in image_url:
                image_caption.append(find_image_caption(soup, img))
            image_caption.append(find_image_caption(soup,result['image']))
            result['image_caption'] = image_caption
            result['url'] = page_url
            return result
        else:
            return "Failed to retrieve webpage"
    except Exception as e:
        return f"Error occurred: {e}"