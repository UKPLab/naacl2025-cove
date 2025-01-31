import requests
import time 
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import numpy as np
import face_recognition
import torch


def get_wikipedia_images(person_name,
                        max_images=3,
                        sleep=0.5):
    '''
    Collect images associated with the Wikipedia page of a specific person.
    Params:
        person_name (str): the name of the person
        max_images (int): the max number of images to retrieve on the person's wikipedia page
        sleep (float): sleeping time after collecting the images
    '''
    # Construct Wikipedia URL from the person's name
    person_name_url = person_name.replace(' ', '_')
    page_url = f"https://en.wikipedia.org/wiki/{person_name_url}"
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    # Fetch the page and parse for the images
    try:
        response = requests.get(page_url,headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all('img')
    except:
        return []
    loaded_images = []
    for img in images:        
        if len(loaded_images) >= max_images:
            break
        # Get the image URL
        img_url = img.get('src')
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif img_url.startswith('/'):
            img_url = 'https://en.wikipedia.org' + img_url
        if 'upload.wikimedia.org' in img_url and '.svg' not in img_url and 'static/images/' not in img_url:
            img_url = '/'.join(img_url.split('/')[:-1]).replace('thumb/','') # Get the full version, not the thumbnail version
            try:
                image_response = requests.get(img_url,headers=headers)
                loaded_image = Image.open(BytesIO(image_response.content)).convert('RGB')
                loaded_images.append(loaded_image)
            except:
                continue
    time.sleep(sleep)
    return loaded_images


def genre_entity_linking(text, 
                         model, 
                         tokenizer, 
                         num_beams=5, 
                         num_return_sequences=5):
    '''
    Takes a text as input and return Wikipedia entities that are mentioned in that text.
    '''
    # Tokenize the input text
    inputs = tokenizer([text], return_tensors="pt", truncation=True)
    # Generate output sequences
    outputs = model.generate(**inputs, num_beams=num_beams, num_return_sequences=num_return_sequences, max_new_tokens=20)
    # Decode the generated sequences to get entity names
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs


def get_face_encodings(image):
    '''
    Compute the face embeddings of an image using the face_recognition library models
    '''
    face_image = np.array(image)
    face_locations = face_recognition.face_locations(face_image)
    face_encodings = face_recognition.face_encodings(face_image, face_locations)
    return list(face_encodings)


def embed_image(image_path, model, processor, device):
    '''
    Compute image embedding
    '''
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy()

def embed_query(query, model, processor, device):
    '''
    Compute text query embedding
    '''
    inputs = processor(text=query, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_features = model.get_text_features(**inputs)
    return query_features.cpu().numpy()


def find_nearest_neighbors(image_path, 
                           index, 
                           query='', 
                           t=0.3, 
                           k=20):
    '''
    Find the joint nearest neighbor of the image and the text query
    Params:
        image_path (str): path to the image to match with entities
        index (obj): index of all 6M OVEN entities
        query (str): a text query that acts as a modifier to the image embedding
        t (float): the weight given to the text query when combining it with the image embedding
        k (int): the number of OVEN entities to retrieve
    '''
    image_embedding = embed_image(image_path).astype(np.float32)

    if query == '': 
        # If no query is provided, return the nearest neighbors of the image
        joint_embedding = image_embedding
    else:
        query_embedding = embed_query(query).astype(np.float32)
        joint_embedding = t * query_embedding + (1-t) * image_embedding
    distances, indices = index.search(joint_embedding, k)
    return distances, indices