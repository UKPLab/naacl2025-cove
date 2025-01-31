import pandas as pd
from tqdm import tqdm
import json
import os
import requests
import base64
import argparse
import time


def detect_labels(image_path, 
                  image_id, 
                  api_key, 
                  output_path,
                  sleep=1):
    '''
    Detect objects in an image using google vision api
    Params:
        image_path (str): path to the image
        image_id (int): unique identifier of the image
        api_key(str): api key to access google vision products
        output_path (str): path to the output file to save the detected objects
        sleep (int): sleeping time between two API calls
    '''
    api_url = 'https://vision.googleapis.com/v1/images:annotate?key=' + api_key
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
        payload = {
            "requests": [
                {
                    "image": {
                        "content": image_data
                    },
                    "features": [
                        {
                            "type": "OBJECT_LOCALIZATION",
                        }
                    ]
                }
            ]
        }
    response = requests.post(api_url, json=payload)
    data = response.json()

    if 'error' in data:
        print('Error:', data['error']['message'])
        return

    try:
        objects = data['responses'][0]['localizedObjectAnnotations']
    except:
        print('No objects detected')
        return

    names = []
    scores = []
    mids = []
    bounds = []
    
    for object in objects:
        names.append(object['name']) # type of object
        scores.append(object['score']) # confidence score
        mids.append(object['mid']) # machine-generated identifier
        bounds.append(object['boundingPoly']['normalizedVertices']) # bounding box coordinates

    df = pd.DataFrame({'image_id': [image_id]*len(objects),'name': names, 'score': scores, 'mid': mids, 'bounds': bounds})

    #Append detected objects to the results file
    df.to_csv(output_path, index=False, mode='a',
              header = not os.path.exists(output_path))
    time.sleep(sleep)
    return df


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Detect objects in images using the Google Vision API.')
    parser.add_argument('--google_vision_api_key', type=str,  default= " ", #Provide your own key here as default value
                        help='Your key to access the Google Vision services.') 
    parser.add_argument('--dataset', type=str,  default= "newsclippings", choices=['newsclippings', '5pils-ooc'],
                        help='The dataset to use.') 
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 
    parser.add_argument('--sleep', type=int, default=1,
                        help='The waiting time between two web detection API calls') 
    

    args = parser.parse_args()
    api_key = os.getenv(args.google_vision_api_key)

    #Load data
    if args.dataset=='newsclippings':
        data = json.load(open(f'data/newsclippings/{args.split}.json', 'r'))
        images_root_folder = 'data/newsclippings/visual_news/origin'
    else:
        data = json.load(open(f'data/5pils_ooc/test.json', 'r'))
        images_root_folder = 'data/5pils_ooc'

    #Define output path
    os.makedirs(f'data/{args.dataset}/evidence/',exist_ok=True)
    os.makedirs(f'data/{args.dataset}/evidence/{args.split}',exist_ok=True)
    output_path = f'data/{args.dataset}/evidence/{args.split}/obj_detection.csv'

    df = pd.DataFrame()
    seen_images = [] 
    #Keep track of images that have already been processed (in newsclippings some image appear twice, in 5pils-ooc, all images appear twice)
    for item in tqdm(data):
        if item['image_id'] in seen_images:
            continue
        else:
            detect_labels(os.path.join(images_root_folder, item['image_path']), 
                          item['image_id'], 
                          api_key, 
                          output_path,
                          sleep=args.sleep)
            # each image used only once
            seen_images.append(item['image_id'])