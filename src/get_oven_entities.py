import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import os
import faiss
import json
import pandas as pd
from wikipedia_entity_collection.utils import *
from utils import *
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Retrieve topk entities from the oven index for an image.')
    parser.add_argument('--dataset', type=str,  default= "newsclippings", choices=['newsclippings', '5pils-ooc'],
                        help='The dataset to use.') 
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 
    parser.add_argument('--t', type=int, default=0.3,
                        help='Similarity threshold to consider a match with a Wikipedia entity') 
    parser.add_argument('--k', type=int, default=5,
                        help='Number of entities to retrieve') 
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='The weight given to the text query accompanying the image (only if People are shown in the image)') 
    

    args = parser.parse_args()
    #Load the FAISS index from disk
    print('Loading index')
    index = faiss.read_index("wikipedia_entity_collection/oven.index")
    #Load the entities
    entities = []
    file_path = 'wikipedia_entity_collection/Wiki6M_ver_1_0_title_only.jsonl'
    #Open and read the file
    print('Loading entitites')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            #Parse each line as JSON and append to the list
            data = json.loads(line)
            entities.append(data['wikipedia_title'])
    print('Number of entities: %s'%len(entities))

    #Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Load google vision objects
    detection = pd.read_csv(f'data/{args.dataset}/evidence/{args.split}/obj_detection.csv')

    #Load data
    if args.dataset=='newsclippings':
        data = json.load(open(f'data/newsclippings/{args.split}.json', 'r'))
        images_root_folder = 'data/newsclippings/visual_news/origin'
    else:
        data = json.load(open(f'data/5pils_ooc/test.json', 'r'))
        images_root_folder = 'data/5pils_ooc'

    results = []
    seen_images = []
    #Each image is processed only once
    for item in tqdm(data):
        im_path = item['image_path']
        if im_path in seen_images:
            continue
        seen_images.append(im_path)
        people_detected = False
        for row_ix, row in detection.iterrows():
            if row['image_id'] == item['image_id']:
                if row['name'] == 'Person':
                    people_detected = True
                    break
        if not people_detected:
            query = ''
        else:
            query = 'A person'
        #Retrieve top 5
        try:
            distances, indices = find_nearest_neighbors(model, processor,
                                                            device, im_path,  images_root_folder,
                                                                index,
                                                                query, 
                                                                t=args.t, 
                                                                k=args.k)  
                
            #Output: list of the top 5 entities
            topk_entities=[entities[idx] for idx in indices.flatten()]
        except:
            topk_entities = []
        results.append({im_path:topk_entities}) 

    output_path = f'data/{args.dataset}/evidence/{args.split}/oven_entities.json'
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)