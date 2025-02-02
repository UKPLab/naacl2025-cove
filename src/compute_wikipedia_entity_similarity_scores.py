#File to precompute and save the similarity scores of wikipedia entities embeddings and the image
import json
import pandas as pd
import numpy as np
import os
import spacy
from tqdm import tqdm
import argparse
from utils import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute similarity scores between images and  wikipedia entities.')
    parser.add_argument('--dataset', type=str,  default= "newsclippings", choices=['newsclippings', '5pils-ooc'],
                        help='The dataset to use.') 
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 


    args = parser.parse_args()

    #Load data
    if args.dataset=='newsclippings':
        data = json.load(open(f'data/newsclippings/{args.split}.json', 'r'))
        images_root_folder = 'data/newsclippings/visual_news/origin'
        image_embeddings = json.load(open(f"data/{args.dataset}/embeddings/{args.split}/clip_L14.json"))

    else:
        data = json.load(open(f'data/5pils_ooc/test.json', 'r'))
        images_root_folder = 'data/5pils_ooc'
        image_embeddings = json.load(open(f"data/{args.dataset}/embeddings/test/clip_L14.json"))

    #Load detected objects
    detection = pd.read_csv(f'data/{args.dataset}/evidence/{args.split}/obj_detection.csv')
    
    
    nlp = spacy.load("en_core_web_lg")


    image_ids = []
    caption_ids = []
    entity_labels = []
    entity_names = []
    match_scores = []
    image_paths = []

    for item in tqdm(data):
        if args.dataset=='newsclippings':
            caption_to_verify = item['true_caption'] if 'false_caption' not in item else item['false_caption']
        else:
            caption_to_verify = item['caption']
        ner_caption = nlp(caption_to_verify)
        clip_embedding = image_embeddings[str(item['image_id'])]

        buildings = list(set([e.text for e in ner_caption.ents if e.label_ == 'FAC']))
        for b in buildings:
            url= b.replace(' ', '_')
            wiki_path = f'wikipedia_encodings/{url}'
            scores = []
            if os.path.exists(f'{wiki_path}/encodings.npy'):
                wiki_encodings = np.load(f'{wiki_path}/encodings.npy')            
                for wiki_encoding in wiki_encodings:
                    # preventing errors with saved embeddings
                    if len(wiki_encoding) != 768:
                        continue 
                    score = cosine_sim(clip_embedding, wiki_encoding)
                    scores.append(score)
                image_ids.append(item['image_id'])
                caption_ids.append(item['id'])
                entity_labels.append('FAC')
                entity_names.append(b)
                match_scores.append(scores)
                image_paths.append(os.path.join(images_root_folder, item['image_path']))

        products = list(set([e.text for e in ner_caption.ents if e.label_ == 'PRODUCT']))
        for p in products:
            url = p.replace(' ', '_')
            wiki_path = f'wikipedia_encodings/{url}'
            scores = []
            if os.path.exists(f'{wiki_path}/encodings.npy'):
                wiki_encodings = np.load(f'{wiki_path}/encodings.npy')           
                for wiki_encoding in wiki_encodings:
                    # preventing errors with with saved embeddings
                    if len(wiki_encoding) != 768:
                        continue 
                    score = cosine_sim(clip_embedding, wiki_encoding)
                    scores.append(score)
                image_ids.append(item['image_id'])
                caption_ids.append(item['id'])
                entity_labels.append('PRODUCT')
                entity_names.append(p)
                match_scores.append(scores)
                image_paths.append(os.path.join(images_root_folder, item['image_path']))      

    df = pd.DataFrame({'image_id': image_ids, 'caption_id': caption_ids, 'entity_label': entity_labels, 'entity_name': entity_names, 'match_scores': match_scores, 'image_path': image_paths})
    df.to_csv(f'data/{args.dataset}/evidence/{args.split}/buildings_products_scores.csv', index=False)