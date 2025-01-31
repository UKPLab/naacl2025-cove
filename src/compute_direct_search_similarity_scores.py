import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import argparse
from utils import *


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute cosine similarity between direct search images and instance images.')
    parser.add_argument('--dataset', type=str,  default= "newsclippings", choices=['newsclippings', '5pils-ooc'],
                        help='The dataset to use.') 
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 
    parser.add_argumen('--match_threshold', type=float, default= 0.92,
                        help="Minimum threshold for direct match")
    parser.add_argumen('--non_match_threshold', type=float, default= 0.7,
                        help="Minimum threshold for direct non match")

    args = parser.parse_args()

    #Load data
    if args.dataset=='newsclippings':
        data = json.load(open(f'data/newsclippings/{args.split}.json', 'r'))
        images_root_folder = 'data/newsclippings/visual_news/origin'
        image_embeddings = json.load(open(f"data/{args.dataset}/embeddings/{args.split}/clip_L14.json"))
        direct_search_file = json.load(open(f"data/{args.dataset}/evidence/direct_search/{args.split}/{args.split}.json", "r"))
    else:
        data = json.load(open(f'data/5pils_ooc/test.json', 'r'))
        images_root_folder = 'data/5pils_ooc'
        image_embeddings = json.load(open(f"data/{args.dataset}/embeddings/test/clip_L14.json"))
        direct_search_file = json.load(open(f"data/5pils_ooc/evidence/direct_search/trafilatura.json", ""))

    veracity = []
    clip_sims = []
    dir_matches = []
    caption_ids = []
    img_ids = []
    dir_search_captions = []
    captions = []

    for item in tqdm(data):
        image_id = item['image_id']
        caption_id = item['id']
        if args.dataset== 'newsclippings':
            caption_to_verify = item['true_caption'] if 'false_caption' not in item else item['false_caption']
        else:
            caption_to_verify = item['caption']
        clip_embedding = image_embeddings[str(image_id)]

        if args.dataset=='newsclippings':
            file_id = 0
            found_file = False
            for k, v in direct_search_file.items():
                #Checking for caption id since it is direct search
                if v['text_id_in_visualNews'] == caption_id:
                    file_id = k
                    found_file = True
                    break
            
            if not found_file:
                print('file not found', caption_id)
                continue
            
            direct_search_dir = f"data/{args.dataset}/evidence/direct_search/{args.split}/{file_id}/"
            annotation = json.load(open(f"data/{args.dataset}/evidence/direct_search/{args.split}/{file_id}/direct_annotation.json", "r"))
        
            direct_search_evidence_clip_embeddings = np.load(os.path.join(direct_search_dir, 'clip_L14_embeddings.npy'))
            captions_to_add = []
            for idx, evidence_img in enumerate(annotation['images_with_captions']):
                evidence_clip_embedding = direct_search_evidence_clip_embeddings[idx]
                clip_sim = cosine_sim(clip_embedding, evidence_clip_embedding)

                clip_match = True if clip_sim > args.match_threshold else False
                clip_non_match = True if clip_sim < args.non_match_threshold else False

                #Direct string match verification
                non_alphanum_pattern = re.compile('[\W_]+')
                is_direct_match = False

                evidence_caption = list(evidence_img['caption'].items())[0][1]
                #Lowercase and remove punctuation and whitespaces
                caption_to_search = caption_to_verify.lower()
                evidence_to_test = evidence_caption.lower()
                caption_to_search = re.sub(non_alphanum_pattern, '', caption_to_search)
                evidence_to_test = re.sub(non_alphanum_pattern, '', evidence_to_test)
                if re.search(caption_to_search, evidence_to_test):
                    is_direct_match = True

                veracity.append('pristine' if 'false_caption' not in item else 'falsified')
                clip_sims.append(clip_sim)
                dir_matches.append(is_direct_match)
                caption_ids.append(caption_id)
                img_ids.append(image_id)
                dir_search_captions.append(evidence_caption.replace('\n', ' ').replace('\r', ' '))
                captions.append(caption_to_verify)

        else:
            #5Pils OOC
            try:
                #Caption id since it is direct search
                direct_search_evidence_clip_embeddings = json.load(open("data/5pils_ooc/evidence/direct_search/clip_L14_direct_search.json"))[caption_id]
            except KeyError:
                continue

            dir_search_results = [x for x in direct_search_file if x['claim'] == caption_to_verify]

            for ev_ix, evidence_embedding in enumerate(direct_search_evidence_clip_embeddings):
                clip_sim = cosine_sim(clip_embedding, evidence_embedding)
                evidence_caption = dir_search_results[ev_ix]['image caption'].replace('\n', ' ').replace('\r', ' ')
                #Lowercase and remove punctuation and whitespaces
                is_direct_match = False
                caption_to_search = caption_to_verify.lower()
                evidence_to_test = evidence_caption.lower()
                caption_to_search = re.sub(non_alphanum_pattern, '', caption_to_search)
                evidence_to_test = re.sub(non_alphanum_pattern, '', evidence_to_test)
                if re.search(caption_to_search, evidence_to_test):
                    is_direct_match = True

                veracity.append('pristine' if caption_id == image_id else 'falsified')
                clip_sims.append(clip_sim)
                dir_matches.append(is_direct_match)
                img_ids.append(image_id)
                caption_ids.append(caption_id)
                dir_search_captions.append(evidence_caption)
                captions.append(caption_to_verify)


    data = pd.DataFrame({'image_id': img_ids, 'caption': captions, 'veracity': veracity, 'clip_sim': clip_sims, 'is_string_match': dir_matches, 'dir_search_caption': dir_search_captions})
    data.to_csv(f'data/{args.dataset}/evidence/{args.split}/direct_search_clip_L14_similarities.csv', index=False)