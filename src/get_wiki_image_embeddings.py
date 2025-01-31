import json
import numpy as np
import os
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import argparse
from wikipedia_entity_collection.utils import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute Wikipedia image embeddings')
    parser.add_argument('--model_path', type=str,  default= "facebook/genre-linking-blink", 
                        help='path to the entity linking mode.') 
    parser.add_argument('--dataset', type=str,  default= "newsclippings", choices=['newsclippings', '5pils-ooc'],
                        help='The dataset to use.') 
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 
    parser.add_argumen('--get_oven_embeddings', type=int, default=1, choices=[0,1],
                       help='If 1, assumes there is a list of oven entities available for that dataset split to be encoded.')
    

    args = parser.parse_args()

    #Load data
    if args.dataset=='newsclippings':
        data = json.load(open(f'data/newsclippings/{args.split}.json', 'r'))
        images_root_folder = 'data/newsclippings/visual_news/origin'
    else:
        data = json.load(open(f'data/5pils_ooc/test.json', 'r'))
        images_root_folder = 'data/5pils_ooc'
    

    #Load spacy model
    nlp = spacy.load("en_core_web_lg")
    #Load entity linking model
    genre_tokenizer = AutoTokenizer.from_pretrained("facebook/genre-linking-blink")
    genre_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/genre-linking-blink").eval()
    #Load CLIP model
    clip_model = SentenceTransformer("clip-ViT-L-14")

    #Make directory for encodings
    os.makedirs('wikipedia_encodings', exist_ok=True)

    for item in tqdm(data):
        if args.dataset=='newsclippings':
            if 'false_caption' in item:
                caption_to_verify = item['false_caption']
            else:
                caption_to_verify = item['true_caption']
        else:
            #5Pils-OOC
            caption_to_verify = item['caption']


        doc = nlp(caption_to_verify)
        
        if len([e for e in doc.ents if e.label_ in ['FAC', 'PRODUCT', 'PERSON']]) != 0:
            #There is at least one entity

            #Keep all unique entities
            entities = list(set([e for e in doc.ents if e.label_ in ['FAC', 'PRODUCT', 'PERSON']]))
            entity_names = [e.text for e in entities]
            entity_labels = [e.label_ for e in entities]

            disambiguation_list = []
            for entity in entities:
                #Entity disambiguation with GENRE
                url = entity.text.replace(' ', '_')
                wiki_path = f'wikipedia_encodings/{url}'
                # if already saved encodings, add empty list and skip
                if os.path.exists(wiki_path):
                    if os.listdir(wiki_path) != []:
                        print('Directory already exists.')
                        disambiguation_list.append([])
                        continue
                #Prepare input text for entity disambiguation
                start = entity.start_char
                end = entity.end_char
                str_list = list(caption_to_verify)
                str_list[start] = '[START_ENT] ' + str_list[start]
                str_list[end-1] = str_list[end-1] + ' [END_ENT]'
                tagged_caption = ''.join(str_list)
                disambiguation_list.append(genre_entity_linking(tagged_caption, genre_model, genre_tokenizer))

            for entity_number,  entity in enumerate(entity_names):
                url = entity.replace(' ', '_')
                wiki_path = f'wikipedia_encodings/{url}'

                # if already saved encodings, skip
                if os.path.exists(wiki_path):
                    if os.listdir(wiki_path) != []:
                        print('Directory already contains data.')
                        continue
                
                for disamb_res in disambiguation_list[entity_number]:
                    #Collect the wikipedia image
                    wiki_images = get_wikipedia_images(disamb_res)
                    if len(wiki_images) == 0:
                        print(f'No images found for {disamb_res}')
                        continue
                    else: # if at least one image is found, stop
                        break

                if len(wiki_images) == 0:
                        print(f'No images found in all pages for {entity}.')
                        continue
                
                os.makedirs(wiki_path, exist_ok=True)

                all_encodings = []
                if entity_labels[entity_number]=='PERSON':
                    for _ , img in enumerate(wiki_images):
                        #Compute face embeddings for every image
                        all_encodings += get_face_encodings(img)
                else:
                    #FAC OR PRODUCT --> CLIP embeddings
                    for _ , result in enumerate(wiki_images):
                    # list so encodings are saved correctly
                        clip_embedding = clip_model.encode(result)
                        all_encodings.append(list(clip_embedding))

                # save a numpy file with all face encodings
                all_encodings = np.array(all_encodings)
                np.save(f'{wiki_path}/encodings.npy', all_encodings)
                print('Saved encodings.')


    #OVEN entities
    if args.get_oven_embeddings:
        try:
            oven_entities = json.load(open(f'data/{args.dataset}/evidence/{args.split}/oven_entities.json'))
        except:
            raise ValueError('OVEN entities file does not exist.')

        for item in tqdm(oven_entities):

            entities_to_check = item[list(item.keys())[0]]

            for doc in nlp.pipe(entities_to_check, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
                entity = doc.text
                is_person = False
                # verify if entity is a person
                if len([e for e in doc.ents if e.label_ == 'PERSON']) != 0:
                    #Person are encoded with face embeddings, the other entities with CLIP embeddings
                    is_person = True
                entity_name_url = entity.replace(' ', '_')
                wiki_path = f'wikipedia_encodings/{entity_name_url}'

                # if already saved encodings, skip
                if os.path.exists(f'{wiki_path}/encodings.npy'):
                    wiki_encodings = np.load(f'{wiki_path}/encodings.npy')
                    if len(wiki_encodings) > 0 and len(wiki_encodings) <= 10:
                        print('Encodings already saved.')
                        continue
                
                wiki_images = get_wikipedia_images(entity)
                if len(wiki_images) == 0:
                    print('No images found.')
                    continue
                
                if not os.path.exists(wiki_path):
                    os.makedirs(wiki_path)

                all_encodings = []

                if is_person:
                    # save face encodings for person entities
                    for ix, img in enumerate(wiki_images):
                        all_encodings += get_face_encodings(img)
                else:
                    # for other entities, save CLIP encodings
                    for ix, img in enumerate(wiki_images):
                        all_encodings.append(list(clip_model.encode(img)))

                all_encodings = np.array(all_encodings)
                np.save(f'{wiki_path}/encodings.npy', all_encodings)