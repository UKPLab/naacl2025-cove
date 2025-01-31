#Prepare all evidence and group them in one csv file
import json
import pandas as pd
import argparse
from tqdm import tqdm
import re
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from utils import *
from wikipedia_entity_collection.utils import *


def matching_data(row, t_clip=0.92):
    #Check whether the instance image and the evidence image are matching based on CLIP threshold
    return row['clip_sim'] >= t_clip

def non_matching_data(row, t_clip=0.7):
    #Check whether the instance image and the evidence image are not matching based on CLIP threshold and caption overlap
    if row['is_string_match']:
        #Only if the web caption is matching the claim (string match) but the image is very different
        return row['clip_sim'] <= t_clip
    else:
        return False


if __name__=='__main__':
    parser = argparse.ArgumentParser("Prepare input dataset for context prediction, grouping all evidence.")
    parser.add_argument('--dataset', type=str,  default= "newsclippings", choices=['newsclippings', '5pils-ooc'],
                        help='The dataset to use.') 
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 
    parser.add_argument('--batch_size', type=int,  default= 128,
                        help='Number of instances to process in each batch')
    parser.add_argument('--vis_entity_t', type=float,  default= 0.1,
                        help='Score to threshold to include visual entities detected by Google Vision')
    parser.add_argument('--direct_match_t', type=float,  default= 0.92,
                        help='Image similarity threshold to detect a direct match with the evidence')
    parser.add_argument('--direct_non_match_t', type=float,  default= 0.7,
                        help='Image similarity threshold to detect a direct non-match with the evidence')
    parser.add_argument('--wiki_image_object_t', type=float,  default= 0.7,
                        help='Image similarity threshold to include Wikipedia entities that are not people')
    parser.add_argument('--wiki_image_people_t', type=float,  default= 0.92,
                        help='Image similarity threshold to include Wikipedia entities that are people')
    parser.add_argument('--wiki_text_t', type=float,  default= 0.23,
                        help='Text similarity threshold  to include Wikipedia entities')
    
    
    args = parser.parse_args()

    #Load data
    if args.dataset=='newsclippings':
        data = json.load(open(f'data/newsclippings/{args.split}.json', 'r'))
        images_root_folder = 'data/newsclippings/visual_news/origin'
    else:
        data = json.load(open(f'data/5pils_ooc/test.json', 'r'))
        images_root_folder = 'data/5pils_ooc'

    #Load predicted captions for the entire image
    predicted_captions = pd.read_csv(f'data/{args.dataset}/evidence/{args.split}/automated_captions/general_captions.csv')
    

    instances = []
    for ix in range(len(data)):

        found_captions = False
        instance_dict = {}

        for index, item in predicted_captions.iterrows():
            if item['image_id'] == data[ix]['image_id']:
                instance_dict['people_caption'] = item['people_caption']
                instance_dict['object_caption'] = item['object_caption']
                found_captions = True
                break

        if not found_captions:
            print('not found: ', data[ix]['image_id'])
            continue
        
        instance_dict['image_id'] = data[ix]['image_id']
        instance_dict['caption_id'] = data[ix]['id']
        if args.dataset=='newsclippings':
            instance_dict['caption'] = data[ix]['false_caption'] if 'false_caption' in data[ix] else data[ix]['true_caption']
        else:
            instance_dict['caption']  = data[ix]['caption']
        instance_dict['true_caption'] = data[ix]['true_caption']
        instance_dict['image_path'] = os.path.join('./visual_news/origin/', data[ix]['image_path'])
        instance_dict['true_veracity'] = 'falsified' if 'false_caption' in data[ix] else 'pristine'
        if 'evidence_captions' in data[ix] and data[ix]['evidence_captions'] != []:
            instance_dict['evidence'] = data[ix]['evidence_captions']
        else:
            instance_dict['evidence'] = []

        # adding visual entities with a sufficient score
        entities = []
        for ent, score in zip(data[ix]['vis_entities'], data[ix]['scores_vis_entities']):
            if score > args.vis_entity_t:
                entities.append(ent)
        instance_dict['vis_entities'] = ", ".join(entities)

        instances.append(instance_dict)


    #Load spacy mode
    nlp = spacy.load("en_core_web_lg")
    #Load GENRE model
    tokenizer_genre = AutoTokenizer.from_pretrained("facebook/genre-linking-blink")
    model_genre = AutoModelForSeq2SeqLM.from_pretrained("facebook/genre-linking-blink").eval()
    #Load CLIP model
    clip_model = SentenceTransformer("clip-ViT-L-14")

    #Load all the evidence file
    cropped_people_captions = pd.read_csv(f'data/{args.dataset}/evidence/{args.split}/automated_captions/cropped_people_captions.csv')
    face_embeddings = json.load(open(f'data/{args.dataset}/embeddings/{args.split}/face_embeddings.json', 'r'))
    direct_search_scores = pd.read_csv(f'data/{args.dataset}/evidence/{args.split}/direct_search_clip_L14_similarities.csv')
    vqa_detected_objects = pd.read_csv(f'data/{args.dataset}/evidence/{args.split}/automated_captions/object_captions.csv')
    wiki_buildings_products = pd.read_csv(f'data/{args.dataset}/evidence/{args.split}/buildings_products_scores.csv')
    clip_embeddings = json.load(open(f'data/{args.dataset}/embeddings/{args.split}/clip_L14.json', 'r'))
    oven_entities = json.load(open(f'data/{args.dataset}/evidence/{args.split}/oven_entities.json', 'r'))


    for i in tqdm.tqdm(range(0, len(instances), args.batch_size)):

        if i+batch_size > len(instances):
            #Final batch length calculation
            batch_size = len(instances) - i

        veracity = np.empty(batch_size, dtype='object')
        predicted_contexts = np.empty(batch_size, dtype='object')
        original_veracity_outputs = np.empty(batch_size, dtype='object')
        wiki_clip_results = np.empty(batch_size, dtype='object')
        is_direct_search_match = np.empty(batch_size, dtype='object')
        is_direct_search_non_match = np.empty(batch_size, dtype='object')
        max_dir_search_sim = np.empty(batch_size, dtype='object')
        min_dir_search_sim = np.empty(batch_size, dtype='object')
        people_captions_list = np.empty(batch_size, dtype='object')
        obj_captions_list = np.empty(batch_size, dtype='object')


        for batch_index in range(batch_size):
            #Process the batch
            sample = instances[i+batch_index]
            if args.dataset== 'newsclippings':
                caption_to_verify = sample['true_caption'] if 'false_caption' not in sample else sample['false_caption']
            else:
                caption_to_verify = sample['caption']
            image_id = sample['image_id']
            caption_id = sample['caption_id']
            img_rel_path = sample['image_path']
            vis_entities = sample['vis_entities']

            #Get oven entities from the image
            oven_entities_img = [item for item in oven_entities if list(item.keys())[0] == img_rel_path]
            if len(oven_entities_img) > 0:
                oven_entities_img = oven_entities_img[0][img_rel_path]

            nlp_caption = nlp(caption_to_verify)
            #List of people detected in the caption
            NER_people = list(set([e.text for e in nlp_caption.ents if e.label_ == 'PERSON']))

            #Add OVEN people to NER people for verifying wiki face matches
            for e in oven_entities_img:
                nlp_oven = nlp(e)
                if len([ent for ent in nlp_oven.ents if ent.label_ == 'PERSON']) != 0:
                    NER_people.append(e)

            face_encodings = face_embeddings[str(image_id)]   
            object_caption = sample['object_caption']

            # Append wiki buildings, products to object caption
            wiki_b_p = wiki_buildings_products[(wiki_buildings_products['image_id'] == image_id) & (wiki_buildings_products['caption_id'] == 'caption_id')]
            for row_ix, row in wiki_b_p.iterrows():
                if any(row['match_scores'] >= args.wiki_image_object_t): 
                    object_caption += f" The image shows {row['entity_name']}."
            # Append oven non-people entities to object caption
            for e in oven_entities_img:
                url = e.replace(' ', '_')
                wiki_oven_path = f'wikipedia_encodings/{url}/encodings.npy'
                if os.path.exists(wiki_oven_path):
                    oven_encodings = np.load(wiki_oven_path)
                    # 768 means clip embedding
                    if len(oven_encodings) > 0 and len(oven_encodings[0]) == 768:
                        img_clip_embedding = clip_embeddings[str(image_id)]
                        if any(cosine_sim(img_clip_embedding, encoding) >= args.wiki_image_object_t for encoding in oven_encodings):
                            object_caption += f" The image shows {e}."

            # if it is a direct string match, save direct match caption (from reverse search)
            direct_match_captions = []

            #Veracity rules: check for direct matches
            non_alphanum_pattern = re.compile('[\W_]+')
            is_direct_match = False
            if sample['evidence'] != []:
                evidence_captions = sample['evidence']
                for caption in evidence_captions:
                    # lowercase and remove punctuation and whitespaces
                    caption_to_search = caption_to_verify.lower()
                    evidence_to_test = caption.lower()
                    caption_to_search = re.sub(non_alphanum_pattern, '', caption_to_search)
                    evidence_to_test = re.sub(non_alphanum_pattern, '', evidence_to_test)
                    if re.search(caption_to_search, evidence_to_test):
                        veracity[batch_index] = 'pristine'
                        original_veracity_outputs[batch_index] = "Direct match found in evidence."
                        wiki_clip_results[batch_index] = []
                        is_direct_match = True
                        direct_match_captions.append(caption)
                        break
        
            # Adding direct search matches and non-matches
            if is_direct_match:
                is_direct_search_match[batch_index] = False
                is_direct_search_non_match[batch_index] = False
                max_dir_search_sim[batch_index] = np.nan
                min_dir_search_sim[batch_index] = np.nan
            else:
                # use image id + caption as key
                direct_search_rows = direct_search_scores[(direct_search_scores['image_id'] == image_id) & (direct_search_scores['caption'] == caption_to_verify)]
                max_dir_search_sim[batch_index] = direct_search_rows['clip_sim'].max()
                min_dir_search_sim[batch_index] = direct_search_rows['clip_sim'].min()
                for row_ix, row in direct_search_rows.iterrows():
                    if matching_data(row, args.direct_match_t):
                        is_direct_search_match[batch_index] = True
                        is_direct_search_non_match[batch_index] = False
                        break
                if is_direct_search_match[batch_index] == None:
                    for row_ix, row in direct_search_rows.iterrows():
                        if non_matching_data(row, args.direct_non_match_t):
                            is_direct_search_match[batch_index] = False
                            is_direct_search_non_match[batch_index] = True
                            break
                        else:
                            is_direct_search_match[batch_index] = False
                            is_direct_search_non_match[batch_index] = False

            if not is_direct_match: 
                #If no direct match, add Wikipedia evidence
                # list of matched people within the caption
                wiki_clip_people = []
                # iterate over NER people in the caption
                for named_person in NER_people:
                    # iterate over detected people in the image
                    wiki_match = False
                    person_name_url = named_person.replace(' ', '_')
                    wiki_path = f'wikipedia_encodings/{person_name_url}/encodings.npy'

                    # trying to find saved wiki image
                    if os.path.exists(wiki_path):
                        wiki_encodings = np.load(wiki_path)
                        # use only images with faces (OVEN entities are included) - 128 is the dim of face encoding
                        if len(wiki_encodings) >  0 and len(wiki_encodings[0]) == 128:
                            match_scores_person = []
                            for face_encoding in face_encodings:                
                                for wiki_encoding in wiki_encodings:
                                    match_score = cosine_sim(face_encoding, wiki_encoding)
                                    match_scores_person.append(match_score)
                            # Wiki face threshold
                            if any(score >= args.wiki_image_people_t for score in match_scores_person):
                                wiki_clip_people.append((named_person, 'wiki_image'))
                                wiki_match = True
                        
                    if not wiki_match:
                        #If match based on Wikipedia image similarity is not successful, try based on text
                        tagged_string = '[START_ENT] ' + named_person + ' [END_ENT]'
                        disambiguated_person = genre_entity_linking(tagged_string)[0] 
                        clip_entity = clip_model.encode(disambiguated_person)
                        img_clip_embedding = clip_embeddings[str(image_id)]
                        if cosine_sim(clip_entity, img_clip_embedding) >= args.wiki_text_t:
                            wiki_clip_people.append((named_person, 'wiki_text'))
                        else: # else, search in cropped people captions
                            cropped_people = cropped_people_captions[cropped_people_captions['image_id'] == image_id]['people_caption']
                            # check if there are any cropped people captions
                            if len(cropped_people) > 0:
                                cropped_people = cropped_people.iloc[0]
                            if type(cropped_people) == str: # nan cases out
                                cropped_people_list = re.sub(r'Person [0-9]+: ', '/sep', cropped_people).split('/sep')
                                for cropped_person in cropped_people_list:
                                    if re.search(named_person.lower(), cropped_person.lower()):
                                        wiki_clip_people.append((cropped_person, 'cropped_caption'))
            
                # use wiki/clip matches if at least one person is matched
                if len(wiki_clip_people) > 0:
                    people_caption = 'The image shows ' + ', '.join([p for p, _ in wiki_clip_people])
                else:
                    people_caption = sample['people_caption']

                wiki_clip_results[batch_index] = wiki_clip_people
            
            else:
                # direct string match, no need to check wiki/clip
                people_caption = sample['people_caption']

            #Store people and object captions
            people_captions_list[batch_index] = people_caption
            obj_captions_list[batch_index] = object_caption
        

    instances = pd.DataFrame({"image_id": [item['image_id'] for item in instances[i:i+batch_size]],
                        "caption_id": [item['caption_id'] for item in instances[i:i+batch_size]],
                        "caption": [item['caption'] for item in instances[i:i+batch_size]],
                        "true_caption": [item['true_caption'] for item in instances[i:i+batch_size]],
                        "true_veracity": [item['true_veracity'] for item in instances[i:i+batch_size]],
                        "people_caption": people_captions_list,
                        "object_caption": obj_captions_list,
                        "evidence_captions": [item['evidence'] for item in instances[i:i+batch_size]],
                        "vis_entities": [item['vis_entities'] for item in instances[i:i+batch_size]],
                        "wiki_clip": wiki_clip_results,
                        "direct_search_match": is_direct_search_match,
                        "direct_search_non_match": is_direct_search_non_match,
                        "max_direct_search_sim": max_dir_search_sim,
                        "min_direct_search_sim": min_dir_search_sim,
                        "image_path": [item['image_path'] for item in instances[i:i+batch_size]],})

    output_path = f"{args.dataset}/context_input_{args.dataset}_{args.split}.csv"
    instances.to_csv(output_path,
                mode="a",
                index=False,
                sep=",",
                header=not os.path.exists(output_path))