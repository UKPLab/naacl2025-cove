import json
import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import face_recognition
from tqdm import tqdm
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute image embeddings of image evidence.')
    parser.add_argument('--dataset', type=str,  default= "newsclippings", choices=['newsclippings', '5pils-ooc'],
                        help='The dataset to use.') 
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 

    args = parser.parse_args()

    os.makedirs(f'data/{args.dataset}/embeddings', exist_ok=True)

    #Load data
    if args.dataset=='newsclippings':
        data = json.load(open(f'data/newsclippings/{args.split}.json', 'r'))
        images_root_folder = 'data/newsclippings/visual_news/origin'
        direct_search_file = json.load(open(f"data/{args.dataset}/evidence/direct_search/{args.split}/{args.split}.json", "r"))
    else:
        data = json.load(open(f'data/5pils_ooc/test.json', 'r'))
        images_root_folder = 'data/5pils_ooc'
        direct_search_file = json.load(open(f"data/5pils_ooc/evidence/direct_search/trafilatura.json", ""))

    #Load CLIP model
    model = SentenceTransformer("clip-ViT-L-14")


    
    clip_image_embeddings = {}
    face_image_embeddings = {}
    for item in tqdm(data):
        image_id = item['image_id']
        caption_id = item['id']
        image_path = os.path.join(images_root_folder, item['image_path'])
        if args.dataset=='newsclippings':
            file_id = 0 # map to the correct direct search directory
            found_file = False
            for k, v in direct_search_file.items():
                # checking for caption id since it's direct search
                if v['text_id_in_visualNews'] == caption_id:
                    file_id = k
                    found_file = True
                    break
            if not found_file:
                print('file not found', caption_id)
                continue

            direct_search_dir = f"data/{args.dataset}/evidence/direct_search/{args.split}/{file_id}/"
            annotation = json.load(open(f"data/{args.dataset}/evidence/direct_search/{args.split}/{file_id}/direct_annotation.json", "r"))
            # embeddings for dir search images
            clip_image_evidence_embeddings = []

            for evidence_img in annotation['images_with_captions']:
                #Loop through all direct search results for that instance
                evidence_image_path = os.path.join(direct_search_dir, evidence_img['image_path'].split('/')[-1])
                emb = model.encode(Image.open(evidence_image_path).convert('RGB'))
                clip_image_evidence_embeddings.append(emb)

            #In each direct search directory, we save the embeddings of the direct search images that have captions
            np.save(f'{direct_search_dir}/clip_L14_embeddings.npy', np.array(clip_image_evidence_embeddings))

        else:
            #5Pils-OOC direct search images
            dir_search_dict = {}
            for i in tqdm(range(len(direct_search_file))):
                img_path = os.path.join('data/5pils_ooc/evidence/direct_search', f"{i}.png")
                claim = direct_search_file[i]['claim']

                pils_item = [x for x in data if x['caption'] == claim][0]
                pils_caption_id = pils_item['id']

                if pils_caption_id not in dir_search_dict.keys():
                    #Create a new list
                    dir_search_dict[pils_caption_id] = []

                try:
                    image = Image.open(img_path).convert('RGB')
                except:
                    print(f"Image {i} not found")
                    continue
                evidence_clip_embedding = model.encode(image).tolist()
                dir_search_dict[pils_caption_id].append(evidence_clip_embedding)
            #For 5Pils OOC, all embeddings are contained in a single json file
            json.dump(dir_search_dict, open("data/5pils_ooc/evidence/direct_search/clip_L14_direct_search.json", "w"))


        #Encode the image instance with CLIP
        if image_id not in clip_image_embeddings.keys():
            #the image has not been encoded yet
            clip_image_embeddings[image_id] = model.encode(Image.open(image_path).convert('RGB'))


        #Encode the faces in the image (list of face encodings)
            if image_id not in face_image_embeddings.keys():
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                if face_locations == []:
                    face_image_embeddings[image_id] = []
            else:
                face_encodings = face_recognition.face_encodings(image, face_locations)
                face_image_embeddings[image_id] = [x.tolist() for x in face_encodings]

    # save the embeddings of the images in a JSON file
    for key, value in clip_image_embeddings.items():
        clip_image_embeddings[key] = value.tolist()
    json.dump(clip_image_embeddings, open(f"data/{args.dataset}/embeddings/{args.split}/clip_L14.json", "w"))
    #save the face encodings
    json.dump(face_image_embeddings, open(f'data/{args.dataset}/embeddings/{args.split}/face_embeddings.json', 'w'))

