import pandas as pd
import json
import os
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import argparse
from automated_caption_generation.utils import * 


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate automated captions for images and detected objects')
    parser.add_argument('--model_path', type=str,  default= "llava-hf/llava-v1.6-mistral-7b-hf", 
                        help='path to the MLLM used for image captioning.') 
    parser.add_argument('--dataset', type=str,  default= "newsclippings", choices=['newsclippings', '5pils-ooc'],
                        help='The dataset to use.') 
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 

    args = parser.parse_args()

    #Load model and processor
    model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto").to('cuda')
    processor = LlavaNextProcessor.from_pretrained(args.model_path)

    #Load data
    if args.dataset=='newsclippings':
        data = json.load(open(f'data/newsclippings/{args.split}.json', 'r'))
        images_root_folder = 'data/newsclippings/visual_news/origin'
    else:
        data = json.load(open(f'data/5pils_ooc/test.json', 'r'))
        images_root_folder = 'data/5pils_ooc'

    #Load detected objects
    detected_objects = pd.read_csv(f'data/{args.dataset}/evidence/{args.split}/obj_detection.csv')


    #Define output path
    os.makedirs(f'data/{args.dataset}/evidence/',exist_ok=True)
    os.makedirs(f'data/{args.dataset}/evidence/{args.split}',exist_ok=True)
    os.makedirs(f'data/{args.dataset}/evidence/{args.split}/automated_captions/',exist_ok=True)

    image_ids = []
    true_captions = []
    image_paths = []
    for item in data:
        if item['image_id'] not in image_ids:
            true_captions.append(item['true_caption'])
            image_ids.append(item['image_id'])
            image_paths.append(item['image_path'])

    instances = {'image_id': image_ids, 'true_caption': true_captions, 'image_path': image_paths}

    #Obtain the two captions over the full image
    output_path = f'data/{args.dataset}/evidence/{args.split}/automated_captions/general_captions.csv'
    caption_image(instances,
                  images_root_folder,
                  output_path,
                  model, 
                  processor, 
                  batch_size=1)
    #Obtain the captions for the cropped people
    output_path_cropped_people = f'data/{args.dataset}/evidence/{args.split}/automated_captions/cropped_people_captions.csv'
    caption_cropped_people(instances,
                           images_root_folder, 
                           output_path_cropped_people,
                           detected_objects, 
                           model, 
                           processor, 
                           batch_size=1)
    #Obtain the captions for all other cropped objects
    output_path_object =  f'data/{args.dataset}/evidence/{args.split}/automated_captions/object_captions.csv'
    caption_object(instances,   
                   images_root_folder,
                   output_path_object, 
                   detected_objects, 
                   model, 
                   processor, 
                   batch_size=1)