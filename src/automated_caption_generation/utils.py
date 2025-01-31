import tqdm
import numpy as np
from PIL import Image
import ast
import pandas as pd 
import torch
import os

def caption_image(instances,
                  images_root_folder, 
                  output_path,
                  model, 
                  processor, 
                  batch_size=1,  
                  max_tokens=120):
    '''
    Generate two captions for the entire image for a list of instances.
    One caption is about the people in the image, while the other is a general description of the image's content.
    Params:
        instances (list): the list of instances to caption
        images_root_folder (str): root folder containing the image files
        output_path (str): path to a csv file where the results should be saved
        model (obj): MLLM used for captioning
        processor (obj): processor of the MLLM used for captioning
        batch_size (int): number of instances to process in one batch
        max_tokens (int): max length in token of the output caption
    '''

    prompt_people = """[INST] <image>
    Answer in one sentence: who is shown in the image? Answer with names and a short biography. [/INST]"""
    prompt_general = """[INST] <image>
    Answer in one to three sentences: what are the people, objects, animals, events, texts shown in the image? Be specific. [/INST]"""
    
    for i in tqdm.tqdm(range(0, len(instances['true_caption']), batch_size)):

        # batch size one since multiple questions are asked for each image
        batch_size = 1

        if i+batch_size > len(instances['true_caption']):
            batch_size = i + batch_size - len(instances['true_caption'])

        image_path = os.path.join(images_root_folder, instances['image_path'][i])
        img = Image.open(image_path).convert('RGB')

        # provide the image twice, once for each prompt
        images = [img, img]

        people_captions = np.empty(batch_size, dtype='object')
        object_captions = np.empty(batch_size, dtype='object')

        input_texts = [prompt_general, prompt_people]

        inputs = processor(input_texts,
                           images,
                           padding=True,
                           return_tensors='pt').to(0, torch.float16)
        output_sequences = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0, top_p=1)
        outputs = processor.batch_decode(output_sequences, skip_special_tokens=True)
        
        # order of outputs: global object, global people
        object_captions[0] = outputs[0].split("[/INST]")[-1].strip().split('\n')[0]
        people_captions[0] = outputs[1].split("[/INST]")[-1].strip().split('\n')[0]

        data = pd.DataFrame({"image_id": instances['image_id'][i:i+batch_size],
                             "true_caption": instances['true_caption'][i:i+batch_size],
                             "people_caption": people_captions,
                             "object_caption": object_captions,
                             })
        
        #save results to a csv file
        data.to_csv(output_path,
                    mode="a",
                    index=False,
                    sep=",",
                    header=not os.path.exists(output_path))
                
        torch.cuda.empty_cache()
    

def caption_cropped_people(instances, 
                           images_root_folder, 
                           output_path,
                           detected_objects, 
                           model, processor, 
                           batch_size=1,  
                           min_score = 0.8, 
                           max_tokens=120):

    '''
    Generate captions for detected objects of type people.
    Params:
        instances (list): the list of instances to caption
        images_root_folder (str): root folder containing the image files
        output_path (str): path to a csv file where the results should be saved
        detected_objects (pandas.DataFrame): dataframe containing the detected objects with their bounding boxes, labels, and confidences scores
        model (obj): MLLM used for captioning
        processor (obj): processor of the MLLM used for captioning
        batch_size (int): number of instances to process in one batch
        min_score (float): minimum confidence score to caption an object
        max_tokens (int): max length in token of the output caption
    '''

    prompt_people = """[INST] <image>
    Answer in one sentence: who is shown in the image? Answer with names and a short biography. [/INST]"""

    for i in tqdm.tqdm(range(0, len(instances['true_caption']), batch_size)):

        batch_size = 1

        if i+batch_size > len(instances['true_caption']):
            batch_size = i + batch_size - len(instances['true_caption'])
        image_path = os.path.join(images_root_folder, instances['image_path'][i])
        image_id = instances['image_id'][i]

        img = Image.open(image_path).convert('RGB')
        images = []

        w, h = img.size

        for _ , item in detected_objects.iterrows():
            left = 0
            top = 0
            right = w
            bottom = h
            if item['image_id'] == image_id and item['name'] == 'Person' and item['score'] > min_score:
                bounds = ast.literal_eval(item['bounds'])
                #Testing if bounds are present, otherwise use default values
                if 'x' in bounds[0]:
                    left = bounds[0]['x']*w
                if 'y' in bounds[0]:
                    top = bounds[0]['y']*h
                if 'x' in bounds[1]:
                    right = bounds[1]['x']*w
                if 'y' in bounds[2]:
                    bottom = bounds[2]['y']*h
                
                cropped_img = img.crop((left, top, right, bottom)).convert('RGB')
                images.append(cropped_img)

        if len(images) == 0:
            continue

        cropped_people_captions = np.empty(batch_size, dtype='object')
        input_texts = [prompt_people]*(len(images))

        #Resize the images to run batch generation
        resized_images = []
        for im in images:
            resized_images.append(im.resize((w, h)))

        inputs = processor(input_texts,
                           resized_images,
                           padding=True,
                           return_tensors='pt').to(0, torch.float16)
        output_sequences = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0, top_p=1)
        outputs = processor.batch_decode(output_sequences, skip_special_tokens=True)

        #Save as one string with multiple people captions: "Person 1: [caption] Person 2: [caption] Person 3: [caption]"
        cropped_people_caption = ' '.join(["Person {}: {}".format(index+1, out.split('[/INST]')[-1].strip().split('\n')[0]) for index, out in enumerate(outputs)])
        cropped_people_captions[0] = cropped_people_caption 

        data = pd.DataFrame({"image_id": instances['image_id'][i:i+batch_size],
                             "true_caption": instances['true_caption'][i:i+batch_size],
                             "people_caption": cropped_people_captions,
                             })
        
        #save results to a csv file
        data.to_csv(output_path,
                    mode="a",
                    index=False,
                    sep=",",
                    header=not os.path.exists(output_path))
                
        torch.cuda.empty_cache()

def caption_object(instances, 
                   images_root_folder,
                   output_path, 
                   detected_objects, 
                   model, 
                   processor, 
                   batch_size=1, 
                   max_tokens=120):
    
    '''
    Generate captions  for a specific objected detected in the image.
    Params:
        instances (list): the list of instances to caption
        images_root_folder (str): root folder containing the image files
        output_path (str): path to a csv file where the results should be saved
        detected_objects (pandas.DataFrame): dataframe containing the detected objects with their bounding boxes, labels, and confidences scores
        model (obj): MLLM used for captioning
        processor (obj): processor of the MLLM used for captioning
        batch_size (int): number of instances to process in one batch
        max_tokens (int): max length in token of the output caption
    '''

    prompt_animals = """[INST] <image>
    Which {} species is shown in this image? Be as specific as possible. [/INST]"""
    prompt_transports = """[INST] <image>
    Which {} model is shown in this image? Be as specific as possible. [/INST]"""
    prompt_sports = """[INST] <image>
    What are the teams playing in this game? Be as specific as possible. [/INST]"""
    prompt_buildings = """[INST] <image>
    Which {} is shown in this image? Provide a location if possible. Be as specific as possible. [/INST]"""
    prompt_food = """[INST] <image>
    Which {} is shown in this image? Be as specific as possible. [/INST]"""
    prompt_weapons = """[INST] <image>
    Which weapon model is shown in this image? Be as specific as possible. [/INST]"""
    prompt_flags = """[INST] <image>
    Which flag is shown in this image? Be as specific as possible. [/INST]"""

    prompt_dict = {'Animals': prompt_animals, 
                   'Transports': prompt_transports, 
                   'Sports': prompt_sports, 
                   'Buildings': prompt_buildings, 
                   'Food': prompt_food, 
                   'Weapons': prompt_weapons, 
                   'Flags': prompt_flags}

    
    for i in tqdm.tqdm(range(0, len(instances['true_caption']), batch_size)):

        batch_size = 1

        if i+batch_size > len(instances['true_caption']):
            batch_size = i + batch_size - len(instances['true_caption'])

        image_path = os.path.join(images_root_folder, instances['image_path'][i])
        image_id = instances['image_id'][i]

        full_img = Image.open(image_path).convert('RGB')
        w, h = full_img.size

        category_label_images = [] # tuples of category, label and image

        for index, item in detected_objects[detected_objects['image_id'] == image_id].iterrows():
            left = 0
            top = 0
            right = w
            bottom = h

            category = ''
            label = ''

            if item['name'] in ['Animal', 'Dog', 'Cat', 'Bird', 'Fish']:
                category = 'Animals'
            elif item['name'] in ['Airplane', 'Boat', 'Car', 'Helicopter', 'Train', 'Ship', 'Tank', 'Bus', 'Van', 'Truck', 'Motorcycle']:
                category = 'Transports'
            elif item['name'] in ['Football', 'Basketball', 'Baseball glove', 'Baseball bat', 'Rugby bal']:
                category = 'Sports'
            elif item['name'] in ['Building', 'Stadium', 'Bridge', 'Castle']:
                category = 'Buildings'
            elif item['name'] in ['Food', 'Drink', 'Fruit']:
                category = 'Food'
            elif item['name'] in ['Weapon']:
                category = 'Weapons'
            elif item['name'] in ['Flag']:
                category = 'Flags'

            if category == '':
                continue
            else:
                label = item['name']

            bounds = ast.literal_eval(item['bounds'])
            #Testing if bounds are present, otherwise use default values
            if 'x' in bounds[0]:
                left = bounds[0]['x']*w
            if 'y' in bounds[0]:
                top = bounds[0]['y']*h
            if 'x' in bounds[1]:
                right = bounds[1]['x']*w
            if 'y' in bounds[2]:
                bottom = bounds[2]['y']*h
            
            cropped_img = full_img.crop((left, top, right, bottom)).convert('RGB')
            category_label_images.append((category, label, cropped_img))

        captions = []

        for obj in category_label_images:
            category = obj[0]
            label = obj[1]
            if category == 'Sports':
                #Sports category uses full image
                cropped_img = full_img
            else:
                cropped_img = obj[2]

            input_text = prompt_dict[category].format(label)

            inputs = processor(input_text,
                            cropped_img,
                            padding=True,
                            return_tensors='pt').to(0, torch.float16)
            output_sequences = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0, top_p=1)
            outputs = processor.batch_decode(output_sequences, skip_special_tokens=True)

            captions.append(outputs[0].split("[/INST]")[-1].strip().split('\n')[0])

        len_to_add = len(captions)

        data = pd.DataFrame({"image_id": [instances['image_id'][i]]*len_to_add,
                             "true_caption": [instances['true_caption'][i]]*len_to_add,
                             "category": [o[0] for o in category_label_images],
                             "label": [o[1] for o in category_label_images],
                             "vqa_answer": captions,
                             })

        #save results to a csv file
        data.to_csv(output_path,
                    mode="a",
                    index=False,
                    sep=",",
                    header=not os.path.exists(output_path))
                
        torch.cuda.empty_cache()

