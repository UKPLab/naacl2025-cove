import os
import json
import argparse
import random


def create_data_with_inverse_search_results(split, path_to_inverse_search_file):
    if split in ['val', 'test']:
        path = f"{path_to_inverse_search_file}/{split}"
        entities_dict = {}
        index_dict = json.load(open(f"{path_to_inverse_search_file}/{split}/{split}.json", "r"))
        caption_dict = {}
        for dir in os.listdir(path):
            ix = dir
            full_path = os.path.join(path, dir)
            captions_path = os.path.join(full_path, 'captions_info')
            keep_idx_path = os.path.join(full_path, 'captions_to_keep_idx')
            if os.path.exists(captions_path):
                with open(captions_path, 'r') as f:
                    captions = json.loads(f.readlines()[0])
                with open(keep_idx_path, 'r') as f:
                    keep_idx = json.loads(f.readlines()[0])
                captions.pop('domains', None)
                captions.update(keep_idx)
                caption_dict[ix] = captions
            try:
                image_id = index_dict[ix]['image_id_in_visualNews']
            except:
                continue
            entities = json.load(open(os.path.join(full_path, 'inverse_annotation.json'), 'r'))['entities']
            scores = json.load(open(os.path.join(full_path, 'inverse_annotation.json'), 'r'))['entities_scores']
            entities_dict[image_id] = (entities, scores)

        caption_dict = dict(sorted(caption_dict.items(), key=lambda item: int(item[0])))

        data = json.load(open(f'data/newsclippings/{split}.json', 'r'))
        for item in data:
            image_id = item['image_id']

            file_id = 0 # map to the corect inverse search directory
            found_file = False
            for k, v in index_dict.items():
                # checking for image id since it's inverse search
                if v['image_id_in_visualNews'] == image_id:
                    file_id = str(k)
                    found_file = True
                    break
            if found_file:
                if file_id in caption_dict:
                    captions_to_add = caption_dict[file_id]
                    item['evidence_captions'] = captions_to_add['captions']
                    item['caption_ix_to_keep'] = captions_to_add['index_of_captions_tokeep']
                image_as_str = str(image_id) # map directly with image id
                if  image_as_str in entities_dict:
                    vis_entities_to_add = entities_dict[image_as_str]
                    item['vis_entities'] = vis_entities_to_add[0]
                    item['scores_vis_entities'] = vis_entities_to_add[1]
    else:
        #Train has a different file structure
        paths= [f"{path_to_inverse_search_file}/{split}_pt{i}/train" for i in range(1, 7)]
        entities_dict = {}
        caption_dict = {}
        index_dicts = [json.load(open(f"{path_to_inverse_search_file}/{split}_pt{i}/train/train.json", 'r')) for i in range(1, 7)]

        for dir, index_dict in zip(paths, index_dicts):
            for subdir in os.listdir(dir):
                ix = subdir
                full_path = os.path.join(dir, subdir)
                captions_path = os.path.join(full_path, 'captions_info')
                keep_idx_path = os.path.join(full_path, 'captions_to_keep_idx')
                if os.path.exists(captions_path):
                    with open(captions_path, 'r') as f:
                        captions = json.loads(f.readlines()[0])
                    with open(keep_idx_path, 'r') as f:
                        keep_idx = json.loads(f.readlines()[0])
                    captions.pop('domains', None)
                    captions.update(keep_idx)
                    caption_dict[ix] = captions
                try:
                    image_id = index_dict[ix]['image_id_in_visualNews']
                except:
                    continue
                entities = json.load(open(os.path.join(full_path, 'inverse_annotation.json'), 'r'))['entities']
                scores = json.load(open(os.path.join(full_path, 'inverse_annotation.json'), 'r'))['entities_scores']
                entities_dict[image_id] = (entities, scores)

        caption_dict = dict(sorted(caption_dict.items(), key=lambda item: int(item[0])))
    
        data = json.load(open(f'data/newsclippings/{split}.json', 'r'))
        for item in data:
            image_id = item['image_id']
            
            file_id = 0 #Map to the corect inverse search directory
            found_file = False

            for mapping in index_dicts:
                for k, v in mapping.items():
                    #Checking for image id since it is inverse search
                    if v['image_id_in_visualNews'] == image_id:
                        file_id = str(k)
                        found_file = True
                        break
                if found_file:
                    break
                
            if found_file:
                if file_id in caption_dict:
                    captions_to_add = caption_dict[file_id]
                    item['evidence_captions'] = captions_to_add['captions']
                    item['caption_ix_to_keep'] = captions_to_add['index_of_captions_tokeep']

                image_as_str = str(image_id) #Map directly with image id
                if  image_as_str in entities_dict:
                    vis_entities_to_add = entities_dict[image_as_str]
                    item['vis_entities'] = vis_entities_to_add[0]
                    item['scores_vis_entities'] = vis_entities_to_add[1]
                
    #Save all files
    json.dump(data, open(f'data/newsclippings/{split}.json', 'w'))
    with open(f'data/newsclippings/evidence/{split}/web_captions.json', 'w') as f:
        json.dump(caption_dict, f)
    with open(f'data/newsclippings/evidence/{split}/vis_entities.json', 'w') as f:
        json.dump(entities_dict, f)





if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Prepare the inverse image search evidence files for newsclippings')
    parser.add_argument('--path_to_inverse_search', type=str,  required=True, 
                        help='Path to the directory containing the inverse search results.') 
    
    args = parser.parse_args()

    random.seed(42) #Set seed for sampling from train and val sets

    os.makedirs("data/newsclippings/evidence/", exist_ok=True)
    os.makedirs("data/newsclippings/evidence/train", exist_ok=True)
    os.makedirs("data/newsclippings/evidence/val", exist_ok=True)
    os.makedirs("data/newsclippings/evidence/test", exist_ok=True)

    #Prepare the datasets
    create_data_with_inverse_search_results('train', args.path_to_inverse_search)
    create_data_with_inverse_search_results('val', args.path_to_inverse_search)
    create_data_with_inverse_search_results('test', args.path_to_inverse_search)

    #Prepare train and val subsets used in experiments
    with open("data/newsclippings/train.json", 'r') as file:
        train = json.load(file)
    with open("data/newsclippings/val.json", 'r') as file:
        val = json.load(file)

    train_samples = random.sample(train['annotations'], 5000)
    json.dump(train_samples, open("data/newsclippings/train.json", 'w'))

    val_samples = random.sample(val['annotations'], 1500)
    json.dump(val_samples, open("data/newsclippings/val.json", 'w'))