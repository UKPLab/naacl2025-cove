import os 
from tqdm import tqdm
import time
from utils import *


if __name__=='__main__':

    root = 'data/5pils_ooc/evidence/direct_search'
    data = load_json(os.path.join(root,'trafilatura.json'))
        
    urls = [d['url'] for d in data]
    images = [d['image url'] for d in data]

    #Download evidence images
    for idx in range(len(images)):
        file_path = os.path.join(root, str(idx))
        download_image(images[idx], file_path)


    output = []
    for u in tqdm(range(len(urls))):
        output.append(extract_info_trafilatura(urls[u],images[u]))
        time.sleep(3)
    
    # Save the list of dictionaries as a JSON file
    with open(os.path.join(root,'trafilatura.json'), 'w') as file:
        json.dump(output, file, indent=4)
