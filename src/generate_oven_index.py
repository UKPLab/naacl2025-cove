import torch
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from tqdm import tqdm
import numpy as np
import json
import faiss
from utils import * 

if __name__=='__main__':

    #Load CLIP model for enconding the wikidata entities
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    #Load the entities file
    entities = []
    #This file needs first to be manually added to the root folder from http://storage.googleapis.com/gresearch/open-vision-language/Wiki6M_ver_1_0_title_only.jsonl
    file_path = 'Wiki6M_ver_1_0_title_only.jsonl'
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            #Parse each line as JSON and append to the list
            data = json.loads(line)
            entities.append(data['wikipedia_title'])
    print('Embedding %s entities'%len(entities))


    batch_size = 1024
    all_embeddings = []
    for i in tqdm(range(0, len(entities), batch_size)):
        #Compute the embeddings for all texts
        batch_texts = entities[i:i+batch_size]
        batch_texts = truncate_texts(batch_texts, tokenizer)
        inputs = processor(text=batch_texts, padding=True, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                outputs = model.get_text_features(**inputs)
                batch_embeddings = outputs.cpu().numpy()
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

        all_embeddings.append(batch_embeddings)



    text_embeddings = np.vstack(all_embeddings).astype(np.float32)
    print('Number of embeddings: %s'%len(text_embeddings))
    # Create a FAISS index
    d = text_embeddings.shape[1]
    index_cpu = faiss.IndexFlatL2(d)  # Create a CPU index
    # Add text embeddings to the index
    index_cpu.add(text_embeddings)
    #Important note: the OVEN index takes up to 10GB in memory
    faiss.write_index(index_cpu, "wikipedia_entity_collection/oven.index") 