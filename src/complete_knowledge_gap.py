import argparse
import pandas as pd
import spacy
import random
import numpy as np
import os
import torch
from vllm import LLM
from knowledge_gap_completion.utils import *
from utils import *


torch.cuda.manual_seed(42)
random.seed(42)


if __name__=='__main__':
    parser = argparse.ArgumentParser("Prepare input dataset for context prediction, grouping all evidence.")
    parser.add_argument('--dataset', type=str,  default= "newsclippings", choices=['newsclippings', '5pils-ooc'],
                        help='The dataset to use.') 
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 
    parser.add_argument('--batch_size', type=int,  default= 128,
                        help='Number of instances to process in each batch')
    parser.add_argumen('--model_path', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help='Path to the LLM used for QA')
    parser.add_argumen('--ip', type=str, required = True,
                        help='Path to the IP where the WikiChat server is running')
        
    args = parser.parse_args()


    #Load data
    if args.dataset=='newsclippings':
        data = json.load(open(f'data/newsclippings/{args.split}.json', 'r'))
        images_root_folder = 'data/newsclippings/visual_news/origin'
    else:
        data = json.load(open(f'data/5pils_ooc/test.json', 'r'))
        images_root_folder = 'data/5pils_ooc'
    context_results = pd.read_csv(f'results/intermediate/context_output_{args.dataset}_{args.split}.csv')

    #Load model
    llm = LLM(
    args.model_path,
    tokenizer=args.model_path,
    dtype=torch.bfloat16,
    gpu_memory_utilization=0.5,
    )

    tokenizer = llm.get_tokenizer()

    #Load spacy model
    nlp = spacy.load("en_core_web_lg")


    #Prepare instances
    instances = []
    for ix in range(len(data)):

        instance_dict = {}        
        instance_dict['image_id'] = data[ix]['image_id']
        instance_dict['caption_id'] = data[ix]['id']
        if args.dataset=='newsclippings':
            instance_dict['caption'] = data[ix]['false_caption'] if 'false_caption' in data[ix] else data[ix]['true_caption']
        else:
            instance_dict['caption'] = data[ix]['caption']
        instance_dict['image_path'] = os.path.join(images_root_folder, data[ix]['image_path'])
        instance_dict['context'] = context_results[(context_results['image_id'] == data[ix]['image_id']) & (context_results['caption'] == instance_dict['caption'])]['predicted_context_QA'].values[0]

        instances.append(instance_dict)

    
    #Generate knowledge questions
    question_output_path = f"results/intermediate/knowledge_questions_{args.dataset}_{args.split}.csv"
    knowledge_question_generation(instances,
                                  question_output_path,
                                  llm,
                                  tokenizer,
                                  nlp,
                                  128)
    

    #Answer the knowledge questions
    knowledge_questions = pd.read_csv(question_output_path)
    instances = []

    for ix in range(len(data)):

        instance_dict = {}        
        instance_dict['image_id'] = data[ix]['image_id']
        instance_dict['caption_id'] = data[ix]['id']
        if args.dataset=='newsclippings':
            instance_dict['caption'] = data[ix]['false_caption'] if 'false_caption' in data[ix] else data[ix]['true_caption']
        else:
            instance_dict['caption'] = data[ix]['caption']
        instance_dict['image_path'] = os.path.join(images_root_folder, data[ix]['image_path'])
        instance_dict['date_questions'] = knowledge_questions[(knowledge_questions['image_id'] == data[ix]['image_id']) & (knowledge_questions['caption_id'] == data[ix]['id'])]['date_questions'].values[0]
        instance_dict['location_questions'] = knowledge_questions[(knowledge_questions['image_id'] == data[ix]['image_id']) & (knowledge_questions['caption_id'] == data[ix]['id'])]['location_questions'].values[0]
        instances.append(instance_dict)

    answer_output_path = f"results/intermediate/knowledge_answers_{args.dataset}_{args.split}.csv"
    knwoledge_answer_generation(instances,
                                answer_output_path,
                                llm,
                                tokenizer,
                                args.ip, 
                                128)


    #Validate the answers and update context if possible
    knowledge_answers = pd.read_csv(answer_output_path)

    instances = []

    for ix in range(len(knowledge_answers)):

        instance_dict = {}        
        instance_dict['image_id'] = knowledge_answers.iloc[ix]['image_id']
        instance_dict['caption_id'] = knowledge_answers.iloc[ix]['caption_id']
        instance_dict['image_path'] = knowledge_answers.iloc[ix]['image_path']
        instance_dict['answer'] = knowledge_answers.iloc[ix]['all_answers']
        instance_dict['question'] = knowledge_answers.iloc[ix]['all_questions']
        instance_dict['context'] = context_results[(context_results['image_id'] == instance_dict['image_id']) & (context_results['caption_id'] == instance_dict['caption_id'])]['predicted_context_QA'].values[0]
        instances.append(instance_dict)

    context_answer_output_path = f"results/intermediate/knowledge_context_answers_{args.dataset}_{args.split}.csv"
    answer_with_context(instances,
                        context_answer_output_path,
                        llm, 
                        tokenizer,
                        128)
