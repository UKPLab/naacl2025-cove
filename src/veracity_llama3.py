
import argparse
import pandas as pd
import tqdm
import random
import numpy as np
import os
import torch
from collections import Counter
from vllm import LLM, SamplingParams
from cove.generation_utils import *
from cove.prompts import *
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
    parser.add_argumen('--knowledge_gap_completion', type=int, default=0,
                        help='If 1, include answers obtained with knowledge gap completion for date and location')
    
    args = parser.parse_args()


    #Load data
    instances = pd.read_csv(f"results/intermediate/context_output_{args.dataset}_{args.split}.csv").to_dict(orient="records")

    #Knowledge gap completion
    if args.knowledge_gap_completion:
        knowledge_answers = pd.read_csv(f"results/intermediate/knowledge_answers__{args.dataset}_{args.split}.csv")
    else:
        knowledge_answers = []

    #Prepare model
    llm = LLM(
        args.model_path,
        tokenizer=args.model_path,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.5,
    )
    tokenizer = llm.get_tokenizer()
    #sampling params veracity
    sampling_params_veracity = SamplingParams(
        temperature=0, top_p=1, max_tokens=175,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    batch_size = args.batch_size
    for i in tqdm.tqdm(range(0, len(instances), batch_size)):


        if i+args.batch_size > len(instances):
            batch_size = len(instances) - i

        predicted_contexts = np.empty(batch_size, dtype='object')
        original_veracity_outputs = np.empty(batch_size, dtype='object')


        all_veracity_inputs = []
        context_info = []

        output_index = 0 # used to keep track of the QA outputs, 7 means with evidence (because of source), 6 means no evidence
        indices_no_exact_match = []
        indices_unknown_veracity = []
        for batch_index in range(batch_size):
            # if already direct match/non-match, skip
            sample = instances[i+batch_index]
            if '|||' in sample['predicted_context_QA']:
                #Direct match/ non match
                output_index += 7
                continue

            image_id = sample['image_id']
            caption_id = sample['caption_id']

            # lists of context QA answers, remove unknown answers
            if args.knowlege_gap_completion:
                all_context_qa = knowledge_answers[(knowledge_answers['image_id'] == image_id) & (knowledge_answers['caption_id'] == caption_id)]
                qa_locations = all_context_qa[all_context_qa['all_questions'].str.startswith('Where')].all_context_answers.values
                qa_dates = all_context_qa[all_context_qa['all_questions'].str.startswith('When')].all_context_answers.values
                qa_locations = [a for a in qa_locations if not 'unknown' in a[:10].lower()]
                qa_dates = [a for a in qa_dates if not 'unknown' in a[:10].lower()]

            caption_to_verify = sample['caption']

            if sample['evidence'] != []: # with evidence
                # remove unknown answers - replace date and location with context QA
                answer_items = ['People: ', 'Things: ', 'Event: ', 'Date: ', 'Location: ', 'Motivation: ', 'Source: ']
                known_answers = []
                for k in answer_items:
                    #Add knowledge answers if available and add spacing between answer items
                    if k == 'Date: ' and k not in sample['predicted_context_QA'] and qa_dates != []:
                        if 'Location: ' in sample['predicted_context_QA']:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'].split('Location: ')[0] + 'Date: ' + f"{' '.join(qa_dates)}"   + 'Location: ' + sample['predicted_context_QA'].split('Location: ')[1]
                        elif 'Motivation: ' in sample['predicted_context_QA']:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'].split('Motivation: ')[0] + 'Date: ' + f"{' '.join(qa_dates)}"   + 'Motivation: ' + sample['predicted_context_QA'].split('Motivation: ')[1]
                        elif 'Source: ' in sample['predicted_context_QA']:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'].split('Source: ')[0] + 'Date: ' + f"{' '.join(qa_dates)}"   + 'Source: ' + sample['predicted_context_QA'].split('Source: ')[1]
                        else:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'] + 'Date: ' + f"{' '.join(qa_dates)}"  
                    if k == 'Location: ' and k not in sample['predicted_context_QA'] and qa_locations != []:
                        if 'Motivation: ' in sample['predicted_context_QA']:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'].split('Motivation: ')[0] + 'Location: ' + f"{' '.join(qa_locations)}"   + 'Motivation: ' + sample['predicted_context_QA'].split('Motivation: ')[1]
                        elif 'Location: ' in sample['predicted_context_QA']:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'].split('Source: ')[0] + 'Location: ' + f"{' '.join(qa_locations)}"   + 'Source: ' + sample['predicted_context_QA'].split('Source: ')[1]
                        else:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'] + 'Location: ' + f"{' '.join(qa_locations)}"  

                    if k in sample['predicted_context_QA']:
                        sample['predicted_context_QA'] = sample['predicted_context_QA'].split(k)[0] + '\n' + k + sample['predicted_context_QA'].split(k)[1]
  
                # if no context information, skip veracity
                if sample['predicted_context_QA']=="":
                    indices_unknown_veracity.append(batch_index)
                    output_index += 7
                    continue

                reason_prompt = """
                Context information:
                {}
                
                Caption to verify: {}
                Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
                veracity_demo = generate_veracity_demo_with_captions()
                input_text = veracity_demo + [{"role": "user", "content": reason_prompt.format(sample['predicted_context_QA'], caption_to_verify)}]
                all_veracity_inputs.append(input_text)
                context_info.append(reason_prompt.format(sample['predicted_context_QA'], caption_to_verify).strip())
                output_index += 7
                indices_no_exact_match.append(batch_index)

            else: # no evidence
                # remove unknown answers - replace date and location with context QA
                answer_items = ['People: ', 'Things: ', 'Event: ', 'Date: ', 'Location: ', 'Motivation: ']
                known_answers = []
                for k in answer_items:
                    #Add knowledge answers if available and add spacing between answer items
                    if k == 'Date: ' and k not in sample['predicted_context_QA'] and qa_dates != []:
                        if 'Location: ' in sample['predicted_context_QA']:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'].split('Location: ')[0] + 'Date: ' + f"{' '.join(qa_dates)}"   + 'Location: ' + sample['predicted_context_QA'].split('Location: ')[1]
                        elif 'Motivation: ' in sample['predicted_context_QA']:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'].split('Motivation: ')[0] + 'Date: ' + f"{' '.join(qa_dates)}"   + 'Motivation: ' + sample['predicted_context_QA'].split('Motivation: ')[1]
                        else:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'] + 'Date: ' + f"{' '.join(qa_dates)}"  
                    if k == 'Location: ' and k not in sample['predicted_context_QA'] and qa_locations != []:
                        if 'Motivation: ' in sample['predicted_context_QA']:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'].split('Motivation: ')[0] + 'Location: ' + f"{' '.join(qa_locations)}"   + 'Motivation: ' + sample['predicted_context_QA'].split('Motivation: ')[1]
                        elif 'Location: ' in sample['predicted_context_QA']:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'].split('Source: ')[0] + 'Location: ' + f"{' '.join(qa_locations)}"   + 'Source: ' + sample['predicted_context_QA'].split('Source: ')[1]
                        else:
                            sample['predicted_context_QA'] = sample['predicted_context_QA'] + 'Location: ' + f"{' '.join(qa_locations)}"  

                    if k in sample['predicted_context_QA']:
                        sample['predicted_context_QA'] = sample['predicted_context_QA'].split(k)[0] + '\n' + k + sample['predicted_context_QA'].split(k)[1]
                # if no context information, skip veracity
                if sample['predicted_context_QA']=="":
                    indices_unknown_veracity.append(batch_index)
                    output_index += 6
                    continue
                
                reason_prompt = """
                Context information:
                {}

                Caption to verify: {}
                Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
                veracity_demo = generate_veracity_demo_no_captions()
                input_text = veracity_demo + [{"role": "user", "content": reason_prompt.format(sample['predicted_context_QA'], caption_to_verify)}]
                all_veracity_inputs.append(input_text)
                context_info.append(reason_prompt.format(sample['predicted_context_QA'], caption_to_verify).strip())
                output_index += 6
                indices_no_exact_match.append(batch_index)


        for u in indices_unknown_veracity:
            predicted_contexts[u] = 'No context information provided. Unknown.'
            original_veracity_outputs[u] = 'No context information provided. Unknown.'

        veracity_input_texts = tokenizer.apply_chat_template(
            all_veracity_inputs,
            tokenize=False,
        )

        ver_outputs = llm.generate(veracity_input_texts, sampling_params=sampling_params_veracity)

        for ix in range(0, len(ver_outputs)):
            output = ver_outputs[ix].outputs[0].text
            # skipping exact matches
            index = indices_no_exact_match[ix]
            predicted_contexts[index] = context_info[ix].split("Context information:")[1].split("Caption to verify:")[0].strip().replace('\n', '').replace('    ', ' ')
            original_veracity_outputs[index] = output.strip().replace('\n', ' ').replace('    ', ' ')


        instances = pd.DataFrame({"image_id": [item['image_id'] for item in instances[i:i+batch_size]],
                            "caption_id": [item['caption_id'] for item in instances[i:i+batch_size]],
                            "caption": [item['caption'] for item in instances[i:i+batch_size]],
                            "true_caption": [item['true_caption'] for item in instances[i:i+batch_size]],
                            "true_veracity": [item['true_veracity'] for item in instances[i:i+batch_size]],
                            "people_caption": [item['people_caption'] for item in instances[i:i+batch_size]],
                            "object_caption": [item['object_caption'] for item in instances[i:i+batch_size]],
                            "predicted_context_QA": predicted_contexts,
                            "original_veracity_output": original_veracity_outputs,
                            "evidence_captions": [item['evidence'] for item in instances[i:i+batch_size]],
                            "vis_entities": [item['vis_entities'] for item in instances[i:i+batch_size]],
                            "wiki_clip": [item['wiki_clip'] for item in instances[i:i+batch_size]],
                            "direct_search_match": [item['direct_search_match'] for item in instances[i:i+batch_size]],
                            "direct_search_non_match": [item['direct_search_non_match'] for item in instances[i:i+batch_size]],
                            "max_direct_search_sim": [item['max_direct_search_sim'] for item in instances[i:i+batch_size]],
                            "min_direct_search_sim": [item['min_direct_search_sim'] for item in instances[i:i+batch_size]],
                            "image_path": [item['image_path'] for item in instances[i:i+batch_size]],
                            })

        output_path = f"results/intermediate/{args.dataset}_{args.split}.csv"
        instances.to_csv(output_path,
                    mode="a",
                    index=False,
                    sep=",",
                    header=not os.path.exists(output_path))


    #Compute final veracity based on raw veracity predictions
    results = pd.read_csv(f"results/intermediate/{args.dataset}_{args.split}.csv")

    generated_answers = [a.split('Answer: ')[1].split('.')[0] if 'Answer: ' in a else a for a in results['original_veracity_output'].to_list()]
    generated_answers = ['Probably out-of-context' if 'probably out-of-context' in a.lower()  else a for a in generated_answers]
    generated_answers = ['Accurate' if 'Direct match' in a else a for a in generated_answers]
    generated_answers = ['Probably accurate' if 'probably accurate' in a else a for a in generated_answers]
    generated_answers = ['Accurate' if 'is accurate' in a else a for a in generated_answers]
    generated_answers = ['Unknown' if a not in ['Probably out-of-context', 'Out-of-context',
                                                'Probably accurate', 'Accurate'] else a for a in generated_answers]
    #Replace Unknown by most frequent label in the dataset
    results['predicted_veracity'] = generated_answers
    results['predicted_veracity'] = results.apply(lambda row: 'pristine' if row['direct_search_match']==True 
                                                           else 'falsified' if row['direct_search_non_match']==True
                                                           else 'pristine' if row['predicted_veracity'] in ['Accurate', 'Probably accurate']
                                                           else 'falsified' if row['predicted_veracity'] in ['Probably out-of-context', 'Out-of-context']
                                                           else row['predicted_veracity'], axis=1)
    most_frequent_answer = Counter(results['predicted_veracity'].to_list()).most_common(1)[0][0]
    results['predicted_veracity'] = [g if g!='Unknown' else most_frequent_answer for g in results['predicted_veracity'].to_list()]
    
    results.to_csv(f"results/intermediate/{args.dataset}_{args.split}.csv",
                   index=False,
                   sep=",")