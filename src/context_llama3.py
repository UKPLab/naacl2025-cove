import argparse
import pandas as pd
import tqdm
import random
import re
import numpy as np
import os
import spacy
import torch
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
    
    args = parser.parse_args()

    #Load data
    instances = pd.read_csv(f"results/intermediate/context_input_{args.dataset}_{args.split}.csv").to_dict(orient="records")
    cropped_object_captions = pd.read_csv(f'data/{args.dataset}/evidence/{args.split}/automated_captions/object_captions.csv')
    
    #Prepare model
    llm = LLM(
        args.model_path,
        tokenizer=args.model_path,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.5,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params_QA = SamplingParams(
        temperature=0, top_p=1, max_tokens=100,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    #Load spacy model
    nlp = spacy.load("en_core_web_lg")

    #Template for questions
    prefix_QA_with_evidence = """

    Web captions that may be relevant: {}
    Caption 1 - Global description: {}
    Caption 2 - More details about the people in the image: {}
    Visual entities that may be relevant, without certitude: {}
    Question: {} Answer in one sentence. Answer with one word ("unknown") if the information is not provided.
    Answer: """

    prefix_QA_no_evidence = """

    Caption 1 - Global description: {}
    Caption 2 - More details about the people in the image: {} {extra_captions}
    Visual entities that may be relevant, without certitude: {}
    Question: {} Answer in one sentence. Answer with one word ("unknown") if the information is not provided.
    Answer: """

    people_Q = "Who is shown in the image?"
    things_Q = "Which animal, plant, building, or object are shown in the image?"
    event_Q = "Which event is depicted in the image?"
    date_Q = "When was the image taken?"
    location_Q = "Where was the image taken?"
    motivation_Q = "Why was the image taken?"
    source_Q = "Who is the source of the image?"

    # ALWAYS keep questions this question order; evidence sorting and demonstrations are based on this indexing
    questions_with_evidence = [people_Q, things_Q, event_Q, date_Q, location_Q, motivation_Q, source_Q]
    questions_no_evidence = [people_Q, things_Q, event_Q, date_Q, location_Q, motivation_Q]

    batch_size = args.batch_size
    for i in tqdm.tqdm(range(0, len(instances), batch_size)):


        if i+args.batch_size > len(instances):
            batch_size = len(instances) - i

        veracity = np.empty(batch_size, dtype='object')
        predicted_contexts = np.empty(batch_size, dtype='object')
        original_veracity_outputs = np.empty(batch_size, dtype='object')
        
        all_qa_inputs = []
        for batch_index in range(batch_size):
            #Loop through the data to prepare input prompt
            sample = instances[i+batch_index]

            #Keep list of direct string match evidence
            direct_match_captions = []
            #Veracity rules: check for direct matches
            non_alphanum_pattern = re.compile('[\W_]+')
            if sample['evidence'] != []:
                evidence_captions = sample['evidence']
                for caption in evidence_captions:
                    # lowercase and remove punctuation and whitespaces
                    caption_to_search = sample['caption'].lower()
                    evidence_to_test = caption.lower()
                    caption_to_search = re.sub(non_alphanum_pattern, '', caption_to_search)
                    evidence_to_test = re.sub(non_alphanum_pattern, '', evidence_to_test)
                    if re.search(caption_to_search, evidence_to_test):
                        direct_match_captions.append(caption)
                        veracity[batch_index] = 'pristine'
                        original_veracity_outputs[batch_index] = "Direct match found in evidence."
                        break
            #Sort web evidence
            if sample['evidence'] != []:
                if direct_match_captions != []:
                    evidence_captions = direct_match_captions
                else:
                    evidence_captions = sample['evidence']
                counts = count_evidence_NER(evidence_captions, nlp)
                # list of caption lists (ordered by NER counts)
                sorted_evidence = sort_evidence_by_QA(evidence_captions, counts)
                inputs = [demo + [{"role": "user", "content": prefix_QA_with_evidence.format(evidence, sample['object_caption'], sample['people_caption'], sample['vis_entities'], q)}] for demo, evidence, q in zip(generate_QA_demos_with_captions(), sorted_evidence, questions_with_evidence)]
            else: # no evidence, use extra object captions
                cropped_detected_objects_rows = cropped_object_captions[cropped_object_captions['image_id'] == sample['image_id']]
                extra_object_captions = ""
                if len(cropped_detected_objects_rows) > 0:
                    for obj_ix, obj_row in cropped_detected_objects_rows.iterrows():
                        extra_object_captions += f"\nCaption {obj_ix+3} - More details about the {obj_row['category'].lower()} in the image: {obj_row['vqa_answer']}\n"

                inputs = [demo + [{"role": "user", "content": prefix_QA_no_evidence.format(sample['object_caption'], sample['people_caption'], sample['vis_entitites'], q, extra_captions=extra_object_captions)}] for demo, q in zip(generate_QA_demos_no_caption(), questions_no_evidence)]
            all_qa_inputs += inputs        


        #Question Answering
        input_texts = tokenizer.apply_chat_template(
            all_qa_inputs,
            tokenize=False,
        )
        QA_outputs = llm.generate(input_texts, sampling_params=sampling_params_QA)


        #Prepare predicted context to be saved
        context_info = []
        output_index = 0 # used to keep track of the QA outputs, 7 means with evidence (because of source), 6 means no evidence
        indices_no_exact_match = []
        indices_unknown_veracity = []
        for batch_index in range(batch_size):
            # if already direct match/non-match, format context items separated with |||
            if veracity[batch_index] == 'pristine':
                predicted_contexts[batch_index] = '|||'.join([o.outputs[0].text.split('\n\n')[-1].strip() for o in QA_outputs[output_index:output_index+7]])
                output_index += 7
                continue

            sample = instances[i+batch_index]
            image_id = sample['image_id']
            caption_id = sample['caption_id']

            caption_to_verify = sample['caption']

            if sample['evidence'] != []: # with evidence
                answers = [o.outputs[0].text.split('\n\n')[-1].strip() for o in QA_outputs[output_index:output_index+7]]
                # standardize to "Unknown." if unknown
                for k in range(len(answers)):
                    if 'unknown' in answers[k][:10].lower():
                        answers[k] = 'Unknown.'

                #Make sure that wiki matches are included in the people answer
                if answers[0] == 'Unknown.' and any(res[1] == 'wiki_image' for res in sample['wiki_clip'][batch_index]):
                    answers[0] = ', '.join([res[0] for res in sample['wiki_clip'][batch_index] if res[1] == 'wiki_image'])
                
                # remove unknown answers
                answer_items = ['People: ', 'Things: ', 'Event: ', 'Date: ', 'Location: ', 'Motivation: ', 'Source: ']
                known_answers = []
                for k in range(len(answers)):
                    if answers[k] == 'Unknown.':
                            continue
                    else:
                        known_answers.append(answer_items[k] + f"{answers[k]}")
                answers = "\n".join(known_answers)
                if known_answers == []:
                    output_index += 7
                    continue
                output_index += 7
                indices_no_exact_match.append(batch_index)
            else: # no evidence
                answers = [o.outputs[0].text.split('\n\n')[-1].strip() for o in QA_outputs[output_index:output_index+6]]

                # standardize to "Unknown." if unknown
                for k in range(len(answers)):
                    if 'unknown' in answers[k][:10].lower():
                        answers[k] = 'Unknown.'

                # make sure that wiki matches are included in the people answer (Wiki matches are enforced but not clip matches)
                if answers[0] == 'Unknown.' and any(res[1] == 'wiki_image' for res in sample['wiki_clip'][batch_index]):
                    answers[0] = ', '.join([res[0] for res in sample['wiki_clip'][batch_index] if res[1] == 'wiki_image'])

                # remove unknown answers
                answer_items = ['People: ', 'Things: ', 'Event: ', 'Date: ', 'Location: ', 'Motivation: ']
                known_answers = []
                for k in range(len(answers)):
                    if answers[k] == 'Unknown.':
                            continue
                    else:
                        known_answers.append(answer_items[k] + f"{answers[k]}")
                answers = "\n".join(known_answers)
                if known_answers == []:
                    indices_unknown_veracity.append(batch_index)
                    output_index += 6
                    continue
                output_index += 6
                indices_no_exact_match.append(batch_index)


        for ix in range(0, len(context_info)):
            # skipping exact matches which have already received contexts
            index = indices_no_exact_match[ix]
            predicted_contexts[index] = answers.replace('\n', '').strip().replace('    ', ' ')


        instances = pd.DataFrame({"image_id": [item['image_id'] for item in instances[i:i+batch_size]],
                            "caption_id": [item['caption_id'] for item in instances[i:i+batch_size]],
                            "caption": [item['caption'] for item in instances[i:i+batch_size]],
                            "true_caption": [item['true_caption'] for item in instances[i:i+batch_size]],
                            "true_veracity": [item['true_veracity'] for item in instances[i:i+batch_size]],
                            "people_caption": [item['people_caption'] for item in instances[i:i+batch_size]],
                            "object_caption": [item['object_caption'] for item in instances[i:i+batch_size]],
                            "predicted_context_QA": predicted_contexts,
                            "evidence_captions": [item['evidence'] for item in instances[i:i+batch_size]],
                            "vis_entities": [item['vis_entities'] for item in instances[i:i+batch_size]],
                            "wiki_clip": [item['wiki_clip'] for item in instances[i:i+batch_size]],
                            "direct_search_match": [item['direct_search_match'] for item in instances[i:i+batch_size]],
                            "direct_search_non_match": [item['direct_search_non_match'] for item in instances[i:i+batch_size]],
                            "image_path": [item['image_path'] for item in instances[i:i+batch_size]],
                            })

        output_path = f"results/intermediate/context_output_{args.dataset}_{args.split}.csv"
        instances.to_csv(output_path,
                    mode="a",
                    index=False,
                    sep=",",
                    header=not os.path.exists(output_path))





