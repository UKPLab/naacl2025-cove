import pandas as pd
import tqdm
import re
import numpy as np
import os
import torch
from vllm import SamplingParams
from knowledge_gap_completion.prompts import *
import requests
import time
import json

def query_wikichat(queries,
                 ip, 
                 port=5000,
                 evi_num=1, 
                 sleep=1):
    '''
    Query the  index with Colbert to retrieve wikipedia knowledge matching a question.
    Params:
        queries (str): the input question
        ip (str): the ip address to the server where the Wikipedia index is hosted
    '''
    url = f"http://{ip}:{port}/search"
    headers = {'Content-Type': 'application/json'}
    
    results = []
    
    for query in tqdm(queries):
        data = {
            "query": query,
            "evi_num": evi_num
        }
        
        response = requests.get(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            results.append({
                "query": query,
                "response": response.json()
            })
        else:
            results.append({
                "query": query,
                "response": None,
                "error": response.text
            })
        time.sleep(sleep)
    
    return results


def knowledge_question_generation(instances, 
                                  output_path,
                                  model, 
                                  tokenizer, 
                                  spacy_model,   
                                  batch_size=1): 

    sampling_params = SamplingParams(
        temperature=0, top_p=1, max_tokens=100,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    
    for i in tqdm.tqdm(range(0, len(instances), batch_size)):

        if i+batch_size > len(instances):
            batch_size = len(instances) - i

        date_questions = np.empty(batch_size, dtype='object')
        location_questions = np.empty(batch_size, dtype='object')

        last_messages_date = []
        last_messages_location = []
        ixs_date_unknown = []
        ixs_location_unknown = []

        for batch_index in range(batch_size):
            sample = instances[i+batch_index]
            context_info = sample['context']
            date_unknown = False
            location_unknown = False
            context_to_use = ''

            labels_list = ['People: ', 'Things: ', 'Event: ', 'Date: ', 'Location: ', 'Motivation: ', 'Source: ']
            if '|||' in context_info:
                # add labels back to context
                people, apo, event, date, location, motivation, source = [l+c.strip() for l,c in zip(labels_list, context_info.split('|||'))]
                context_list = [people, apo, event, date, location, motivation] # remove source
            else:
                pos_to_find = [context_info.find(l) for l in labels_list if l in context_info] # find positions to split string
                if len(pos_to_find) == 0:
                    context_list = []
                else:
                    context_list = [context_info[i:j] for i,j in zip(pos_to_find, pos_to_find[1:]+[None])]
                    if context_list[-1].startswith('Source: '):
                        # remove source
                        context_list = context_list[:-1]
            
            # remove unknowns
            context_list = [c for c in context_list if "Unknown." not in c]

            if len([c for c in context_list if 'Date: ' in c]) == 0:
                # only if (location+people) OR event OR motivation is not unknown
                if (len([c for c in context_list if ('People: ' in c or 'Location: ' in c)]) != 2) or \
                    (len([c for c in context_list if 'Event: ' in c]) == 0) or \
                    (len([c for c in context_list if 'Motivation: ' in c]) == 0):
                    date_unknown = True
            if len([c for c in context_list if 'Location: ' in c]) == 0:
                # only if (date+people) OR event OR motivation is not unknown
                if (len([c for c in context_list if ('People: ' in c or 'Date: ' in c)]) != 2) or \
                    (len([c for c in context_list if 'Event: ' in c]) == 0) or \
                    (len([c for c in context_list if 'Motivation: ' in c]) == 0):
                    location_unknown = True

            if len(context_list) > 0:
                context_to_use = "\n".join(context_list)


            prompt_date = """Given the following context, generate world knowledge questions that will help determine when the image was taken. The questions should always start with “When” or “On which date”. The question should be self-contained and specific enough to be answerable based on world knowledge. If no self-contained and specific question can be generated, your response should be “No questions can be generated given the context”.

            Context information:
            {}

            Generated questions (up to 3): """
            if date_unknown and context_to_use != '':
                last_message_date = {"role": "user", "content": prompt_date.format(context_to_use)}
                last_messages_date.append(last_message_date)
                ixs_date_unknown.append(batch_index)

            prompt_location = """Given the following context, generate world knowledge questions that will help determine where the image was taken. The questions should start with “Where”. The questions should be self-contained and specific enough to be answerable based on world knowledge. If no self-contained and specific question can be generated, your response should be “No questions can be generated given the context”.

            Context information:
            {}

            Generated question: """            
            if location_unknown and context_to_use != '':
                last_message_location = {"role": "user", "content": prompt_location.format(context_to_use)}
                last_messages_location.append(last_message_location)
                ixs_location_unknown.append(batch_index)


        messages_date, messages_location = assemble_prompt_knowledge_question_generation()
        date_input_texts = tokenizer.apply_chat_template(
            [messages_date + [p] for p in last_messages_date],
            tokenize=False,
        )

        location_input_texts = tokenizer.apply_chat_template(
            [messages_location + [p] for p in last_messages_location],
            tokenize=False,
        )

        date_outputs = model.generate(date_input_texts, sampling_params=sampling_params)
        location_outputs = model.generate(location_input_texts, sampling_params=sampling_params)

        for ix in range(0, len(date_outputs)):
            output = date_outputs[ix].outputs[0].text
            date_out = output.split("\n\n")[-1].replace('\n', ' ')
            # check that index is from ones with date unknown
            date_questions[ixs_date_unknown[ix]] = date_out

        for ix in range(0, len(location_outputs)):
            output = location_outputs[ix].outputs[0].text
            location_out = output.split("\n\n")[-1].replace('\n', ' ')
            if location_out != 'No questions can be generated given the context.':
                # verify that there are no hallucinated entities
                context = instances[i+ixs_location_unknown[ix]]['context']
                questions = re.split(r'Q[0-9]*:', location_out)
                good_questions = []
                for q in questions:
                    bad_question = False
                    doc = spacy_model(q)
                    for ent in doc.ents:
                        if ent.text not in context:
                            #Filter all questions that hallucinate new named entities that were not part of the context
                            bad_question = True
                            break
                    if not bad_question:
                        good_questions.append(q)
                good_questions = [q for q in good_questions if q != '']
                location_out = ' '.join([f'Q{i+1}:{q}' for i, q in enumerate(good_questions)])
            # check that index is from ones with location unknown
            location_questions[ixs_location_unknown[ix]] = location_out

        data = pd.DataFrame({"image_id": [item['image_id'] for item in instances[i:i+batch_size]],
                             "caption_id": [item['caption_id'] for item in instances[i:i+batch_size]],
                             "caption": [item['caption'] for item in instances[i:i+batch_size]],
                             "context": [item['context'] for item in instances[i:i+batch_size]],
                             "date_questions": date_questions,
                             "location_questions": location_questions,
                             "image_path": [item['image_path'] for item in instances[i:i+batch_size]],})

        data.to_csv(output_path,
                    mode="a",
                    index=False,
                    sep=",",
                    header=not os.path.exists(output_path))
                
        torch.cuda.empty_cache()



def knwoledge_answer_generation(instances, 
                                output_path, 
                                model,
                                tokenizer, 
                                ip, 
                                batch_size=1, 
                                min_wiki_score=20):

    sampling_params = SamplingParams(
        temperature=0, top_p=1, max_tokens=150,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    
    for i in tqdm.tqdm(range(0, len(instances), batch_size)):

        if i+batch_size > len(instances):
            batch_size = len(instances) - i

        all_date_questions = []
        all_location_questions = []
        all_answers = []
        all_date_image_ids = []
        all_date_caption_ids = []
        all_date_image_paths = []
        all_location_image_ids = []
        all_location_caption_ids = []
        all_location_image_paths = []

        last_messages_date = []
        last_messages_location = []


        prompt_date = """Given the following question, generate an answer based on the available Wikipedia knowledge in 1 or 2 sentences that are as specific as possible. If the question cannot be answered based on the available knowledge, your response should be "Unknown".

        Wikipedia knowledge: {}
        Question: {}

        Answer: """  

        prompt_location = """Given the following question, generate an answer based on the available Wikipedia knowledge in 1 or 2 sentences that are as specific as possible. If the question cannot be answered based on the available knowledge, your response should be "Unknown".

        Wikipedia knowledge: {}
        Question: {}

        Answer: """

        for batch_index in range(batch_size):
            sample = instances[i+batch_index]
            date_questions = sample['date_questions']
            location_questions = sample['location_questions']

            if type(date_questions) == str:
                if date_questions.startswith('Q1'):
                    d_questions = re.split(r'Q[0-9]+: ', date_questions)
                    
                    date_wiki = query_wikichat(d_questions, ip) 
                
                    for q in d_questions:
                        for item in date_wiki:
                            # check if the question is in the wiki data and the score is high enough
                            if q == item['query'] and item['response']['passage_scores'][0] > min_wiki_score:
                                d_wiki = item['response']['passages'][0]
                                last_message_date = {"role": "user", "content": prompt_date.format(d_wiki, q)}
                                last_messages_date.append(last_message_date)
                                # save info for csv
                                all_date_questions.append(q)
                                all_date_image_ids.append(sample['image_id'])
                                all_date_caption_ids.append(sample['caption_id'])
                                all_date_image_paths.append(sample['image_path'])
                                break
                    

            if type(location_questions) == str:
                if location_questions.startswith('Q1'):
                    l_questions = re.split(r'Q[0-9]+: ', location_questions)
                    location_wiki = query_wikichat(l_questions, ip)
                    for q in l_questions:
                        for item in location_wiki:
                            # check if the question is in the wiki data and the score is high enough
                            if q == item['query'] and item['response']['passage_scores'][0] > min_wiki_score:
                                l_wiki = item['response']['passages'][0]
                                last_message_location = {"role": "user", "content": prompt_location.format(l_wiki, q)}
                                last_messages_location.append(last_message_location)
                                # save info for csv
                                all_location_questions.append(q)
                                all_location_image_ids.append(sample['image_id'])
                                all_location_caption_ids.append(sample['caption_id'])
                                all_location_image_paths.append(sample['image_path'])
                                break

        messages_date, messages_location = assemble_prompt_knowledge_qa()

        date_input_texts = tokenizer.apply_chat_template(
            [messages_date + [p] for p in last_messages_date],
            tokenize=False,
        )

        location_input_texts = tokenizer.apply_chat_template(
            [messages_location + [p] for p in last_messages_location],
            tokenize=False,
        )

        date_outputs = model.generate(date_input_texts, sampling_params=sampling_params)
        location_outputs = model.generate(location_input_texts, sampling_params=sampling_params)

        for ix in range(0, len(date_outputs)):
            output = date_outputs[ix].outputs[0].text
            date_out = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace('\n', ' ').strip()
            all_answers.append(date_out)

        for ix in range(0, len(location_outputs)):
            output = location_outputs[ix].outputs[0].text
            location_out = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace('\n', ' ').strip()
            all_answers.append(location_out)

        data = pd.DataFrame({"image_id": all_date_image_ids + all_location_image_ids,
                             "caption_id": all_date_caption_ids + all_location_caption_ids,
                             "all_questions": all_date_questions + all_location_questions,
                             "all_answers": all_answers,
                             "image_path": all_date_image_paths + all_location_image_paths,})


        data.to_csv(output_path,
                    mode="a",
                    index=False,
                    sep=",",
                    header=not os.path.exists(output_path))
                
        torch.cuda.empty_cache()


def answer_with_context(instances, 
                        output_path,
                        model, 
                        tokenizer, 
                        batch_size=1):

    sampling_params = SamplingParams(
        temperature=0, top_p=1, max_tokens=150,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    
    for i in tqdm.tqdm(range(0, len(instances), batch_size)):

        if i+batch_size > len(instances):
            batch_size = len(instances) - i

        all_date_questions = []
        all_location_questions = []
        all_date_answers = []
        all_location_answers = []

        all_context_answers = []
        
        all_date_image_ids = []
        all_date_caption_ids = []
        all_date_image_paths = []
        all_location_image_ids = []
        all_location_caption_ids = []
        all_location_image_paths = []

        last_messages_date = []
        last_messages_location = []

        prompt_date = """Given the following question, generate an answer based on the context and the knowledge answers. If there is not enough information to provide an answer, your response should be “Unknown”.

        Context:
        {}

        World knowledge:
        {}

        When was the image taken?

        Answer: """

        prompt_location = """Given the following question, generate an answer based on the context and the knowledge answers. If there is not enough information to provide an answer, your response should be “Unknown”.

        Context:
        {}

        World knowledge:
        {}

        Where was the image taken?

        Answer: """

        for batch_index in range(batch_size):
            sample = instances[i+batch_index]
            context_info = sample['context']            
            original_answer = sample['answer']
            question = sample['question']

            labels_list = ['People: ', 'Things: ', 'Event: ', 'Date: ', 'Location: ', 'Motivation: ', 'Source: ']
            if '|||' in context_info:
                # add labels back to context
                people, apo, event, date, location, motivation, source = [l+c.strip() for l,c in zip(labels_list, context_info.split('|||'))]
                context_list = [people, apo, event, date, location, motivation] # remove source
            else:
                pos_to_find = [context_info.find(l) for l in labels_list if l in context_info] # find positions to split string
                if len(pos_to_find) == 0:
                    context_list = []
                else:
                    context_list = [context_info[i:j] for i,j in zip(pos_to_find, pos_to_find[1:]+[None])]
                    if context_list[-1].startswith('Source: '):
                        # remove source
                        context_list = context_list[:-1]

            # remove unknowns from context
            context_list = [c for c in context_list if "Unknown." not in c]
            context_to_use = '\n'.join(context_list)

            if question.startswith('When'):
                last_message_date = {"role": "user", "content": prompt_date.format(context_to_use, original_answer)}
                last_messages_date.append(last_message_date)
                all_date_questions.append(question)
                all_date_image_ids.append(sample['image_id'])
                all_date_caption_ids.append(sample['caption_id'])
                all_date_image_paths.append(sample['image_path'])
                all_date_answers.append(original_answer)

            if question.startswith('Where'):
                last_message_location = {"role": "user", "content": prompt_location.format(context_to_use, original_answer)}
                last_messages_location.append(last_message_location)
                all_location_questions.append(question)
                all_location_image_ids.append(sample['image_id'])
                all_location_caption_ids.append(sample['caption_id'])
                all_location_image_paths.append(sample['image_path'])
                all_location_answers.append(original_answer)


        messages_date, messages_location = assemble_prompt_knowledge_validation()

        date_input_texts = tokenizer.apply_chat_template(
            [messages_date + [p] for p in last_messages_date],
            tokenize=False,
        )

        location_input_texts = tokenizer.apply_chat_template(
            [messages_location + [p] for p in last_messages_location],
            tokenize=False,
        )

        date_outputs = model.generate(date_input_texts, sampling_params=sampling_params)
        location_outputs = model.generate(location_input_texts, sampling_params=sampling_params)

        for ix in range(0, len(date_outputs)):
            output = date_outputs[ix].outputs[0].text
            date_out = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace('\n', ' ').strip()
            all_context_answers.append(date_out)

        for ix in range(0, len(location_outputs)):
            output = location_outputs[ix].outputs[0].text
            location_out = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace('\n', ' ').strip()
            all_context_answers.append(location_out)
        
        data = pd.DataFrame({"image_id": all_date_image_ids + all_location_image_ids,
                             "caption_id": all_date_caption_ids + all_location_caption_ids,
                             "all_questions": all_date_questions + all_location_questions,
                             "all_answers": all_date_answers + all_location_answers,
                             "all_context_answers": all_context_answers,
                             "image_path": all_date_image_paths + all_location_image_paths,})

        data.to_csv(output_path,
                    mode="a",
                    index=False,
                    sep=",",
                    header=not os.path.exists(output_path))
                
        torch.cuda.empty_cache()