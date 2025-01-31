from evaluate import load
from dateutil import parser
import numpy as np
from scipy.optimize import linear_sum_assignment
from dateutil.tz import tzutc
from dateutil.relativedelta import relativedelta
from haversine import haversine, Unit
from itertools import combinations
from geonames_collection import *
from utils import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score

#Load the metrics
meteor = load('meteor')
rouge = load('rouge')
bertscore = load("bertscore")


def veracity_evaluation(results):
    #Loop through results and cumulate the scores before taking the average
    accuracy = []
    recall_acc = []
    recall_ooc = []
    f1 = []
    for r in results:
        accuracy.append(accuracy_score(r['true_veracity'], r['predicted_veracity']))
        recall_acc.append(recall_score(r['true_veracity'], r['predicted_veracity']),pos_label='pristine')
        recall_ooc.append(recall_score(r['true_veracity'], r['predicted_veracity']),pos_label='falsified')
        f1.append(f1_score(r['true_veracity'], r['predicted_veracity'], average='macro'))
    
    accuracy = sum(accuracy)/len(accuracy)
    recall_acc = sum(recall_acc)/len(recall_acc)
    recall_ooc = sum(recall_ooc)/len(recall_ooc)
    f1 = sum(f1)/len(f1)
    return accuracy, recall_acc, recall_ooc, f1


def extract_named_entities(text, model, entity_type):
    '''
    Return a list of entities of a certain type contained in a string.
    Params:
        text (str) : the text string
        model (object) : the spaCy NLP model used to process the text
        entity_type (str) : the type of entity to search for. One of ["date_and_times", "locations"]
    '''
    # Process the input text using spaCy
    doc = model(text)
    # Initialize a list to store the extracted entities
    entities = []
    current_entity = []

    # Define a mapping of entity type names to spaCy labels
    entity_type_map = {
        "people": ["PERSON"],
        "dates_and_times": ["DATE", "TIME"],
        "locations": ["LOC", "GPE"]
    }
    # Iterate through the tokens in the processed text
    for token in doc:
        # Check if the token is an entity of the specified type
        if token.ent_type_ in entity_type_map[entity_type]:
            if token.ent_iob_ == 'B':  # Beginning of an entity
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
                current_entity.append(token.text)
            else:  # Inside or last token of an entity
                current_entity.append(token.text)
    # Add the last entity if the sentence ends with one
    if current_entity:
        entities.append(' '.join(current_entity))     
    return entities


def get_numeric_date_label(date,spacy_model):
    '''
    Convert dates to numeric labels
    '''
    date_NER = extract_named_entities(date,spacy_model, "dates_and_times")
    output=[]
    for d in date_NER:
        try:
            output.append(parser.parse(d).replace(tzinfo=tzutc()))
        except:
            pass
    if len(output)==0:
        output='not enough information'
    else:
        output = [d.isoformat() for d in output]
    return output



def date_distance(dt1, dt2):
    '''
    Compute the distance between two dates
    '''
    if dt1.tzinfo is None:
        dt1 = dt1.replace(tzinfo=tzutc())
    if dt2.tzinfo is None:
        dt2 = dt2.replace(tzinfo=tzutc())
    if dt1 is None or dt2 is None:
        return float('inf')

    delta = relativedelta(dt1, dt2)
    return abs(delta.years + delta.months / 12 + delta.days / 365.25)


def location_coordinate_distance(coordinates1,coordinates2,unit=1000):
    '''
    Compute the coordinates distance between the prediction and ground truth. 
    Compare all pairs of GeoNames entities and take the smallest distance as optimist heuristic.
    '''
    d = min([haversine(c1,c2,unit=Unit.KILOMETERS) for c1 in coordinates1 for c2 in coordinates2])
    d /= unit
    return d


def hierarchical_distance_metric(pred_hierarchy, gt_hierarchy):
    '''
    Compute the distance between two hierarchies based on their common parent.
    '''
    if  all(i in pred_hierarchy for i in gt_hierarchy):
        return 0
    else:
        common_length = 0
        for p, g in zip(pred_hierarchy, gt_hierarchy):
            if p == g:
                common_length += 1
            else:
                break
        return len(pred_hierarchy) + len(gt_hierarchy) - 2 * common_length


def location_hierarchy_distance(hierarchy1,hierarchy2):
    '''
    Compute the distance between two locations given their GeoNames hierarchies.
    Compare all pairs of GeoNames entities and take the smallest distance as optimist heuristic.
    '''
    d = min([hierarchical_distance_metric(h1,h2) for h1 in hierarchy1 for h2 in hierarchy2])
    return d


def is_strict_subset(sublist, mainlist):
    '''
    Compute whether the content of one list is the subset of the content of another list.
    '''
    return set(sublist).issubset(set(mainlist)) and len(sublist) < len(mainlist)


def find_locations_to_remove(l):
    '''
    Remove locations for which the hierarchy is a strict subset of another location.
    '''
    indices_to_remove = []
    
    def contains_strict_subset(outer_list, other_lists):
        for sublist in outer_list:
            for other_list in other_lists:
                for other_sublist in other_list:
                    if is_strict_subset(sublist, other_sublist):
                        return True
        return False

    for i, outer_list in enumerate(l):
        if contains_strict_subset(outer_list, [other_list for j, other_list in enumerate(l) if i != j]):
            indices_to_remove.append(i)
    
    return indices_to_remove


def evaluate_context_item(prediction, 
             ground_truth, 
             task, 
             NER_model = None,
             geonames_data_path="geonames_results.json",
             geonames_username=None,
             sleep_geonames=2):
    '''
    Main 5Pils evaluation function.
    Params:
        prediction (str): a string containing the prediction for an image
        ground_truth (str): the ground truth context information
        NER_model (object): a spacy NER model to extract dates and location named entities from predictions.
        geonames_data_path (str): path to the json file storing the geoname entries. Only needed for the location task
        geonames_username (str): user name to connect to the GeoNames API. Only needed for the location task
        sleep_geonames (int): the waiting time in seconds between two calls of the GeoNames API
    '''
    #Source
    if task=="source":
        if not isinstance(prediction, list):
            prediction = [prediction]
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        rouge_result = rouge.compute(predictions=prediction, references=[ground_truth])['rougeL']
        meteor_result = meteor.compute(predictions=prediction, references=[ground_truth])['meteor']
        return {'rougeL':rouge_result,"meteor": meteor_result}
    
    elif task=='person' :
        ground_truth = set([g.lower() for g in ground_truth])
        ground_truth_person_NER = set([l for l in extract_named_entities(' , '.join(ground_truth),NER_model,'people')])
        prediction_person_NER = set([l.lower() for l in extract_named_entities(prediction,NER_model,'people')])
        true_positive = len(ground_truth_person_NER.intersection(prediction_person_NER))
        false_positive = len(prediction_person_NER.difference(ground_truth_person_NER))
        false_negative = len(ground_truth_person_NER.difference(prediction_person_NER))
    
        # Calculate precision, recall, and F1 score
        precision_result = true_positive / (true_positive+ false_positive) if (true_positive+ false_positive) > 0 else 0
        recall_result = true_positive / (true_positive+ false_negative) if (true_positive+ false_negative) > 0 else 0
        f1_result = 2 * (precision_result * recall_result) / (precision_result + recall_result) if (precision_result + recall_result) > 0 else 0
        return {'recall':recall_result, 'precision': precision_result,'f1': f1_result}

    elif task=='object' :
        if isinstance(ground_truth, list):
            ground_truth = [', '.join(ground_truth)]
        else:
            ground_truth = [ground_truth]
        prediction = [prediction]
        rouge_result = rouge.compute(predictions=prediction, references=[ground_truth])['rougeL']
        meteor_result = meteor.compute(predictions=prediction, references=[ground_truth])['meteor']
        berts_result = bertscore.compute(predictions=prediction,references=ground_truth,
                                        lang='en', model_type="distilbert-base-uncased")['f1'][0]
        return {'rougeL':rouge_result,"meteor": meteor_result, 'BertS':berts_result}


    #Motivation/Event
    elif task=="motivation" or task=="event":
        if not isinstance(prediction, list):
            prediction = [prediction]
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        rouge_result = rouge.compute(predictions=prediction, references=[ground_truth])['rougeL']
        meteor_result = meteor.compute(predictions=prediction, references=[ground_truth])['meteor']
        berts_result = bertscore.compute(predictions=prediction,references=ground_truth,
                                        lang='en', model_type="distilbert-base-uncased")['f1'][0]
        return {'rougeL':rouge_result,"meteor": meteor_result, 'BertS':berts_result}
    
    #Location
    elif task=="location":
        if not isinstance(prediction, list):
            prediction = [prediction]
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        rouge_result = rouge.compute(predictions=prediction, references=[ground_truth])['rougeL']
        meteor_result = meteor.compute(predictions=prediction, references=[ground_truth])['meteor']
        return {'rougeL':rouge_result,"meteor": meteor_result}
    
    elif task == "location NER":
            geonames_data = load_json(geonames_data_path)
            geonames_entries = list(set([d['query'].lower() for d in geonames_data]))
            prediction_location_NER = list(set([l for l in extract_named_entities(prediction,NER_model,'locations')]))
            prediction_coordinates = []
            prediction_hierarchies = []
            matching_records = []
            #Prepare the predictions
            for p in prediction_location_NER:
                if p.lower() not in geonames_entries: 
                        #Add a new entry to the collected GeoName database if the prediction is not there yet
                        matching_records = search_location(p,geonames_username,sleep_geonames)
                        time.sleep(sleep_geonames)
                        print('New location added %s'%p)
                        save_result(matching_records,geonames_data_path)
                        if len(matching_records)==0:
                            save_result({'query':p.lower()},geonames_data_path)
                else:
                    matching_records = [d for d in geonames_data if 'coordinates' in d.keys() and d['query'].lower()==p.lower()]
                if len(matching_records) > 0 :            
                    prediction_coordinates.append([r['coordinates'] for r in matching_records])
                    prediction_hierarchies.append([r['hierarchy'] for r in matching_records])
            ground_truth_location_NER = list(set(ground_truth))
            ground_truth_coordinates = []
            ground_truth_hierarchies = []
            for g in ground_truth_location_NER:
                matching_records = [d for d in geonames_data if 'coordinates' in d.keys() and d['query'].lower()==g.lower()]
                if len(matching_records) > 0 : 
                    ground_truth_hierarchies.append([r['hierarchy'] for r in matching_records])
                    ground_truth_coordinates.append([r['coordinates'] for r in matching_records])
            idx_to_remove  = find_locations_to_remove(ground_truth_hierarchies)
            ground_truth_coordinates = [ground_truth_coordinates[i] for i in range(len(ground_truth_coordinates)) if i not in idx_to_remove]
            ground_truth_hierarchies = [ground_truth_hierarchies[i] for i in range(len(ground_truth_hierarchies)) if i not in idx_to_remove]
            if len(prediction_coordinates) > 0:
                
                if len(prediction_coordinates) > len(ground_truth_coordinates):
                    # Generate all combinations of size up to x
                    candidates = []
                    size = len(ground_truth_coordinates)
                    candidates.extend(combinations(prediction_coordinates, size))
                else: 
                    candidates = [prediction_coordinates]

                best_codelta = 0
                for candidate in candidates:
                    try:
                #We find the minimal distance among all pairs
                        distances = np.array([[location_coordinate_distance(pc, gc) for gc in ground_truth_coordinates] for pc in candidate])           
                        row_ind, col_ind = linear_sum_assignment(distances)
                        scores = 0
                        non_zero_distance_list = [distances[r,c] for r, c in zip(row_ind, col_ind)]
                        non_zero_distance_list = sorted(non_zero_distance_list)
                        for d in non_zero_distance_list:
                            scores += 1/(1+d)
                        
                        coefficient = 1/len(ground_truth_coordinates) 
                        codelta = coefficient *scores
                        if codelta > best_codelta:
                            best_codelta = codelta
                    except:
                        print(prediction_location_NER)
                        print(ground_truth_location_NER)
                        print(candidate)
                        print(ground_truth_coordinates)
                        pass
                
            else:
                best_codelta = 0

            if len(prediction_hierarchies) > 0:
                
                if len(prediction_hierarchies) > len(ground_truth_hierarchies):
                    # Generate all combinations of size up to x
                    candidates = []
                    size = len(ground_truth_hierarchies)
                    candidates.extend(combinations(prediction_hierarchies, size))
                else: 
                    candidates = [prediction_hierarchies]

                best_hierarchy_delta = 0
                for candidate in candidates:
                    try:
                        distances = np.array([[location_hierarchy_distance(pc, gc) for gc in ground_truth_hierarchies] for pc in candidate])          
                        row_ind, col_ind = linear_sum_assignment(distances)
                        scores = 0
                        non_zero_distance_list = [distances[r,c] for r, c in zip(row_ind, col_ind)]
                        non_zero_distance_list = sorted(non_zero_distance_list)
                        for d in non_zero_distance_list:
                            scores += 1/(1+d)
                        coefficient = 1/len(ground_truth_hierarchies) 
                        hierarchy_delta  = coefficient *scores
                        if hierarchy_delta > best_hierarchy_delta :
                            best_hierarchy_delta = hierarchy_delta 
                    except:
                        pass
            else:
                best_hierarchy_delta  = 0
            return {"codelta": best_codelta, "hldelta": best_hierarchy_delta}
    #Date
    elif task == "date":
        if prediction!='':
            if prediction[0]=='[':
                prediction = prediction[1:-1]
        prediction_dates = extract_named_entities(prediction, NER_model,'dates_and_times')
        prediction_dates = [convert_to_date(date_str) for date_str in prediction_dates]
        prediction_dates = [d for d in prediction_dates if d is not None]
        ground_truth_dates = [convert_to_date(date_str) for date_str in ground_truth]
        if len(ground_truth_dates) > 0 and len(prediction_dates) > 0:
            if len(prediction_dates) > len(ground_truth_dates):
                # Generate all combinations of size up to x
                candidates = []
                size = len(ground_truth_dates)
                candidates.extend(combinations(prediction_dates, size))
            else: 
                candidates = [prediction_dates]
            best_delta = 0
            best_EM = 0
            for candidate in candidates:
                distances = np.array([[date_distance(pd, gd) for gd in ground_truth_dates] for pd in candidate])          
                row_ind, col_ind = linear_sum_assignment(distances)
                scores = 0
                non_zero_distance_list = [distances[r,c] for r, c in zip(row_ind, col_ind)]
                non_zero_distance_list = sorted(non_zero_distance_list)
                for d in non_zero_distance_list:
                    scores += 1/(1+d)
                exact_match = np.all(distances[row_ind, col_ind] == 0)
                coefficient = 1/len(ground_truth_dates) 
                delta = coefficient *scores
                if delta > best_delta:
                    best_delta = delta
                    best_EM = exact_match
            
            if len(prediction_dates) > len(ground_truth_dates):
                best_EM=0

            return {"exact_match": best_EM, "delta": best_delta}
        else:
            return {"exact_match": 0, "delta": 0}   
    else:
        raise ValueError("Invalid task name")
    

def get_context_answer(predicted_context,question='people'):
    '''
    Extract a specific context item answer from the predicted context
    '''
    all_questions= ['people: ', 'things: ','event: ','date: ','location: ','motivation: ','source: ']
    if question + ': ' in predicted_context.lower():
        try:
            answer = predicted_context.lower().split(question+': ')[1]
        except:
            answer = predicted_context.lower().replace(question + ': ' ,'')
        
        for q in all_questions:
            if q!= question + ': ':
                if q in answer :
                    answer = answer.split(q)[0]
        if 'unknown.' in answer:
            answer=False
    elif '|||' in predicted_context.lower():
        items = predicted_context.lower().split('|||')
        if len(items)==6 and question=='source':
            return False
        answer = items[all_questions.index(question+': ')]
        if 'unknown.' in answer:
            answer=False
        return answer
    else:
        answer = False
    return answer
    


def context_evaluation(results, task, spacy_model=None, 
                       geonames_data=None, geonames_username=None, sleep_geonames=2):
    rougeL = [] 
    meteor = [] #main
    exact_match = [] 
    delta = [] #main
    codelta = [] #main
    hldelta = [] 
    berts  = []
    recall = []
    precision = []
    f1 = [] #main
    for r in tqdm(results):
        
        if task ==  'source':
            if r['output']==False:
                scores = {'rougeL':0, 'meteor':0}
            else:
                scores = evaluate_context_item(r['output'], r['ground_truth'], task)
            rougeL.append(scores['rougeL'])
            meteor.append(scores['meteor'])
        elif task=='date':
            if r['output']==False:
                scores = {'exact_match':0, 'delta':0}
            else:
                scores = evaluate_context_item(r['output'], r['ground_truth'], task, spacy_model)
            exact_match.append(scores['exact_match'])
            delta.append(scores['delta'])
        elif task=='location':
            
            geonames_entries = set([d['query'] for d in load_json(geonames_data)])
            if type(r['ground_truth'])!=list:
                NER_ground_truth = [e for e in extract_named_entities(r['ground_truth'], spacy_model, 'locations')  if e in geonames_entries]
            else:
                NER_ground_truth = [e for e in extract_named_entities(' , '.join(r['ground_truth']), spacy_model, 'locations')  if e in geonames_entries]
            if r['output']==False:
                scores = {'rougeL':0, 'meteor':0}
            else:
                scores = evaluate_context_item(r['output'], r['ground_truth'], task)
            if len(NER_ground_truth) > 0:
                if r['output']==False:
                    NER_scores = {'codelta':0, 'hldelta':0}
                else:
                    #Compute the location delta metrics too because the ground truth has Geonames entries
                    NER_scores = evaluate_context_item(r['output'], NER_ground_truth, 'location NER', spacy_model, geonames_data, geonames_username, sleep_geonames)
                #codelta and hldelta is only reported for the subset with GeoNames entries
                codelta.append(NER_scores['codelta'])
                hldelta.append(NER_scores['hldelta'])
                if NER_scores['hldelta'] == 1:
                    #If there is an exact match in terms of GeoNames entries, i.e., HLDelta is equal to 1, then we count the prediction has accurate.
                    scores['meteor']=1
                    scores['rougeL']=1
            rougeL.append(scores['rougeL'])
            meteor.append(scores['meteor'])
        elif task ==  'motivation':
            if r['output']==False:
                scores = {'rougeL':0, 'meteor':0, 'BertS':0}
            else:
                scores = evaluate_context_item(r['output'], r['ground_truth'], task)
            rougeL.append(scores['rougeL'])
            meteor.append(scores['meteor'])
            berts.append(scores['BertS'])
        elif task ==  'event':
            if r['output']==False:
                scores = {'rougeL':0, 'meteor':0, 'BertS':0}
            else:
                scores = evaluate_context_item(r['output'], r['ground_truth'], task)
            rougeL.append(scores['rougeL'])
            meteor.append(scores['meteor'])
            berts.append(scores['BertS'])
        elif task ==  'person':
            
            if r['output']==False:
                scores = {'recall':0, 'precision':0, 'f1':0}
            else:
                scores = evaluate_context_item(r['output'], r['ground_truth'], task, spacy_model)
            NER_ground_truth = set([l for l in extract_named_entities(' , '.join(r['ground_truth']),spacy_model,'people')])
            if len(NER_ground_truth)>0:
                recall.append(scores['recall'])
                precision.append(scores['precision'])
                f1.append(scores['f1'])
        elif task ==  'object':
            if r['output']==False:
                scores = {'rougeL':0, 'meteor':0, 'BertS':0,}
            else:
                scores = evaluate_context_item(r['output'], r['ground_truth'], task)
            berts.append(scores['BertS'])
            rougeL.append(scores['rougeL'])
            meteor.append(scores['meteor'])
        else:
            print('Invalid task name')
            break

    print('------------------------')
    print('Evaluation for task: ' + task)
    print('------------------------')
    if task=='source':
        print('RougeL score %s'%np.mean(rougeL))
        print('Meteor score %s'%np.mean(meteor))
        return rougeL, meteor
    elif task=='date':
        print('EM score %s'%np.mean(exact_match))
        print('Delta score %s'%np.mean(delta))
        return exact_match, delta
    elif task=='location':
        print('RougeL score %s'%np.mean(rougeL))
        print('Meteor score %s'%np.mean(meteor))
        print('OCDelta score %s'%np.mean(codelta))
        print('HL Delta score %s'%np.mean(hldelta))
        return rougeL, meteor, codelta, hldelta
    elif task=='motivation' or task=='event':
        print('RougeL score %s'%np.mean(rougeL))
        print('Meteor score %s'%np.mean(meteor))
        print('Bert score %s'%np.mean(berts))
        return rougeL, meteor, berts
    elif task=='object':
        print('RougeL score %s'%np.mean(rougeL))
        print('Meteor score %s'%np.mean(meteor))
        print('Bert score %s'%np.mean(berts))
        return rougeL, meteor, berts
    elif task=='person':
        print('Recall score %s'%np.mean(recall))
        print('Precision score %s'%np.mean(precision))
        print('F1 score %s'%np.mean(f1))
        return recall, precision, f1
    else:
        print('Invalid task name')