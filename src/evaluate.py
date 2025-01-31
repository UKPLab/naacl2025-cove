import argparse
import pandas as pd
import spacy
from evaluation.evaluation_metrics import *
from evaluation.geonames_collection import *
from utils import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate context and veracity prediction scores')
    parser.add_argument('--dataset', type=str,  default= "newsclippings", choices=['newsclippings', '5pils-ooc'],
                        help='The dataset to evaluate.')
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 
    parser.add_argument('--results_path', type=str,  required = True,
                        help='The path to the results file.')    
    parser.add_argument('--geonames_username', type=str,  required = True,
                        help='Geonames username used for geolocation eval.')                   
    
    args = parser.parse_args()
    
    results_file = pd.read_csv(f"results/{args.results_path}")

    nlp = spacy.load("en_core_web_lg")

    #Load ground truth context
    context_ground_truth = pd.DataFrame(load_json(f"data/{args.dataset}/context_{args.dataset}_{args.split}.json"))
    data = pd.merge(results_file, context_ground_truth, on='image_path',how='inner')
    #Context evaluation
    for q in ['people', 'things', 'event', 'date', 'location', 'motivation', 'source']:
        context_results = []
        for i in range(len(results_file)):
            if data.loc[i,q]!= 'not enough information':
                #The ground truth is not NEI
                context_results.append({'ground_truth': data.loc[i,q], 'output':get_context_answer(data.loc[i,'predicted_context_QA'],q)})
    

        #Compute score
        context_evaluation(context_results, 
                           q, 
                           spacy_model=nlp, 
                           geonames_data=f"evaluation/geonames_results_{args.dataset}.json", 
                           geonames_username=args.geonames_user_name, 
                           sleep_geonames=2)
        
    
    #Veracity evaluation
    veracity_results = results_file[['true_veracity','predicted_veracity']].to_dict(orient="records")
    accuracy, recall_acc, recall_ooc, f1 = veracity_evaluation(veracity_results)
    print(f"Accuracy score {accuracy}")
    print(f"Recall accurate score {recall_acc}")
    print(f"Recall out-of-context score {recall_ooc}")
    print(f"Macro F1 score {f1}")