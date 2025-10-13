import pandas as pd
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score
import evaluate
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import set_seed
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from tqdm import tqdm
import torch.nn.functional as F 
import json
import argparse


def create_input_and_labels(df):
    inputs = [f'Context: {i}  | Caption to verify: {j}' for i, j in zip(df['predicted_context_QA'], df['caption'])]
    labels = [1 if l == 'pristine' else 0 for l in df['true_veracity']]
    return inputs, labels

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Fine-tune deberta model on newsclippings.')
    parser.add_argument('--dataset', type=str, default='newsclippings', choices=['newsclippings', '5pils-ooc'],
                        help='The test set on which to evaluate the performance.')
    parser.add_argument('--split', type=str,  default= "test", choices=['train', 'val', 'test'],
                        help='The dataset split to use.') 
    parser.add_argument('--model_path', type=str, default = "",
                        help='Name of the location to save the model.')
    parser.add_argument('--train_model', type=int, default=1,
                        help='Whether to fine-tune a model or use an existing local checkpoint (model_path)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay value')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of instance per batch')
    parser.add_argument('--epoch', type=int, default=5, 
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    args = parser.parse_args()

    set_seed(args.seed)

    #Load data
    train = pd.read_csv('results/newsclippings_train.csv') 
    val = pd.read_csv('results/newsclippings_val.csv')
    if args.dataset=='newsclippings':
        test = pd.read_csv('results/newsclippings_test.csv') 
    else:
        test  = pd.read_csv('results/5pils_ooc_test.csv')

    train_input, train_label = create_input_and_labels(train)
    val_input, val_label = create_input_and_labels(val)
    test_input, test_label = create_input_and_labels(test,False)


    data_dict = {
        "train": {"text": train_input, "label": train_label},
        "validation": {"text": val_input, "label": val_label},
        "test_news": {"text": test_input, "label": test_label}
    }

    train_dataset = Dataset.from_dict(data_dict['train'])
    val_dataset = Dataset.from_dict(data_dict['validation'])
    test_dataset = Dataset.from_dict(data_dict['test'])

    # Combine into a DatasetDict
    hf_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    if args.train_model:
        #Load the huggingface base model to be fine-tuned
        model_path = "microsoft/deberta-v3-large"
    else:
        #Load the already fine-tuned model
        model_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_dataset = hf_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    id2label = {0: "falsified", 1: "pristine"}
    label2id = {"falsified": 0, "pristine": 1}


    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, id2label=id2label, label2id=label2id
    )

    if args.train_model:
        training_args = TrainingArguments(
            output_dir=args.model_path,
            learning_rate=args.learning_rate, 
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size, 
            num_train_epochs=args.epoch,
            weight_decay=args.weight_decay, 
            eval_strategy="epoch",
            save_strategy="epoch", 
            warmup_steps=500,              
            save_total_limit=1,                  
            load_best_model_at_end=True,  
            metric_for_best_model="eval_accuracy",   
            greater_is_better=True,     
            logging_dir="logs",                  
            logging_strategy="epoch",            
            save_steps=None,                     
            push_to_hub=False,
            seed=42
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() 

        #Predictions on test set if a fine-tuned model is provided
        predictions = []
        true_labels = []
        probabilities = []
        for example in tqdm(tokenized_dataset[args.split]):
            inputs = {
                'input_ids': torch.tensor(example['input_ids']).unsqueeze(0).to(device),  # Add batch dimension
                'attention_mask': torch.tensor(example['attention_mask']).unsqueeze(0).to(device)  # Add batch dimension
            }
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_id = logits.argmax(dim=-1).item()
                probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()[0].astype(float)
            
            #Store the predictions and true labels
            predictions.append(predicted_class_id)
            probabilities.append(probs)
            true_labels.append(example['label'])



        accuracy = accuracy_score(true_labels, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        #Prepare the results in a structured format
        results = []
        for i, predicted_class_id in enumerate(predictions):
            predicted_veracity = model.config.id2label[predicted_class_id]
            true_veracity = model.config.id2label[true_labels[i]]
            score = probabilities[i]
            
            results.append({
                'index': i,
                'predicted_veracity': predicted_veracity,
                'true_veracity': true_veracity,
                'score': score
            })


        filename = f'results/deberta_{args.dataset}_{args.split}.json'
        with open(filename, 'w') as file:
            json.dump(results, file, indent=4)   