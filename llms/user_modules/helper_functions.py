import pandas as pd
import re

# import necessary scoring files
import nltk
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
nltk.download('punkt')
nltk.download('wordnet')
from bert_score import score
from nltk.tokenize import word_tokenize
from user_modules.table_instances import Models, Summaries, Sentiments, Dialogues, SentimentEvaluation
from user_modules.db import Session, engine

# import model module files
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# import key performance stats
import time
import GPUtil
import logging

# Global Variables
model = None
tokenizer = None
sentiment_label = {-1:'negative', 0:'neutral', 1: 'positive'}

def generate_sentiment(row: pd.Series, model_id: str) -> pd.Series:

    # extract dialogue from row and place into prompt.
    dialogue = row['dialogue_text']
    prompt = "Given the following dialogue, output a single digit representing the sentiment: " \
            "-1 for negative, 0 for neutral, and 1 for positive. Do not provide any additional " \
            " text or explanation.\n\n{}\n\nSentiment:".format(dialogue)

    # load model and avoids reloading the model for each row
    global tokenizer, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None or tokenizer is None:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Encode the prompt to tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True).to(device)

    # Record GPU before processing
    gpu_before = GPUtil.getGPUs()[0]
    row['gpu_before'] = GPUtil.getGPUs()[0]
    row['load_before'] = gpu_before.load
    row['mem_before'] = gpu_before.memoryUsed
    row['start_time'] = time.time()

    # Generate output from the model
    generated_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True)

    # Decode the generated ids to text
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Use regular expression to extract the sentiment value
    sentiment_match = re.search(r'Sentiment: (-?\d)', decoded_output)
    sentiment_value = int(sentiment_value.group(1)) if sentiment_match else 0
    row['generated_sentiment'] = sentiment_label[int(sentiment_value)]

    # Record GPU after processing
    row['gpu_after'] = GPUtil.getGPUs()[0]
    row['load_after'] = gpu_after.load
    row['mem_after'] = gpu_after.memoryUsed
    row['end_time'] = time.time()

    return row

def generate_summary(df: pd.DataFrame, prompt: str, model: str) ->pd.DataFrame:
    return True

def compute_summary_scores(row: pd.Series) -> pd.Series:

    # gather the text from the current row
    actual_summary: str = row['actual_summary']
    generated_summary: str = row['generated_summary']

    # generate the rouge score and add to the generated corresponding column
    rouge = Rouge()
    r_scores = rouge.get_scores(generated_summary, actual_summary, avg = True)
    row['rouge_1'] = r_scores['rouge-1']['f']
    row['rouge_2'] = r_scores['rouge-2']['f']
    row['rouge_l'] = r_scores['rouge-l']['f']

    # generate the bert score and add to the generated corresponding column
    bert_p, bert_r, bert_f = score([generated_summary], [actual_summary], lang='en', verbose=False)
    row['bert_precision'] = bert_p.mean().item()
    row['bert_recall'] = bert_r.mean().item()
    row['bert_f1'] = bert_f.mean().item()

    # meteor requires the both generated and actual summaries to be tokenized 
    # before they can be evaluated
    tokenized_actual_summary = word_tokenize(actual_summary)
    tokenized_generated_summary = word_tokenize(generated_summary)

    # generate the meteor score and add to the generated corresponding column
    meteor = meteor_score([tokenized_actual_summary], tokenized_generated_summary)
    row['meteor'] = meteor

    return row