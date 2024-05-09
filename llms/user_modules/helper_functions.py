import pandas as pd
import re

# import necessary scoring files
import nltk
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from bert_score import score
from nltk.tokenize import word_tokenize
from table_instances import Models, Summaries, Sentiments, Dialogues, SentimentEvaluation
from db import Session, engine

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
initial_memory_usage = None

def generated_decoded_output(row: pd.Series, model_id: str) -> pd.Series:

    # extract dialogue from row and place into prompt.
    dialogue = row['dialogue_text']
    prompt = "Given the following dialogue, output a single digit representing the sentiment label: " \
            "-1 for negative, 0 for neutral, and 1 for positive. Do not provide any additional " \
            " text or explanation.\n\n{}\n\nSentiment:".format(dialogue)

    # load model and avoids reloading the model for each row
    global tokenizer, model, initial_memory_usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None or tokenizer is None:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Encode the prompt to tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True).to(device)

    # Record initial memory usage
    torch.cuda.reset_peak_memory_stats(device)
    if initial_memory_usage is None:
        initial_memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    start_time = time.time()

    # Generate output from the model
    generated_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)

    # Decode the generated ids to text
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    row["generated_sentiment"] = decoded_output

    # Record GPU after processing
    row['memory_sentiment_usage'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    row['time_sentiment_taken'] = time.time() - start_time

    return row

def find_sentiment(row: pd.Series) -> pd.Series:
    # # Use regular expression to extract the sentiment value
    decoded_output = row["generated_sentiment"].lower()

    # find first occurence of -1, 0, 1 immediately after "Sentiment:"
    sentiment_match = re.search(r"sentiment:\s*(1|0|-1)\b", decoded_output)

    sentiment_label = {'-1':'negative', '0':'neutral', '1': 'positive'}

    if sentiment_match:
        sentiment_value = sentiment_match.group(1)
        row["generated_sentiment"] = sentiment_label[sentiment_value]
    else:
        # if sentiment_match is not found, search for actual labels
        sentiment_match = re.search(r"sentiment:.*(positive|neutral|negative)", 
            decoded_output, re.DOTALL)
        if sentiment_match:
            row["generated_sentiment"] = str(sentiment_match.group(1))

        else:
            row["generated_sentiment"] = 'neutral'

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