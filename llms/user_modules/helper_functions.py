import pandas as pd
import nltk
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
nltk.download('punkt')
nltk.download('wordnet')
from bert_score import score
from nltk.tokenize import word_tokenize
from table_instances import Models, Summaries, Sentiments, Dialogues, SentimentEvaluation
from db import Session


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