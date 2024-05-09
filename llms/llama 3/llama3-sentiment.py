from dotenv import load_dotenv
load_dotenv()

# get to working directory of llms
import sys
sys.path.append('/pub/anvieyra/copilot/llms/user_modules')
from db import Session, get_training_data, add_sentiments_data
from helper_functions import generated_decoded_output, find_sentiment
import pandas as pd

# get training data for sentiment analysis
training_data = get_training_data()

# generated a decoded output column labelled generated_sentiment
unproc_llama3 = training_data.apply(generated_decoded_output, axis = 1, model_id = "meta-llama/Meta-Llama-3-8B-Instruct")

# add to a csv file just in case
unproc_llama3.to_csv("llama3-sentiment.csv")

# post processing that will overwrite and correctly label the sentiment
proc_llama3 = unproc_llama3.apply(find_sentiment, axis = 1)

# add column with the model id
proc_llama3["model_id"] = "llama 3 8b"

# prepare for uploading 
proc_llama3 = proc_llama3[["dialogue_id", "model_id", "generated_sentiment", "memory_sentiment_usage", "time_sentiment_taken"]]

# add data to postgre sql server
add_sentiments_data(proc_llama3)