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
unproc_falcon = training_data.apply(generated_decoded_output, axis = 1, model_id = "tiiuae/falcon-7b-instruct")

# add to a csv file just in case
unproc_falcon.to_csv("falcon-sentiment.csv")

# post processing that will overwrite and correctly label the sentiment
proc_falcon = unproc_falcon.apply(find_sentiment, axis = 1)

# add column with the model id
proc_falcon["model_id"] = "falcon 7b"

# prepare for uploading 
proc_falcon = proc_falcon[["dialogue_id", "model_id", "generated_sentiment", "memory_sentiment_usage", "time_sentiment_taken"]]

# add data to postgre sql server
add_sentiments_data(proc_falcon)