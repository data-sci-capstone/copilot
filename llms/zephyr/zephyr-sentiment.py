from dotenv import load_dotenv
load_dotenv()

# get to working directory of llms
import sys
sys.path.append('/data/homezvol2/denisw1/copilot/llms/user_modules')
from db import Session, get_training_data, add_sentiments_data
from helper_functions import generated_decoded_output, find_sentiment
import pandas as pd

# get training data for sentiment analysis
training_data = get_training_data()

prompt = (
		"Given the following dialogue, output a single digit representing the sentiment label: " \
  "-1 for negative, 0 for neutral, and 1 for positive. Do not provide any additional " \
  " text or explanation. \n\n{}\n\nSentiment:\n"
)

# generated a decoded output column labelled generated_sentiment
unproc_zephyr = training_data.apply(generated_decoded_output, axis = 1, model_id = "HuggingFaceH4/zephyr-7b-beta", prompt_template = prompt)

# add to a csv file just in case
unproc_zephyr.to_csv("zephyr-sentiment.csv")

# post processing that will overwrite and correctly label the sentiment
proc_zephyr = unproc_zephyr.apply(find_sentiment, axis = 1)

# add column with the model id
proc_zephyr["model_id"] = "zephyr 7b beta"

# prepare for uploading 
proc_zephyr = proc_zephyr[["dialogue_id", "model_id", "generated_sentiment", "memory_sentiment_usage", "time_sentiment_taken"]]

# add data to postgre sql server
add_sentiments_data(proc_zephyr)