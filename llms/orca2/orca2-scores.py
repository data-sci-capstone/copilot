# import os + os.chir is used to get to the llms directory to use the usermodules 
import os
os.chdir('/pub/anvieyra/copilot/llms/user_modules') 

from  table_instances import Summaries, Sentiments, Base
from  db import Session, get_training_data, add_summaries_data
from helper_functions import compute_summary_scores
import pandas as pd
import matplotlib.pyplot as plt

session = Session()

orca = pd.read_csv("../../Datasets/deciLM_summary_results.csv")

training_data = get_training_data()

merged_df = pd.merge(training_data[["dialogue_id","actual_summary"]], orca, on="dialogue_id", how='inner')

merged_df = merged_df.rename(columns={'summary': 'generated_summary', 'time_taken':'time_summary_taken', 'memory_usage':'memory_summary_usage'})

merged_df["model_id"] = "orca2 7b"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

merged_df = merged_df.apply(compute_summary_scores, axis=1)

merged_df.to_csv("../../Datasets/orca2_scores.csv", index = False)

merged_df = merged_df[['dialogue_id', 'model_id', 'generated_summary', 'rouge_1', 'rouge_2', 'rouge_l', 'bert_precision', 'bert_recall', 'bert_f1', 'meteor', 'memory_summary_usage', 'time_summary_taken']]
add_summaries_data(merged_df)