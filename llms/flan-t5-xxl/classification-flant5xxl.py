from dotenv import load_dotenv
load_dotenv()

# get to working directory of llms
import sys
sys.path.append('/pub/anvieyra/copilot/llms/user_modules')
from db import Session, get_data
from table_instances import Dialogues
from helper_functions import label_sentiment

import pandas as pd
import re
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import transformers

dialogue_data = get_data('training')

model_id = 'google/flan-t5-xxl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def generate_t5xxl_sentiment(row: pd.Series) -> pd.Series:

    # extract dialogue from row and place into prompt.
    dialogue = row['dialogue_text']
    prompt = "Given the following dialogue, output a single digit representing the sentiment: " \
            "-1 for negative, 0 for neutral, and 1 for positive. Do not provide any additional " \
            " text or explanation.\n\n{}\n\nSentiment:".format(dialogue)

    # Encode the prompt to tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True).to(device)

    # Generate output from the model
    generated_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True)

    # Decode the generated ids to text
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    row["decoded_output"] = str(decoded_output)

    row = sentiment_label(row)
    return row


xxl_data = dialogue_data.apply(generate_t5xxl_sentiment, axis = 1)
xxl_data.to_csv('flant5xxl.csv')

session = Session()
for index, row in xxl_dialogue_data.iterrows():
    dialogue = session.query(Dialogues).filter(Dialogues.dialogue_id == row["dialogue_id"]).first()

    if dialogue:
        dialogue.actual_sentiment = row["actual_sentiment"]

    if index % 100 == 0:
        session.commit()
        print(f"******** row: {index} **********")

session.commit()
session.close()