from dotenv import load_dotenv
load_dotenv()

import os
from sqlalchemy import create_engine
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from tqdm import tqdm
import pandas as pd

def split_dialogue(row, max_length=1024):
    text = row['dialogue_text']
    if len(text) > max_length:
        # Split by newline, but reassemble if any segment is longer than max_length characters
        parts = text.split('\n')
        new_texts = []
        current_part = ''
        for part in parts:
            if len(current_part) + len(part) < max_length:
                current_part += part + '\n'
            else:
                new_texts.append(current_part.strip())
                current_part = part + '\n'
        if current_part:
            new_texts.append(current_part.strip())
        return [(row['dialogue_id'], new_text) for new_text in new_texts]
    else:
        return [(row['dialogue_id'], text)]

def split_long_dialogues(df, max_length=1024):
    new_rows = []
    for _, row in df[df["dialogue_text"].str.len() >= max_length].iterrows():
        results = split_dialogue(row,max_length=max_length)
        new_rows.extend(results)
    return pd.concat([df[df["dialogue_text"].str.len() < max_length][['dialogue_id', 'dialogue_text']], 
                    pd.DataFrame(new_rows, columns=['dialogue_id', 'dialogue_text'])])

def generate_summary(model, tokenizer, dialogues, dialogue_ids, batch_size=5, device="cpu", 
                        output_file="./Datasets/mistral_summary_results.csv", memory_offset=0):

    for i in tqdm(range(0, len(dialogues), batch_size), desc="Summarizing Dialogues"):
        batch = dialogues[i:i+batch_size]
        prompts = [f"Summarize the following dialogue:\n{dialogue}\n Summary:" for dialogue in batch]

        start_time = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        
        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                                 max_length=1024).to(device)
        
        with torch.no_grad():
            model_outputs = model.generate(**model_inputs, max_new_tokens=200, do_sample=False, 
                                           pad_token_id=tokenizer.eos_token_id)
        
        end_time = time.time()
        # Takes the average time of each batch
        # for _ in range(len(batch)):
        #     times.append((end_time - start_time) / len(batch))
        time_per_dialogue = (end_time - start_time) / len(batch)

        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
            # gpu_usages.append(peak_memory)
            # Memory usage takes the average of each batch
            memory_per_dialogue = (peak_memory - memory_offset) / len(batch)

        summaries = []
        for output in model_outputs: 
            summary = tokenizer.decode(output, skip_special_tokens=True)
            # Extract summary from model output
            prompt_end_index = summary.find("Summary:") + len("Summary:")

            summary = summary[prompt_end_index:].strip().split("\n")[0]

            if prompt_end_index == len("Summary:")-1:
                summary = ""
            
            summaries.append(summary)

        batch_results = pd.DataFrame({
            'dialogue_id': dialogue_ids[i:i+batch_size],
            'summary': summaries,
            'time_taken': [time_per_dialogue] * len(batch),
            'memory_usage': [memory_per_dialogue] * len(batch)
        })

        if i == 0:
            batch_results.to_csv(output_file, mode='w', header=True, index=False)
        else:
            batch_results.to_csv(output_file, mode='a', header=False, index=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "google/gemma-7b-it"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# establish connection
engine = create_engine(os.getenv('POSTGRE_DB_URL'))
# read the dialogues table into a pandas dataframe
dialogues_df = split_long_dialogues(pd.read_sql("SELECT * FROM dialogues WHERE dataset = 'training';", engine), max_length=1024)

num_dialogues_to_summarize = 99999
output_file = "./Datasets/gemma_summary_results.csv"
# output_file = "test.csv"
batch_size = 25

dialogues = dialogues_df["dialogue_text"][:num_dialogues_to_summarize].tolist()
dialogue_ids = dialogues_df["dialogue_id"][:num_dialogues_to_summarize].tolist()

pre_training_memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

generate_summary(model, tokenizer, dialogues, dialogue_ids, batch_size=batch_size, 
                    device=device, output_file=output_file, memory_offset=pre_training_memory_usage)

df = pd.read_csv(output_file)

order_df = pd.DataFrame(df.drop_duplicates(subset=["dialogue_id"]).drop(columns=["Summary"]).reset_index(drop=True))

# order_df.merge(df.groupby('dialogue_id')['Summary'].apply(' '.join), on="dialogue_id").to_csv(output_file, mode='w', header=True, index=False)

merged_df = df.groupby('dialogue_id').agg(
    summary=('summary', ''.join),
    time_taken=('time_taken', 'sum'),  # Replace 'Time_Column' with your actual column name
    memory_usage=('memory_usage', 'sum')

).reset_index().to_csv(output_file, mode='w', header=True, index=False)

f = open("./llms/gemma/gemma_pre_training_memory_usage.txt", "w")
f.write(f"pre_training_memory_usage: {pre_training_memory_usage} MB")
f.close()