from flask import Flask, request, jsonify
import regex as re
from huggingface_hub import InferenceClient
import json

app = Flask(__name__)

REPO_ID: str = "mistralai/Mistral-7B-Instruct-v0.1"
LLM_CLIENT: InferenceClient = InferenceClient(token="replace_with_your_token_here", model=REPO_ID,timeout=120)
MAX_DIALOGUE_LENGTH: int = 1024

@app.route("/")
def demo():
    return open('./templates/demo.html').read()

@app.route("/generate_output", methods=['POST'])


def generate_output() -> json:
    data: dict = request.get_json()
    dialogue: str = data['response']

    prompt_sentiment: str = "Given the following dialogue, output a single digit representing the sentiment: " \
                  "-1 for negative, 0 for neutral, and 1 for positive. Do not provide any additional " \
                  " text or explanation.\n\n{}\n\nSentiment:".format(dialogue)
    
    prompt_summary: str = "Provide a brief summary of the following text:.\n\n{}\n\nSummarization:".format(dialogue)

    response: json = {"sentiment" : generate_sentiment(LLM_CLIENT, prompt_sentiment),
                      "summary": generate_summary(LLM_CLIENT, prompt_summary)}
    return jsonify(response)

"""
Generates the sentiment of the text
"""
def generate_sentiment(inference_client: InferenceClient, dialogue: str) -> str:
    
    response = inference_client.post(
        json={
            "inputs":dialogue,
            "parameters": {"max_new_tokens": 50},
            "task": "text-classification",
        },
    )
    response = json.loads(response.decode())[0]["generated_text"]

    sentiment_value = re.search(r'Sentiment: (-?\d)', response)

    sentiment = {-1:'negative', 0:'neutral', 1: 'positive'}
    if sentiment_value:
        sentiment_value = int(sentiment_value.group(1))
    else:
        sentiment_value = 0

    return sentiment[sentiment_value]

"""
Generates the summary of the text
"""
def generate_summary(inference_client: InferenceClient, dialogue: str) -> str:


    # check if prompt exceed 1024 characters (roughly 256 tokens)
    if len(dialogue) >= MAX_DIALOGUE_LENGTH:
        return generate_long_summary(dialogue)

    response = inference_client.post(
        json={
            "inputs":dialogue,
            "parameters": {"max_new_tokens": 1000},
            "task": "summarization",
        },
    )

    response: str = json.loads(response.decode())[0]["generated_text"]
    print(f"response:\n{response}")

    summary: str = re.search(r'Summarization:\s*\n\n(.*)', response, re.S)

    if summary:
        summary: str = summary.group(1).strip()
        print("Summary:", summary)
    else:
        print("No summary found.")

    return "" if summary == None else summary

"""
function dedicated to producing summaries for excessively long dialogues
"""
def generate_long_summary(prompt: str) -> str:
    list_of_split_summaries: set = set()

    parts: list = prompt.split('\n')
    current_part: str = ''

    for part in parts:

        if len(current_part) + len(part) < MAX_DIALOGUE_LENGTH:
            current_part += part + '\n'
        else:
            if current_part:
                list_of_split_summaries.add(generate_summary(LLM_CLIENT, current_part))
                current_part = part + '\n'
    
    if current_part:
        list_of_split_summaries.add(generate_summary(LLM_CLIENT, current_part))

    long_generated_summary = '\n'.join(list(list_of_split_summaries))
    print(f"List of Split Summaries: {long_generated_summary}")

    return long_generated_summary

if __name__ == "__main__":
    app.run(debug=True)