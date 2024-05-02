from flask import Flask, request, jsonify
import regex as re
from huggingface_hub import InferenceClient
import json

app = Flask(__name__)

repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
llm_client = InferenceClient(model=repo_id,timeout=120)

@app.route("/")
def demo():
    return open('./templates/demo.html').read()

@app.route("/generate_output", methods=['POST'])


def generate_output():
    data: dict = request.get_json()
    dialogue: str = data['response']
    response: json = {"sentiment" : generate_sentiment(llm_client, dialogue),
                      "summary": generate_summary(llm_client, dialogue)}
    return jsonify(response)

"""
Generates the sentiment of the text
"""
def generate_sentiment(inference_client: InferenceClient, dialogue: str):
    prompt: str = "Given the following dialogue, output a single digit representing the sentiment: " \
                  "-1 for negative, 0 for neutral, and 1 for positive. Do not provide any additional " \
                  " text or explanation.\n\n{}\n\nSentiment:".format(dialogue)
    
    response = inference_client.post(
        json={
            "inputs":prompt,
            "parameters": {"max_new_tokens": 50},
            "task": "text-classification",
        },
    )

    print(response.decode())
    sentiment_value = re.search(r'Sentiment: (-?\d)', response.decode())

    if sentiment_value:
        if int(sentiment_value.group(1)) == -1:
            sentiment_value = "negative"
        elif int(sentiment_value.group(1)) == 1:
            sentiment_value = "positive"
        else:
            sentiment_value = "neutral"
    else:
        sentiment_value = "neutral"

    return sentiment_value

"""
Generates the summary of the text
"""
def generate_summary(inference_client: InferenceClient, dialogue: str):
    prompt: str = "Provide a brief summary of the following text:.\n\n{}\n\nSummarization:".format(dialogue)
    response = inference_client.post(
        json={
            "inputs":prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "summarization",
        },
    )

    summary = re.search(r'Summarization:\s*(.*?)}\]$', response.decode(), re.S)
    if summary:
        summary = summary.group(1).strip()
    else:
        summary = "No summary provided."

    summary = re.sub(r'\s+', ' ', summary)
    summary = re.sub(r'"$', '', summary)
    summary = summary.replace('"', '')
    summary = summary.replace('\\n', '')
    return summary


if __name__ == "__main__":
    app.run(debug=True)