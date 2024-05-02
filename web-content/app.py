from flask import Flask, request, jsonify
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
    print(f"This is the data: {dialogue}")
    response = generate_sentiment(llm_client, dialogue)
    print(response)
    return response

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
    return json.loads(response.decode())[0]["generated_text"]


def generate_summary(inference_client: InferenceClient, dialogue: str):
    return True


if __name__ == "__main__":
    app.run(debug=True)