import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


response = requests.get("https://openrouter.ai/api/v1/models")
models = []
for model in response.json()["data"]:
    if model["id"].endswith("free"):
        models.append(model["id"])


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def generate(query: str, model_name: str):
    completion = client.chat.completions.create(
        # model="google/gemini-2.0-flash-lite-preview-02-05:free",
        model=model_name,
        messages=[{"role": "user", "content": "What is the meaning of life?"}],
        stream=True,
    )

    # print(completion.choices[0].message.content)
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)


for model in models[:5]:
    print(f"Model: {model}")
    generate("What is the meaning of life?", model)
    print("\n")
    print("-" * 50)
    print("\n")
