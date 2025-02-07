import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

system_prompt = "You are an AI assistant that strictly answers from the context given to you." 

def generate_response(query: str, context: str = None, model_name: str = "deepseek-ai/deepseek-r1"):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"{'Context: ' + context + '\\n\\n' if context else ''}{query}",
            }
        ],
        temperature=0.1,
        top_p=0.7,
        max_tokens=4096,
        stream=True,
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
