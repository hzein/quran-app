import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# response = requests.get("https://openrouter.ai/api/v1/models")
# models = []
# for model in response.json()["data"]:
#     if model["id"].endswith("free"):
#         models.append(model["id"])

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

system_prompt = "You are an AI assistant that strictly answers from the context given to you."


async def generate_openrouter(query: str, context: str, model: str):
    completion = client.chat.completions.create(
        # model="google/gemini-2.0-flash-lite-preview-02-05:free",
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{'Context: ' + context + '\\n\\n' if context else ''}{query}",
            },
        ],
        stream=True,
    )

    # print(completion.choices[0].message.content)
    for chunk in completion:
        yield chunk.choices[0].delta.content


# url = "https://openrouter.ai/api/v1/chat/completions"
# headers = {
#   "Authorization": f"Bearer <OPENROUTER_API_KEY>",
#   "Content-Type": "application/json"
# }
# payload = {
#   "model": "openai/gpt-4o",
#   "messages": [{"role": "user", "content": question}],
#   "stream": True
# }
# buffer = ""
# with requests.post(url, headers=headers, json=payload, stream=True) as r:
#   for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
#     buffer += chunk
#     while True:
#       try:
#         # Find the next complete SSE line
#         line_end = buffer.find('\n')
#         if line_end == -1:
#           break
#         line = buffer[:line_end].strip()
#         buffer = buffer[line_end + 1:]
#         if line.startswith('data: '):
#           data = line[6:]
#           if data == '[DONE]':
#             break
#           try:
#             data_obj = json.loads(data)
#             content = data_obj["choices"][0]["delta"].get("content")
#             if content:
#               print(content, end="", flush=True)
#           except json.JSONDecodeError:
#             pass
#       except Exception:
#         break
