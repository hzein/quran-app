from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
)

system_prompt = "You are an AI assistant that strictly answers from the context given to you." 

def generate_response(query: str, context: str = None):
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
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



from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="deepseek-r1-distill-llama-70b",
    messages=[
        {
            "role": "user",
            "content": ""
        }
    ],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=0.95,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
