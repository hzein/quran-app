import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

system_prompt = "You are an AI assistant that strictly answers from the context given to you."


async def generate_groq(query: str, context: str, model: str):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{'Context: ' + context + '\\n\\n' if context else ''}{query}",
            },
        ],
        temperature=0.1,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=False,
        stop=None,
    )

    response = completion.choices[0].message.content
    return response

    # for chunk in completion:
    #     if chunk.choices[0].delta.content is not None:
    #         yield chunk.choices[0].delta.content
