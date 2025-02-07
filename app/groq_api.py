import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

system_prompt = "You are an AI assistant that strictly answers from the context given to you." 

def generate_response(query: str, context: str = None, model_name: str = "deepseek-r1-distill-llama-70b"):
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
        max_completion_tokens=4096,
        top_p=0.95,
        stream=True,
        stop=None,
    )

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")


generate_response("Explain how AI works", context="AI is a field of computer science that focuses on the creation of machines that can perform tasks that require human intelligence. This includes tasks such as visual perception, speech recognition, decision-making, and language translation. AI systems can be designed to learn from data, adapt to new inputs, and perform human-like tasks. Machine learning is a subset of AI that focuses on the development of algorithms that can learn from and make predictions or decisions based on data. Deep learning is a subset of machine learning that uses neural networks with many layers to model and make sense of complex data. Neural networks are a type of AI model that is inspired by the structure of the human brain. They are composed of layers of nodes that process and transform data, and they can be trained to recognize patterns in data and make predictions based on new inputs.")