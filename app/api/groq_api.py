import os
import json
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


async def get_filters_groq(query: str, model: str):
    system_prompt = """
    You are an expert in keyword filter extraction from user query. There are two filtration we are looking for:
    -type: A list of type filters. Return empty list if no filters found. Consist of the following options ['index', 'appendix', 'glossary", 'introduction', 'proclamation', 'preface']
    -subType: A list of subType filters. Return empty list if no filters found. Consist of the following options ['index', 'appendix <NUMBER>', 'glossary', 'introduction', 'proclamation', 'preface', 'verse'].
    Examples of appendix options are: appendix 1, appendix one, appendix 28, etc.
    Return only in Json format. Example:{'type': [],'subType': ['verse']}
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_completion_tokens=8192,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    response_content = completion.choices[0].message.content
    print(f"Raw response: {response_content}")

    try:
        # Parse the JSON string into a dictionary
        filters_dict = json.loads(response_content)
        print(f"Parsed filters: {filters_dict}")
        return filters_dict
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        # Return empty filters if JSON parsing fails
        return {"type": [], "subType": []}
