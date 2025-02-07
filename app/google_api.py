import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for chunk in client.models.generate_content_stream(
    model="gemini-2.0-pro-exp-02-05",
    contents="Tell me a story in 300 words.",
    config=types.GenerateContentConfig(
        system_instruction="you are an AI assistant that strictly follows the context provided to answer question. Don't bring any outside information outside of the context provided.",
        temperature=0.1,
        response_mime_type="application/json",
        stop_sequences=["\n"],
        seed=42,
    ),
):
    print(chunk.text)

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_response(
    query: str, context: str = None, model_name: str = "gemini-2.0-pro-exp-02-05"
):
    # --- CONTEXT AND PROMPT ---
    if context is None:  # Handle the case where no context is provided
        context = ""

    system_instruction = """you are an AI assistant that strictly follows the context provided to answer question. 
    Don't bring any outside information outside of the context provided.
    """

    contents = [context, query]

    # --- CONFIGURATION ---
    config = types.GenerateContentConfig(
        # system_instruction is good for overall role/behavior, not specific facts.
        system_instruction=system_instruction,
        temperature=0.1,
        # response_mime_type="application/json",  #  Gemini doesn't reliably produce structured JSON.  Remove this.
        stop_sequences=[
            "\n"
        ],  # Stop sequences might prematurely cut off the response.  Consider removing.
        seed=42,
    )

    # --- GENERATE CONTENT ---
    try:
        for chunk in client.models.generate_content_stream(
            model=model_name,  # Replace with a current model name
            contents=contents,
            config=config,
        ):
            print(chunk.text, end="", flush=True)  # Print incrementally
        print()  # Newline at the end
    except Exception as e:
        print(f"An error occurred: {e}")
