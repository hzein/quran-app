from app.api.google_api import generate_google
from app.api.groq_api import generate_groq
from app.api.openrouter_api import generate_openrouter


def concatenate_contents(context):
    """
    Concatenates the 'content' field from each item in the API response.

    Args:
      context: A list of dictionaries, where each dictionary represents
                    an item from the API response and has a 'content' key.

    Returns:
      A string containing the concatenated contents from all items in the
      API response.
    """
    concatenated_content = ""
    array_of_contents = []
    for item in context:
        if "content" in item:
            concatenated_content += f"{item['content']} \n"
            array_of_contents.append(f"{item['content']} \n")
    return concatenated_content, array_of_contents


def get_approproate_generate_func(model: str) -> str:
    """
    Get the correct generate function based on the provider.

    Args:
      model: A string containing the model to use for generation.format(MODEL_PROVIDER:MODEL_NAME)

    Returns:
      The function to call
    """
    provider_name, model_name = model.split(":", maxsplit=1)

    if model_name == "free":
        raise ValueError(f"Invalid provider name: {provider_name}")

    if provider_name == "google":
        return model_name, generate_google
    elif provider_name == "openrouter":
        return model_name, generate_openrouter
    elif provider_name == "groq":
        return model_name, generate_groq
    else:
        raise ValueError(f"Invalid provider name: {provider_name}")


def generate_redis_chunk_response(text: str):
    for char in text:
        # Simulate data generation and yield data as bytes.
        yield f"{char}".encode("utf-8")


def generate_redis_response(text: str):
    return text
