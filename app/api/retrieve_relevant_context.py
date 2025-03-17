import os
from dataclasses import dataclass
from dotenv import load_dotenv

from openai import AsyncOpenAI
from supabase import Client
from typing import List
import numpy as np
import redis

from redis.commands.search.query import Query

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


async def query_cache(embedding_query: List) -> str:
    """Query the cache for the user query."""
    q = (
        Query("*=>[KNN 3 @embedding $vec AS vector_distance]")
        .return_fields("query", "content", "vector_distance")
        .dialect(2)
    )

    result = redis_client.ft("vector_idx").search(
        q, query_params={"vec": np.array(embedding_query, dtype=np.float32).tobytes()}
    )

    if not result:
        return None

    filtered_results = [doc for doc in result.docs if float(doc["vector_distance"]) < 0.9]

    if not filtered_results:
        return None
    return filtered_results[0]["content"]


async def retrieve_relevant_documentation(embedding: List) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.

    Args:
        user_query: The user's question or query
    Returns:
        A formatted string containing the top  most relevant documentation chunks
    """
    try:
        # Query Supabase for relevant documents
        result = supabase.rpc(
            "match_quran", {"query_embedding": embedding, "match_count": 10}
        ).execute()
        if not result.data:
            return "No relevant documentation found."

        return result.data

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"
