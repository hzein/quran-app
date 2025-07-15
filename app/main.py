from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

from app.api.retrieve_relevant_context import (
    get_embedding,
    retrieve_relevant_documentation,
    retrieve_relevant_documentation_ids,
    query_cache,
    set_cache,
    delete_cache,
)

# Generate response
from app.utils.utils import (
    concatenate_contents,
    get_approproate_generate_func,
    generate_redis_response,
)

load_dotenv()

# Authentication setup
# Instead of OAuth2 we use a simple API key authentication with HTTPBearer.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "mydefaultapikey")
USER_API_KEY = os.getenv("USER_API_KEY", "mydefaultapikey")
# bearer_scheme = HTTPBearer()
X_API_KEY = APIKeyHeader(name="X-API-Key")


def api_key_auth(x_api_key: str = Depends(X_API_KEY)):
    """takes the X-API-Key header and validate it with the X-API-Key in the database/environment"""
    if x_api_key != USER_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key. Check that you are passing a 'X-API-Key' on your header.",
        )


app = FastAPI()


@app.get("/")
async def root(current_user=Depends(api_key_auth)):
    return {"Health": "Server running"}


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(None),
    text: str = Form(None),
    current_user=Depends(api_key_auth),
):
    if file:
        content = (await file.read()).decode()
    elif text:
        content = text
    else:
        raise HTTPException(status_code=400, detail="No input provided")

    return {"response": f"Not implemented yet {text}"}


@app.get("/retrieve")
async def retrieve_documents(
    query: str,
    match_count: int = 10,
    current_user=Depends(api_key_auth),
):
    # Get the embedding for the query
    embedding = await get_embedding(query)

    return await retrieve_relevant_documentation(embedding, match_count)


@app.get("/retrieveids")
async def retrieve_ids(
    query: str,
    match_count: int = 10,
    current_user=Depends(api_key_auth),
):
    # Get the embedding for the query
    embedding = await get_embedding(query)

    return await retrieve_relevant_documentation_ids(embedding, match_count)


@app.get("/generateWithoutRef")
async def generate_response(
    query: str,
    model: str | None = None,
    current_user=Depends(api_key_auth),
):
    if query is None:
        raise HTTPException(
            status_code=400,
            detail="No query provided",
        )
    if model is None:
        model = "google:gemini-2.0-pro-exp-02-05"
    if ":" not in model:
        raise HTTPException(
            status_code=400,
            detail="Invalid model format. Please use format 'provider_name:model_name'",
        )

    # Get the embedding for the query
    embedding = await get_embedding(query)

    # Check the cache for the query
    cached_result = await query_cache(embedding)
    if cached_result:
        return StreamingResponse(
            generate_redis_response(cached_result),
            headers={"X-Response-Source": "cache"},
            media_type="text/plain",
        )
    else:
        response = await retrieve_relevant_documentation(embedding)
        context, array_of_contents = concatenate_contents(response)

        # Call the generate function (note: not awaited since it returns a generator)
        model, generate_func = get_approproate_generate_func(model=model)

        # Return a StreamingResponse so that the client receives the stream in real time.
        return StreamingResponse(
            content=generate_func(context=context, query=query, model=model),
            headers={"X-Response-Source": "llm"},
            media_type="text/plain",
        )


@app.get("/generate")
async def generate_response(
    query: str,
    model: str | None = None,
    current_user=Depends(api_key_auth),
):
    if query is None:
        raise HTTPException(
            status_code=400,
            detail="No query provided",
        )
    if model is None:
        model = "google:gemini-2.0-pro-exp-02-05"
    if ":" not in model:
        raise HTTPException(
            status_code=400,
            detail="Invalid model format. Please use format 'provider_name:model_name'",
        )

    # Get the embedding for the query
    embedding = await get_embedding(query)

    response = await retrieve_relevant_documentation(embedding)
    context, array_of_contents = concatenate_contents(response)

    # Check the cache for the query
    cached_result = await query_cache(embedding)
    if cached_result:
        # return StreamingResponse(
        #     content=generate_redis_response(cached_result),
        #     headers={"X-Response-Source": "cache"},
        #     media_type="text/plain",
        # )
        content, doc_id = generate_redis_response(cached_result)
        return {
            "content": content,
            "context": array_of_contents,
            "response_source": f"cache-{doc_id}",
        }
    else:
        # Call the generate function (note: not awaited since it returns a generator)
        model, generate_func = get_approproate_generate_func(model=model)

        # Return a StreamingResponse so that the client receives the stream in real time.
        # return StreamingResponse(
        #     content=generate_func(context=context, query=query, model=model),
        #     headers={"X-Response-Source": "llm"},
        #     media_type="text/plain",
        # )
        response = await generate_func(context=context, query=query, model=model)
        return {
            "content": response,
            "context": array_of_contents,
            "response_source": "llm",
        }


@app.post("/addtocache")
async def addtocache(
    query: str,
    content: str,
    current_user=Depends(api_key_auth),
):
    response = await set_cache(query=query, content=content)
    if response == "success":
        return {"response": "Cache set successfully"}
    else:
        raise HTTPException(status_code=500, detail=response)


@app.post("/updatecache")
async def addtocache(
    query: str,
    content: str,
    doc_id: str,
    current_user=Depends(api_key_auth),
):
    response = await set_cache(query=query, content=content, doc_id=doc_id)
    if response == "success":
        return {"response": "Cache updated successfully"}
    else:
        raise HTTPException(status_code=500, detail=response)


@app.delete("/deletecache")
async def deletecache(
    doc_id: str,
    current_user=Depends(api_key_auth),
):
    response = await delete_cache(doc_id=doc_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
