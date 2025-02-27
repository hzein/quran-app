from functools import partial
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel
from typing import Annotated, List, Optional
from datetime import datetime, timedelta
import uuid
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

# Database setup
from sqlalchemy import create_engine, Column, String, Text, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# Load retrieve relevant context
from app.retrieve_relevant_context import retrieve_relevant_documentation

# Generate response
from app.google_api import generate

load_dotenv()

SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database models
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    disabled = Column(String, default=False)


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    content = Column(Text)
    owner_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("ChatMessage", back_populates="session")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    message = Column(Text)
    sender = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")


# Create tables
Base.metadata.create_all(bind=engine)

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


# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str


class DocumentCreate(BaseModel):
    content: str


# New Pydantic model for Document output
class DocumentOut(BaseModel):
    id: str
    content: str
    owner_id: str
    created_at: datetime

    class Config:
        orm_mode = True


class ChatMessageCreate(BaseModel):
    message: str


class ChatSessionOut(BaseModel):
    id: str
    created_at: datetime
    messages: List[dict]


# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# HTTPBearer to check the Admin API key
# async def get_current_admin_user(
#     credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
#     db: Session = Depends(get_db),
# ):
#     if credentials.scheme.lower() != "bearer":
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Invalid authentication scheme.",
#         )
#     token = credentials.credentials
#     if token != ADMIN_API_KEY:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid API Key.",
#             headers={"WWW-Authenticate": "Bearer"},
#         )


# # HTTPBearer to check the User API key
# async def get_current_user(
#     credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
#     db: Session = Depends(get_db),
# ):
#     if credentials.scheme.lower() != "bearer":
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Invalid authentication scheme.",
#         )
#     token = credentials.credentials
#     if token != USER_API_KEY:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid API Key.",
#             headers={"WWW-Authenticate": "Bearer"},
#         )


# Endpoints


def concatenate_contents(api_response):
    """
    Concatenates the 'content' field from each item in the API response.

    Args:
      api_response: A list of dictionaries, where each dictionary represents
                    an item from the API response and has a 'content' key.

    Returns:
      A string containing the concatenated contents from all items in the
      API response.
    """
    concatenated_content = ""
    for item in api_response:
        if "content" in item:
            concatenated_content += f"{item['content']} \n"
    return concatenated_content


@app.get("/")
async def root(current_user: User = Depends(api_key_auth)):
    return {"Health": "Server running"}


@app.post("/ingest/")
async def ingest_document(
    file: UploadFile = File(None),
    text: str = Form(None),
    current_user: User = Depends(api_key_auth),
):
    if file:
        content = (await file.read()).decode()
    elif text:
        content = text
    else:
        raise HTTPException(status_code=400, detail="No input provided")

    return {"response": f"Not implemented yet {text}"}


@app.get("/retrieve/")
async def retrieve_documents(
    query: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(api_key_auth),
):
    return await retrieve_relevant_documentation(user_query=query)


@app.post("/generate")
async def generate_response(
    query: str,
    current_user: User = Depends(api_key_auth),
):
    response = await retrieve_relevant_documentation(user_query=query)
    context = concatenate_contents(response)

    # Call the generate function (note: not awaited since it returns a generator)
    stream = generate(query=query, context=context)

    # Return a StreamingResponse so that the client receives the stream in real time.
    return StreamingResponse(stream, media_type="text/plain")


# @app.post("/chat/start", response_model=ChatSessionOut)
# async def start_chat_session(
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user),
# ):
#     session_id = str(uuid.uuid4())
#     db_session = ChatSession(id=session_id, user_id=current_user.id)
#     db.add(db_session)
#     db.commit()
#     return db_session


# @app.post("/chat/", response_model=ChatSessionOut)
# async def chat(
#     session_id: str,
#     message: ChatMessageCreate,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user),
# ):
#     db_session = (
#         db.query(ChatSession)
#         .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
#         .first()
#     )

#     if not db_session:
#         raise HTTPException(status_code=404, detail="Session not found")

#     message_id = str(uuid.uuid4())
#     db_message = ChatMessage(
#         id=message_id, session_id=session_id, message=message.message, sender="user"
#     )
#     db.add(db_message)

#     # Add mock AI response
#     ai_message_id = str(uuid.uuid4())
#     db_ai_message = ChatMessage(
#         id=ai_message_id,
#         session_id=session_id,
#         message=f"Received your message: {message.message}",
#         sender="assistant",
#     )
#     db.add(db_ai_message)

#     db.commit()
#     db.refresh(db_session)
#     return db_session


# @app.get("/chat/{session_id}", response_model=ChatSessionOut)
# async def get_chat_session(
#     session_id: str,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user),
# ):
#     session = (
#         db.query(ChatSession)
#         .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
#         .first()
#     )

#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")

#     return session


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
