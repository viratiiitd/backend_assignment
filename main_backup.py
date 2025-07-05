from fastapi import (
    FastAPI, HTTPException, Depends, Query, Path,
    BackgroundTasks, WebSocket, WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import openai
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager
from websocket_handler import ConnectionManager, WebSocketHandler
import uvicorn
import traceback

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "chat_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# MongoDB client
client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to MongoDB when the application starts
    global client
    logger.info("Connecting to MongoDB...")
    client = AsyncIOMotorClient(MONGODB_URI)
    logger.info("Connected to MongoDB successfully")
    yield
    # Close the MongoDB connection when the application shuts down
    if client:
        logger.info("Closing MongoDB connection...")
        client.close()
        logger.info("MongoDB connection closed")

app = FastAPI(
    title="Chat Summarization and Insights API",
    description="A FastAPI-based REST API for processing chat data with LLM-powered summarization",
    version="1.0.0",
    lifespan=lifespan,
)

# Initialize WebSocket manager
manager = ConnectionManager()
websocket_handler = WebSocketHandler(manager)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, update for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Database dependency
async def get_db():
    return client[DATABASE_NAME]

# Pydantic models
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if not ObjectId.is_valid(value):
            raise ValueError("Invalid ObjectId")
        return ObjectId(value)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class UserModel(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    username: str
    email: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class MessageModel(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    conversation_id: PyObjectId
    user_id: PyObjectId
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    read: bool = False

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class ConversationModel(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    title: str
    participants: List[PyObjectId]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    summary: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class MessageCreate(BaseModel):
    conversation_id: str
    user_id: str
    content: str

    @validator('conversation_id', 'user_id')
    def validate_object_id(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return v

class ConversationCreate(BaseModel):
    title: str
    participants: List[str]

    @validator('participants')
    def validate_participants(cls, v):
        for participant_id in v:
            if not ObjectId.is_valid(participant_id):
                raise ValueError(f"Invalid participant ObjectId: {participant_id}")
        return v

class SummarizationRequest(BaseModel):
    conversation_id: str

    @validator('conversation_id')
    def validate_object_id(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return v

class ConversationResponse(BaseModel):
    id: str
    title: str
    participants: List[str]
    created_at: datetime
    updated_at: datetime
    summary: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None

# Helper functions
def convert_objectid_to_str(obj):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if isinstance(obj[key], ObjectId):
                obj[key] = str(obj[key])
            elif isinstance(obj[key], (dict, list)):
                obj[key] = convert_objectid_to_str(obj[key])
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = convert_objectid_to_str(item)
    return obj

async def get_messages_for_conversation(db, conversation_id: ObjectId):
    cursor = db.messages.find({"conversation_id": conversation_id}).sort("timestamp", 1)
    messages = await cursor.to_list(length=None)
    return convert_objectid_to_str(messages)

async def generate_summary_with_llm(messages: List[Dict[str, Any]]):
    if not messages:
        return "No messages to summarize."
    
    try:
        # Check if OpenAI API key is available
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            # Mock response if no API key
            logger.info("Using mock LLM response (no valid API key)")
            message_count = len(messages)
            unique_users = len(set([
                msg.get('user_id', 'unknown') for msg in messages
            ]))
            return (
                f"This is a mock summary of {message_count} messages. "
                f"In a production environment, this would be generated by "
                f"OpenAI's GPT-3.5. The conversation has contributions from "
                f"{unique_users} participants."
            )
        
        # Format messages for the LLM
        message_texts = [
            f"{msg.get('user_id', 'Unknown')}: {msg.get('content', '')}"
            for msg in messages if msg.get('content')
        ]
        conversation_text = "\n".join(message_texts)
        
        prompt = f"""
        Please provide a concise summary of the following conversation:
        
        {conversation_text}
        
        Summary:
        """
        
        # Use the new OpenAI client
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": ("You are a helpful assistant that summarizes "
                                "conversations.")
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Failed to generate summary. Please try again later."


async def generate_insights_with_llm(messages: List[Dict[str, Any]]):
    if not messages:
        return {"sentiment": "neutral", "keywords": [], "topics": []}
    
    try:
        # Check if OpenAI API key is available
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            # Mock response if no API key
            logger.info("Using mock LLM insights (no valid API key)")
            return {
                "sentiment": "positive",
                "topics": ["mock topic 1", "mock topic 2",
                           "development discussion"],
                "keywords": ["api", "development", "chat",
                             "summarization", "mock"]
            }
        
        # Format messages for the LLM
        message_texts = [
            f"{msg.get('user_id', 'Unknown')}: {msg.get('content', '')}"
            for msg in messages if msg.get('content')
        ]
        conversation_text = "\n".join(message_texts)
        
        prompt = f"""
        Analyze the following conversation and provide insights:
        1. Overall sentiment (positive, negative, or neutral)
        2. Key topics discussed (as a list)
        3. Important keywords (as a list)
        
        Conversation:
        {conversation_text}
        
        Format your response as JSON:
        {{
            "sentiment": "sentiment_value",
            "topics": ["topic1", "topic2", ...],
            "keywords": ["keyword1", "keyword2", ...]
        }}
        """
        
        # Use the new OpenAI client
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": ("You are a helpful assistant that analyzes "
                                "conversations and returns structured "
                                "insights.")
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5,
        )
        
        insights_text = response.choices[0].message.content.strip()
        
        # Extract JSON from the response
        import json
        import re
        
        json_match = re.search(r'{.*}', insights_text, re.DOTALL)
        if json_match:
            insights_json = json.loads(json_match.group(0))
            return insights_json
        else:
            logger.error(f"Failed to parse insights JSON: {insights_text}")
            return {"sentiment": "neutral", "keywords": [], "topics": []}
            
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return {"sentiment": "neutral", "keywords": [], "topics": []}


# Async task to update summary
async def update_conversation_summary(conversation_id: str, db):
    try:
        conversation_obj_id = ObjectId(conversation_id)
        messages = await get_messages_for_conversation(db, conversation_obj_id)
        summary = await generate_summary_with_llm(messages)
        
        await db.conversations.update_one(
            {"_id": conversation_obj_id},
            {"$set": {"summary": summary}}
        )
        logger.info(f"Updated summary for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error updating summary for conversation "
                     f"{conversation_id}: {e}")


# API endpoints
@app.post("/chats", status_code=201, response_model=MessageModel)
async def create_chat_message(
    message: MessageCreate,
    background_tasks: BackgroundTasks,
    db=Depends(get_db)
):
    try:
        # Check if conversation exists
        conversation = await db.conversations.find_one(
            {"_id": ObjectId(message.conversation_id)}
        )
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        # Check if user exists
        user = await db.users.find_one({"_id": ObjectId(message.user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Create message
        message_data = {
            "conversation_id": ObjectId(message.conversation_id),
            "user_id": ObjectId(message.user_id),
            "content": message.content,
            "timestamp": datetime.utcnow(),
            "read": False
        }
        
        # Insert message
        result = await db.messages.insert_one(message_data)
        
        # Update conversation's updated_at field
        await db.conversations.update_one(
            {"_id": ObjectId(message.conversation_id)},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
        
        # Schedule summary update in the background
        background_tasks.add_task(
            update_conversation_summary,
            message.conversation_id,
            db
        )
        
        # Get created message
        created_message = await db.messages.find_one(
            {"_id": result.inserted_id}
        )
        created_message["_id"] = str(created_message["_id"])
        created_message["conversation_id"] = str(
            created_message["conversation_id"]
        )
        created_message["user_id"] = str(created_message["user_id"])
        
        return created_message
    except Exception as e:
        logger.error(f"Error creating message: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/users", response_model=List[UserModel])
async def get_users(db=Depends(get_db)):
    try:
        # Get all users
        cursor = db.users.find()
        users = await cursor.to_list(length=100)  # Limit to 100 users
        
        # Convert ObjectId to string for JSON response
        for user in users:
            user["_id"] = str(user["_id"])
        
        return users
    except Exception as e:
        logger.error(f"Error retrieving users: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/chats/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str = Path(
        ...,
        description="The ID of the conversation to retrieve"
    ),
    db=Depends(get_db)
):
    try:
        if not ObjectId.is_valid(conversation_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid conversation ID format"
            )
        
        conversation_obj_id = ObjectId(conversation_id)
        conversation = await db.conversations.find_one(
            {"_id": conversation_obj_id}
        )
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        # Log the raw conversation data
        print(f"Raw conversation data: {conversation}")
        
        # Get messages for this conversation
        try:
            messages = await get_messages_for_conversation(
                db, conversation_obj_id
            )
            print(f"Retrieved {len(messages)} messages")
        except Exception as msg_error:
            print(f"Error getting messages: {msg_error}")
            messages = []
        
        # Convert ObjectId to string for JSON response
        try:
            # Make a copy to avoid modifying the original
            conversation_copy = dict(conversation)
            for key, value in conversation_copy.items():
                if isinstance(value, ObjectId):
                    conversation_copy[key] = str(value)
                elif key == "participants" and isinstance(value, list):
                    conversation_copy[key] = [
                        str(p) if isinstance(p, ObjectId) else p
                        for p in value
                    ]
            
            # Use the explicit copy with string IDs
            conversation = conversation_copy
            conversation["id"] = conversation.pop("_id")
            conversation["messages"] = messages
            
            return conversation
        except Exception as convert_error:
            print(f"Error converting ObjectIds: {convert_error}")
            raise
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        error_detail = traceback.format_exc()
        print(f"Detailed error: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/chats/summarize", status_code=200)
async def summarize_chat(request: SummarizationRequest, db=Depends(get_db)):
    try:
        conversation_obj_id = ObjectId(request.conversation_id)
        conversation = await db.conversations.find_one(
            {"_id": conversation_obj_id}
        )
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        # Get messages
        messages = await get_messages_for_conversation(db, conversation_obj_id)
        
        # Generate summary
        summary = await generate_summary_with_llm(messages)
        
        # Update conversation with new summary
        await db.conversations.update_one(
            {"_id": conversation_obj_id},
            {"$set": {"summary": summary}}
        )
        
        # Generate insights as a bonus feature
        insights = await generate_insights_with_llm(messages)
        
        return {
            "conversation_id": request.conversation_id,
            "summary": summary,
            "insights": insights
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error summarizing conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/users/{user_id}/chats")
async def get_user_chats(
    user_id: str = Path(...),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    db=Depends(get_db)
):
    try:
        # Make sure you're converting string ID to ObjectId
        if not ObjectId.is_valid(user_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid user ID format"
            )
        
        user_obj_id = ObjectId(user_id)
        
        # Check if user exists (add this if not already there)
        user = await db.users.find_one({"_id": user_obj_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get conversations with proper error handling
        cursor = db.conversations.find(
            {"participants": user_obj_id}
        ).sort("updated_at", -1)
        
        # Print debug info
        print(f"Looking for conversations with participant: {user_id}")
        
        # Return empty list if no conversations yet
        conversations = await cursor.to_list(length=None) or []
        
        # Convert ObjectId to string
        for conversation in conversations:
            conversation["_id"] = str(conversation["_id"])
            # Convert participant IDs too
            conversation["participants"] = [
                str(p) for p in conversation["participants"]
            ]
        
        return {
            "conversations": conversations,
            "total": len(conversations),
            "page": page,
            "limit": limit,
            "total_pages": max(1, (len(conversations) + limit - 1) // limit)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        # Log the actual error
        print(f"Error in get_user_chats: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.delete("/chats/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str = Path(..., description="The ID of the conversation to delete"),
    db=Depends(get_db)
):
    try:
        if not ObjectId.is_valid(conversation_id):
            raise HTTPException(status_code=400, detail="Invalid conversation ID format")
        
        conversation_obj_id = ObjectId(conversation_id)
        
        # Check if conversation exists
        conversation = await db.conversations.find_one({"_id": conversation_obj_id})
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete all messages in the conversation
        await db.messages.delete_many({"conversation_id": conversation_obj_id})
        
        # Delete the conversation
        await db.conversations.delete_one({"_id": conversation_obj_id})
        
        return None
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Additional endpoint for creating conversations
@app.post("/conversations", status_code=201, response_model=ConversationModel)
async def create_conversation(conversation: ConversationCreate, db=Depends(get_db)):
    try:
        # Convert participant IDs to ObjectId
        participant_obj_ids = [ObjectId(pid) for pid in conversation.participants]
        
        # Check if all users exist
        for participant_id in participant_obj_ids:
            user = await db.users.find_one({"_id": participant_id})
            if not user:
                raise HTTPException(status_code=404, detail=f"User with ID {participant_id} not found")
        
        # Create conversation
        conversation_data = {
            "title": conversation.title,
            "participants": participant_obj_ids,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "summary": None
        }
        
        # Insert conversation
        result = await db.conversations.insert_one(conversation_data)
        
        # Get created conversation
        created_conversation = await db.conversations.find_one({"_id": result.inserted_id})
        created_conversation = convert_objectid_to_str(created_conversation)
        
        return created_conversation
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Additional endpoint for creating users
@app.post("/users", status_code=201, response_model=UserModel)
async def create_user(user: UserModel, db=Depends(get_db)):
    try:
        # Check if username already exists
        existing_user = await db.users.find_one({"username": user.username})
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Check if email already exists
        existing_email = await db.users.find_one({"email": user.email})
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create user
        user_data = {
            "username": user.username,
            "email": user.email,
            "created_at": datetime.utcnow()
        }
        
        # Insert user
        result = await db.users.insert_one(user_data)
        
        # Get created user
        created_user = await db.users.find_one({"_id": result.inserted_id})
        created_user["_id"] = str(created_user["_id"])
        
        return created_user
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Additional endpoint for conversation insights (Bonus feature)
@app.get("/chats/{conversation_id}/insights")
async def get_conversation_insights(
    conversation_id: str = Path(..., description="The ID of the conversation to analyze"),
    db=Depends(get_db)
):
    try:
        if not ObjectId.is_valid(conversation_id):
            raise HTTPException(status_code=400, detail="Invalid conversation ID format")
        
        conversation_obj_id = ObjectId(conversation_id)
        conversation = await db.conversations.find_one({"_id": conversation_obj_id})
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages
        messages = await get_messages_for_conversation(db, conversation_obj_id)
        
        # Generate insights
        insights = await generate_insights_with_llm(messages)
        
        return {
            "conversation_id": conversation_id,
            "insights": insights
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating conversation insights: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# WebSocket endpoint for real-time chat
@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str, db=Depends(get_db)):
    await websocket_handler.handle_websocket(websocket, conversation_id, db)

# Broadcast message to all connected clients
async def broadcast_message(conversation_id: str, message: Dict[str, Any]):
    await manager.broadcast(message, conversation_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)