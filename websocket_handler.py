from fastapi import WebSocket, WebSocketDisconnect, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, List, Any
import json
import logging
from datetime import datetime
from bson import ObjectId

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        logger.info("WebSocket connection manager initialized")

    async def connect(self, websocket: WebSocket, conversation_id: str):
        await websocket.accept()
        if conversation_id not in self.active_connections:
            self.active_connections[conversation_id] = []
        self.active_connections[conversation_id].append(websocket)
        logger.info(f"Client connected to conversation {conversation_id}. Total connections: {len(self.active_connections[conversation_id])}")

    def disconnect(self, websocket: WebSocket, conversation_id: str):
        if conversation_id in self.active_connections:
            if websocket in self.active_connections[conversation_id]:
                self.active_connections[conversation_id].remove(websocket)
                logger.info(f"Client disconnected from conversation {conversation_id}. Remaining connections: {len(self.active_connections[conversation_id])}")
            if not self.active_connections[conversation_id]:
                del self.active_connections[conversation_id]
                logger.info(f"No more connections for conversation {conversation_id}. Removed from active conversations.")

    async def broadcast(self, message: Dict[str, Any], conversation_id: str):
        if conversation_id in self.active_connections:
            disconnected_clients = []
            for connection in self.active_connections[conversation_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting message: {e}")
                    disconnected_clients.append(connection)
            
            # Clean up disconnected clients
            for client in disconnected_clients:
                self.disconnect(client, conversation_id)


class WebSocketHandler:
    """Handler for WebSocket connections and real-time chat functionality"""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        logger.info("WebSocket handler initialized")
    
    async def handle_websocket(self, websocket: WebSocket, conversation_id: str, db: AsyncIOMotorDatabase):
        await self.manager.connect(websocket, conversation_id)
        
        try:
            # Validate conversation_id
            if not ObjectId.is_valid(conversation_id):
                await websocket.send_json({"error": "Invalid conversation ID format"})
                await websocket.close()
                return
            
            # Check if conversation exists
            conversation_obj_id = ObjectId(conversation_id)
            conversation = await db.conversations.find_one({"_id": conversation_obj_id})
            
            if not conversation:
                await websocket.send_json({"error": "Conversation not found"})
                await websocket.close()
                return
            
            # Send initial conversation data
            cursor = db.messages.find({"conversation_id": conversation_obj_id}).sort("timestamp", 1)
            messages = await cursor.to_list(length=None)
            
            # Convert ObjectId to string for JSON response
            for message in messages:
                message["_id"] = str(message["_id"])
                message["conversation_id"] = str(message["conversation_id"])
                message["user_id"] = str(message["user_id"])
            
            await websocket.send_json({
                "type": "initial_data",
                "conversation_id": conversation_id,
                "messages": messages,
                "summary": conversation.get("summary")
            })
            
            # Main WebSocket loop
            while True:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle different message types
                message_type = message_data.get("type", "message")
                
                if message_type == "message":
                    # Store new message
                    if "user_id" not in message_data or "content" not in message_data:
                        await websocket.send_json({"error": "Invalid message format. Required fields: user_id, content"})
                        continue
                    
                    # Check if user exists
                    user_obj_id = ObjectId(message_data["user_id"])
                    user = await db.users.find_one({"_id": user_obj_id})
                    if not user:
                        await websocket.send_json({"error": "User not found"})
                        continue
                    
                    # Create message
                    new_message = {
                        "conversation_id": conversation_obj_id,
                        "user_id": user_obj_id,
                        "content": message_data["content"],
                        "timestamp": datetime.utcnow(),
                        "read": False
                    }
                    
                    # Insert message
                    result = await db.messages.insert_one(new_message)
                    
                    # Update conversation's updated_at field
                    await db.conversations.update_one(
                        {"_id": conversation_obj_id},
                        {"$set": {"updated_at": datetime.utcnow()}}
                    )
                    
                    # Get created message
                    created_message = await db.messages.find_one({"_id": result.inserted_id})
                    created_message["_id"] = str(created_message["_id"])
                    created_message["conversation_id"] = str(created_message["conversation_id"])
                    created_message["user_id"] = str(created_message["user_id"])
                    
                    # Broadcast message to all clients
                    await self.manager.broadcast({
                        "type": "new_message",
                        "message": created_message
                    }, conversation_id)
                    
                    # Update summary in background if there are enough messages
                    message_count = await db.messages.count_documents({"conversation_id": conversation_obj_id})
                    if message_count % 5 == 0:  # Update summary every 5 messages
                        from main import generate_summary_with_llm, get_messages_for_conversation
                        
                        # Get all messages
                        messages = await get_messages_for_conversation(db, conversation_obj_id)
                        
                        # Generate summary
                        summary = await generate_summary_with_llm(messages)
                        
                        # Update conversation with new summary
                        await db.conversations.update_one(
                            {"_id": conversation_obj_id},
                            {"$set": {"summary": summary}}
                        )
                        
                        # Broadcast updated summary
                        await self.manager.broadcast({
                            "type": "summary_update",
                            "summary": summary
                        }, conversation_id)
                
                elif message_type == "request_summary":
                    # Generate and send summary on demand
                    from main import generate_summary_with_llm, get_messages_for_conversation
                    
                    # Get all messages
                    messages = await get_messages_for_conversation(db, conversation_obj_id)
                    
                    # Generate summary
                    summary = await generate_summary_with_llm(messages)
                    
                    # Update conversation with new summary
                    await db.conversations.update_one(
                        {"_id": conversation_obj_id},
                        {"$set": {"summary": summary}}
                    )
                    
                    # Send summary to requesting client
                    await websocket.send_json({
                        "type": "summary_update",
                        "summary": summary
                    })
                    
                    # Also broadcast to other clients
                    await self.manager.broadcast({
                        "type": "summary_update",
                        "summary": summary
                    }, conversation_id)
                
                elif message_type == "request_insights":
                    # Generate and send insights on demand
                    from main import generate_insights_with_llm, get_messages_for_conversation
                    
                    # Get all messages
                    messages = await get_messages_for_conversation(db, conversation_obj_id)
                    
                    # Generate insights
                    insights = await generate_insights_with_llm(messages)
                    
                    # Send insights to requesting client
                    await websocket.send_json({
                        "type": "insights_update",
                        "insights": insights
                    })
        
        except WebSocketDisconnect:
            self.manager.disconnect(websocket, conversation_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            try:
                await websocket.send_json({"error": f"An error occurred: {str(e)}"})
            except:
                pass
            self.manager.disconnect(websocket, conversation_id)