import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
import os
import asyncio
import websockets
import threading
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_URL = os.getenv("API_URL", "http://api:8000")

# Set page configuration
st.set_page_config(
    page_title="Chat Summarization App",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    .chat-message.other {
        background-color: #f6f6f6;
        border-left: 5px solid #d9d9d9;
    }
    .chat-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .chat-content {
        white-space: pre-wrap;
    }
    .summary-box {
        background-color: #fffbe6;
        border-left: 5px solid #faad14;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .insights-box {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def fetch_users():
    try:
        response = requests.get(f"{API_URL}/users")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch users: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return []

def fetch_user_conversations(user_id, page=1, limit=10):
    try:
        response = requests.get(f"{API_URL}/users/{user_id}/chats?page={page}&limit={limit}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch conversations: {response.text}")
            return {"total": 0, "conversations": []}
    except Exception as e:
        st.error(f"Error fetching conversations: {e}")
        return {"total": 0, "conversations": []}

def fetch_conversation(conversation_id):
    try:
        response = requests.get(f"{API_URL}/chats/{conversation_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch conversation: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching conversation: {e}")
        return None

def send_message(conversation_id, user_id, content):
    try:
        data = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "content": content
        }
        response = requests.post(f"{API_URL}/chats", json=data)
        if response.status_code == 201:
            return response.json()
        else:
            st.error(f"Failed to send message: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending message: {e}")
        return None

def create_user(username, email):
    try:
        data = {
            "username": username,
            "email": email
        }
        response = requests.post(f"{API_URL}/users", json=data)
        if response.status_code == 201:
            return response.json()
        else:
            st.error(f"Failed to create user: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return None

def create_conversation(title, participants):
    try:
        data = {
            "title": title,
            "participants": participants
        }
        response = requests.post(f"{API_URL}/conversations", json=data)
        if response.status_code == 201:
            return response.json()
        else:
            st.error(f"Failed to create conversation: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error creating conversation: {e}")
        return None

def summarize_conversation(conversation_id):
    try:
        data = {
            "conversation_id": conversation_id
        }
        response = requests.post(f"{API_URL}/chats/summarize", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to summarize conversation: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error summarizing conversation: {e}")
        return None

def get_conversation_insights(conversation_id):
    try:
        response = requests.get(f"{API_URL}/chats/{conversation_id}/insights")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get insights: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting insights: {e}")
        return None

# Initialize session state
if "active_user" not in st.session_state:
    st.session_state.active_user = None
if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "insights" not in st.session_state:
    st.session_state.insights = None
if "websocket_connected" not in st.session_state:
    st.session_state.websocket_connected = False
if "websocket_thread" not in st.session_state:
    st.session_state.websocket_thread = None

# WebSocket Handler
def connect_to_websocket(conversation_id):
    """Connect to WebSocket and handle real-time messages"""
    st.session_state.websocket_connected = True
    
    # Convert API URL to WebSocket URL
    ws_base_url = API_URL.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_base_url}/ws/{conversation_id}"
    
    async def websocket_listener():
        try:
            async with websockets.connect(ws_url) as websocket:
                # Handle message receiving in a loop
                while st.session_state.websocket_connected:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        # Handle different message types
                        if data.get("type") == "initial_data":
                            st.session_state.messages = data.get("messages", [])
                            st.session_state.summary = data.get("summary")
                        
                        elif data.get("type") == "new_message":
                            new_message = data.get("message")
                            if new_message:
                                # Add message if it's not already in the list
                                message_ids = [msg.get("_id") for msg in st.session_state.messages]
                                if new_message.get("_id") not in message_ids:
                                    st.session_state.messages.append(new_message)
                        
                        elif data.get("type") == "summary_update":
                            st.session_state.summary = data.get("summary")
                        
                        elif data.get("type") == "insights_update":
                            st.session_state.insights = data.get("insights")
                        
                        # Force Streamlit to update the UI
                        st.experimental_rerun()
                    
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        st.error(f"WebSocket error: {e}")
                        break
        
        except Exception as e:
            st.error(f"Failed to connect to WebSocket: {e}")
        
        finally:
            st.session_state.websocket_connected = False
    
    # Start the async WebSocket listener in a background thread
    def run_async_websocket():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_listener())
    
    # Store the thread in the session state so we can stop it later
    if st.session_state.websocket_thread is not None:
        st.session_state.websocket_connected = False
        st.session_state.websocket_thread.join(timeout=1.0)
    
    st.session_state.websocket_thread = threading.Thread(target=run_async_websocket)
    st.session_state.websocket_thread.daemon = True
    st.session_state.websocket_thread.start()

# Sidebar for user selection and conversation management
with st.sidebar:
    st.title("Chat Summarization")
    
    # User Management
    st.header("Users")
    
    # Create new user
    with st.expander("Create New User"):
        username = st.text_input("Username", key="new_username")
        email = st.text_input("Email", key="new_email")
        if st.button("Create User"):
            if username and email:
                user = create_user(username, email)
                if user:
                    st.success(f"User '{username}' created successfully!")
                    st.session_state.active_user = user
            else:
                st.warning("Username and email are required!")
    
    # Select existing user
    st.subheader("Select User")
    users = fetch_users()
    user_options = [f"{user['username']} ({user['email']})" for user in users]
    user_map = {f"{user['username']} ({user['email']})": user for user in users}
    
    selected_user = st.selectbox("Select a user", ["None"] + user_options)
    
    if selected_user != "None":
        st.session_state.active_user = user_map[selected_user]
    
    # Conversation Management
    if st.session_state.active_user:
        st.header("Conversations")
        
        # Create new conversation
        with st.expander("Create New Conversation"):
            title = st.text_input("Title", key="new_conversation_title")
            
            # Multi-select for participants
            all_users = fetch_users()
            participant_options = [(user["_id"], user["username"]) for user in all_users]
            
            selected_participants = st.multiselect(
                "Select participants",
                options=[option[0] for option in participant_options],
                format_func=lambda x: next((option[1] for option in participant_options if option[0] == x), x),
                default=[st.session_state.active_user["_id"]]
            )
            
            if st.button("Create Conversation"):
                if title and selected_participants:
                    conversation = create_conversation(title, selected_participants)
                    if conversation:
                        st.success(f"Conversation '{title}' created successfully!")
                        st.session_state.active_conversation = conversation
                else:
                    st.warning("Title and at least one participant are required!")
        
        # View existing conversations
        st.subheader("Select Conversation")
        
        # Pagination for conversations
        page = st.session_state.get("conversation_page", 1)
        limit = 10
        
        # Fetch conversations
        conversations_data = fetch_user_conversations(st.session_state.active_user["_id"], page, limit)
        conversations = conversations_data.get("conversations", [])
        total_pages = conversations_data.get("total_pages", 1)
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if page > 1:
                if st.button("Previous"):
                    st.session_state.conversation_page = page - 1
                    st.experimental_rerun()
        with col2:
            st.write(f"Page {page} of {total_pages}")
        with col3:
            if page < total_pages:
                if st.button("Next"):
                    st.session_state.conversation_page = page + 1
                    st.experimental_rerun()
        
        # Display conversations
        conversation_options = [f"{conv['title']} (Created: {conv['created_at'][:10]})" for conv in conversations]
        conversation_map = {f"{conv['title']} (Created: {conv['created_at'][:10]})": conv for conv in conversations}
        
        selected_conversation = st.selectbox("Select a conversation", ["None"] + conversation_options)
        
        if selected_conversation != "None":
            st.session_state.active_conversation = conversation_map[selected_conversation]
            
            # Connect to WebSocket for real-time updates
            connect_to_websocket(st.session_state.active_conversation["id"])
            
            # Also load conversation data through REST API as a fallback
            conversation_data = fetch_conversation(st.session_state.active_conversation["id"])
            if conversation_data:
                st.session_state.messages = conversation_data.get("messages", [])
                st.session_state.summary = conversation_data.get("summary")
            
            # Conversation actions
            st.subheader("Actions")
            
            if st.button("Generate Summary"):
                summary_result = summarize_conversation(st.session_state.active_conversation["id"])
                if summary_result:
                    st.session_state.summary = summary_result.get("summary")
                    st.session_state.insights = summary_result.get("insights")
                    st.success("Summary generated successfully!")
                    st.experimental_rerun()
            
            if st.button("Generate Insights"):
                insights_result = get_conversation_insights(st.session_state.active_conversation["id"])
                if insights_result:
                    st.session_state.insights = insights_result.get("insights")
                    st.success("Insights generated successfully!")
                    st.experimental_rerun()

# Main content area
if not st.session_state.active_user:
    st.header("Welcome to Chat Summarization App")
    st.write("Please select or create a user from the sidebar to get started.")
elif not st.session_state.active_conversation:
    st.header(f"Welcome, {st.session_state.active_user['username']}!")
    st.write("Please select or create a conversation from the sidebar to get started.")
else:
    # Display conversation title
    st.header(f"Conversation: {st.session_state.active_conversation['title']}")
    
    # Display summary if available
    if st.session_state.summary:
        with st.expander("Conversation Summary", expanded=True):
            st.markdown(f"""
            <div class="summary-box">
                <h3>Summary</h3>
                <p>{st.session_state.summary}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display insights if available
    if st.session_state.insights:
        with st.expander("Conversation Insights", expanded=True):
            insights = st.session_state.insights
            
            # Sentiment visualization
            sentiment = insights.get("sentiment", "neutral")
            sentiment_color = {
                "positive": "green",
                "neutral": "gray",
                "negative": "red"
            }.get(sentiment.lower(), "gray")
            
            st.markdown(f"""
            <div class="insights-box">
                <h3>Insights</h3>
                <p><strong>Sentiment:</strong> <span style="color: {sentiment_color};">{sentiment}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display topics
            if "topics" in insights and insights["topics"]:
                st.subheader("Topics")
                topics = insights["topics"]
                # Create a bar chart of topics
                topic_df = pd.DataFrame({
                    "Topic": topics,
                    "Count": [1] * len(topics)  # Assign count of 1 to each topic for visualization
                })
                fig = px.bar(
                    topic_df, 
                    x="Topic", 
                    y="Count",
                    title="Conversation Topics",
                    color="Topic",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display keywords
            if "keywords" in insights and insights["keywords"]:
                st.subheader("Keywords")
                keywords = insights["keywords"]
                st.write(", ".join(keywords))
    
    # Display messages
    st.subheader("Messages")
    
    for message in st.session_state.messages:
        user_id = message.get("user_id")
        username = user_id  # Default to user_id if username not found
        
        # Check if this message is from the current user
        is_current_user = user_id == st.session_state.active_user["_id"]
        
        # Display message with appropriate styling
        st.markdown(f"""
        <div class="chat-message {'user' if is_current_user else 'other'}">
            <div class="chat-header">{username}</div>
            <div class="chat-content">{message.get('content', '')}</div>
            <div class="chat-timestamp" style="font-size: 0.8rem; color: gray; text-align: right;">
                {datetime.fromisoformat(message.get('timestamp', '').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Send message form
    st.subheader("Send Message")
    message_content = st.text_area("Type your message", height=100)
    
    if st.button("Send"):
        if message_content:
            # If WebSocket is connected, we don't need to call the API directly
            # as the message will be broadcast back to us through the WebSocket
            if st.session_state.websocket_connected:
                # Send message through WebSocket
                try:
                    async def send_ws_message():
                        ws_base_url = API_URL.replace("http://", "ws://").replace("https://", "wss://")
                        ws_url = f"{ws_base_url}/ws/{st.session_state.active_conversation['id']}"
                        
                        async with websockets.connect(ws_url) as websocket:
                            await websocket.send(json.dumps({
                                "type": "message",
                                "user_id": st.session_state.active_user["_id"],
                                "content": message_content
                            }))
                    
                    # Run the async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(send_ws_message())
                    
                    # Clear message input (message will be added via WebSocket)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error sending message via WebSocket: {e}")
                    # Fallback to REST API
                    message = send_message(
                        st.session_state.active_conversation["id"],
                        st.session_state.active_user["_id"],
                        message_content
                    )
                    
                    if message:
                        st.session_state.messages.append(message)
                        st.experimental_rerun()
            else:
                # Fallback to REST API if WebSocket is not connected
                message = send_message(
                    st.session_state.active_conversation["id"],
                    st.session_state.active_user["_id"],
                    message_content
                )
                
                if message:
                    st.session_state.messages.append(message)
                    st.experimental_rerun()
        else:
            st.warning("Message cannot be empty!")