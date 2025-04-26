# Chat Summarization and Insights API

A FastAPI-based REST API that processes user chat data, stores conversations in MongoDB, and generates summaries & insights using OpenAI's LLM.

## Features

- **Real-time Chat Processing**: Store and retrieve chat messages.
- **Conversation Management**: Create, retrieve, and delete conversations.
- **User Management**: Create and manage user accounts.
- **Chat Summarization**: Generate conversation summaries using LLM.
- **Conversation Insights**: Analyze sentiment, extract keywords, and identify topics.
- **Streamlit UI**: Interactive interface for chat visualization and interaction.
- **Optimized Database Operations**: Efficient indexing for high-volume CRUD operations.
- **Docker Deployment**: Easy deployment using Docker and docker-compose.

## API Endpoints

- **Store Chat Messages**: `POST /chats`
- **Retrieve Chats**: `GET /chats/{conversation_id}`
- **Summarize Chat**: `POST /chats/summarize`
- **Get User's Chat History**: `GET /users/{user_id}/chats?page=1&limit=10`
- **Delete Chat**: `DELETE /chats/{conversation_id}`
- **Create User**: `POST /users`
- **Create Conversation**: `POST /conversations`
- **Get Conversation Insights**: `GET /chats/{conversation_id}/insights`

## Tech Stack

- **Backend**: FastAPI
- **Database**: MongoDB
- **LLM Integration**: OpenAI API
- **UI**: Streamlit
- **Containerization**: Docker

## Prerequisites

- Docker and Docker Compose
- OpenAI API Key

## Installation and Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/chat-summarization-api.git
   cd chat-summarization-api
   ```

2. **Set up environment variables**

   Copy the `.env.example` file to create a `.env` file:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to add your OpenAI API key and configure other settings.

3. **Build and run with Docker Compose**

   ```bash
   docker-compose up -d
   ```

   This will start three services:
   - FastAPI backend on port 8000
   - MongoDB on port 27017
   - Streamlit UI on port 8501

4. **Initialize database indexes** (automatically done during startup)

   ```bash
   docker-compose exec api python database_setup.py
   ```

## Usage

### API Usage

You can interact with the API using tools like curl, Postman, or directly from your application.

#### Example: Create a user

```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"username":"john_doe","email":"john@example.com"}'
```

#### Example: Create a conversation

```bash
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{"title":"Team Meeting","participants":["user_id_1","user_id_2"]}'
```

#### Example: Send a message

```bash
curl -X POST http://localhost:8000/chats \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"conversation_id","user_id":"user_id","content":"Hello, team!"}'
```

#### Example: Generate a summary

```bash
curl -X POST http://localhost:8000/chats/summarize \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"conversation_id"}'
```

### Streamlit UI

Access the Streamlit UI by opening `http://localhost:8501` in your web browser. The UI provides a user-friendly interface for:

- Creating and managing users
- Creating and joining conversations
- Sending and viewing messages
- Generating and viewing summaries and insights

## Database Optimization

The application uses the following optimization techniques for efficient database operations:

- **Indexing**: Automated index creation for frequently queried fields
- **Async Database Queries**: Using Motor for non-blocking database operations
- **Pagination**: Limiting result sets for large collections
- **Text Indexing**: For efficient keyword search

## Scaling Considerations

- **Horizontal Scaling**: The API is stateless and can be scaled horizontally
- **Database Sharding**: MongoDB supports sharding for larger datasets
- **Caching**: Add Redis for caching frequently accessed data
- **Load Balancing**: Deploy behind a load balancer for high-traffic environments

## Development

To run the application in development mode:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application with auto-reload
uvicorn main:app --reload
```

## Security Considerations

For production deployment, consider:

- Setting restricted CORS policies
- Adding authentication and authorization
- Using HTTPS/TLS
- Securing the MongoDB instance
- Implementing rate limiting
