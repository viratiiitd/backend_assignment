from motor.motor_asyncio import AsyncIOMotorClient
import os
import asyncio
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get MongoDB connection string
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "chat_db")

async def setup_indexes():
    """
    Set up indexes for MongoDB collections to optimize query performance
    """
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        logger.info(f"Connected to MongoDB database: {DATABASE_NAME}")

        # Create indexes for the messages collection
        logger.info("Creating indexes for the messages collection...")
        await db.messages.create_index("conversation_id")
        await db.messages.create_index("user_id")
        await db.messages.create_index("timestamp")
        await db.messages.create_index([("content", "text")])  # Text index for search functionality

        # Create indexes for the conversations collection
        logger.info("Creating indexes for the conversations collection...")
        await db.conversations.create_index("participants")
        await db.conversations.create_index("updated_at")
        await db.conversations.create_index([("title", "text")])  # Text index for search functionality

        # Create indexes for the users collection
        logger.info("Creating indexes for the users collection...")
        await db.users.create_index("username", unique=True)
        await db.users.create_index("email", unique=True)

        logger.info("All indexes created successfully!")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
    finally:
        # Close the MongoDB connection
        client.close()
        logger.info("MongoDB connection closed")

if __name__ == "__main__":
    # Run the async function
    asyncio.run(setup_indexes())