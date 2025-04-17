FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY streamlit-requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r streamlit-requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Expose the port the app runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]