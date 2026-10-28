# YouTube RAG Chatbot

A web-based AI application that lets users ask questions about YouTube videos and receive answers based on the video's transcript. It also generates video summaries using a Large Language Model (LLM).

## Features

- **YouTube Video Processing:** Extracts transcripts from supported YouTube videos.
- **Question Answering:** Ask questions about the video and get context-based answers.
- **Video Summarization:** Generate a structured summary of the video.
- **RAG Architecture:** Uses Retrieval-Augmented Generation to retrieve relevant transcript content before generating answers.
- **Semantic Search:** Uses embeddings and FAISS to find relevant transcript chunks.
- **Chat History:** Maintains recent conversation context during a session.
- **Error Handling:** Displays helpful messages for invalid URLs and unavailable transcripts.
- **Simple Web Interface:** Built with HTML, CSS, and JavaScript, with a Flask backend.

## Technologies Used

- Python
- Flask
- HTML, CSS, and JavaScript
- YouTube Transcript API
- LangChain
- Google Gemini
- Hugging Face Sentence Transformers
- FAISS Vector Database
- Retrieval-Augmented Generation (RAG)

## How It Works

1. The user enters a YouTube video URL.
2. The application extracts the video's transcript.
3. The transcript is split into smaller chunks.
4. Hugging Face embeddings convert the chunks into numerical vector representations.
5. FAISS stores the vectors and helps retrieve relevant content.
6. When the user asks a question, the application retrieves relevant transcript chunks.
7. Google Gemini generates an answer using the retrieved context.
8. Users can also generate a summary of the video.

## Project Structure

```text
youtube-rag-chatbot/
├── backend/
│   └── app.py
├── frontend/
│   ├── index.html
│   └── static/
│       └── style.css
├── main youtube.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Prerequisites

- Python 3.10 or a compatible Python version for the installed dependencies
- A Google Gemini API key
- Internet connection

## Setup and Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

Replace the placeholder values with your actual repository URL and folder name.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root and add your Gemini API key:

```text
GOOGLE_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key. **Never publish your actual API key or `.env` file.**

### 5. Run the application

From the project root, run:

```bash
python backend/app.py
```

Open the local URL displayed in the terminal, typically `http://127.0.0.1:5000`.

## Important Notes

- Transcript availability depends on the YouTube video and its available captions.
- AI-generated answers and summaries may occasionally contain errors.
- A valid Gemini API key and internet connection are required for AI features.

## Future Improvements

- Support for additional languages and transcript sources.
- Improved retrieval and answer relevance.
- Deployment so users can access the application online.

## Author

Developed as a Python and AI project to explore Retrieval-Augmented Generation, semantic search, vector databases, and LLM-powered applications.