
import os
import re
from urllib.parse import urlparse, parse_qs

import truststore
truststore.inject_into_ssl()

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__, static_folder=STATIC_DIR)

TOP_K = 8
RETRIEVAL_CANDIDATES = 30

FALLBACK_ANSWER = "I don't know based on the provided transcript."

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# 2. INITIALIZE EMBEDDINGS AND GLOBAL VARIABLES
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

vector_store = None
current_video_id = None
transcript_chunks = []

# Remember recent conversation for follow-up questions.
chat_history = []
MAX_HISTORY_MESSAGES = 6


# ============================================================
# 3. EXTRACT YOUTUBE VIDEO ID
# ============================================================

def extract_video_id(video_url):
    if not isinstance(video_url, str):
        return None

    video_url = video_url.strip()

    if re.fullmatch(r"[\w-]{11}", video_url):
        return video_url

    try:
        parsed_url = urlparse(video_url)
        hostname = parsed_url.netloc.lower().replace("www.", "")

        if hostname == "youtu.be":
            video_id = parsed_url.path.strip("/").split("/")[0]
            return (
                video_id
                if re.fullmatch(r"[\w-]{11}", video_id)
                else None
            )

        if hostname in (
            "youtube.com",
            "m.youtube.com",
            "music.youtube.com",
        ):
            if parsed_url.path == "/watch":
                video_id = parse_qs(parsed_url.query).get(
                    "v", [None]
                )[0]
            else:
                match = re.match(
                    r"^/(?:shorts|embed|live)/([\w-]{11})",
                    parsed_url.path,
                )
                video_id = match.group(1) if match else None

            return (
                video_id
                if video_id and re.fullmatch(r"[\w-]{11}", video_id)
                else None
            )

    except Exception as error:
        print("Video ID extraction error:", error)

    return None


# ============================================================
# 4. FETCH YOUTUBE TRANSCRIPT
# ============================================================

def fetch_transcript(video_id):
    api = YouTubeTranscriptApi()

    fetched_transcript = api.fetch(
        video_id,
        languages=["hi", "en"],
    )

    transcript_text = " ".join(
        snippet.text for snippet in fetched_transcript
    )

    if not transcript_text.strip():
        raise ValueError("The transcript for this video is empty.")

    return transcript_text


# ============================================================
# 5. CLEAN TRANSCRIPT
# ============================================================

def clean_transcript(transcript):
    transcript = re.sub(
        r"WEBVTT.*?(?=\n\n|\Z)",
        " ",
        transcript,
        flags=re.DOTALL | re.IGNORECASE,
    )

    transcript = re.sub(
        r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*"
        r"\d{2}:\d{2}:\d{2}[,.]\d{3}",
        " ",
        transcript,
    )

    transcript = re.sub(r"<[^>]+>", " ", transcript)
    transcript = re.sub(r"\s+", " ", transcript).strip()

    replacements = {
        r"\bevaluation[s]?\b": "evaluation",
        r"\bइवैल्यूएशन\b": "evaluation",
        r"\bइवैलुएशन\b": "evaluation",
        r"\bएवैल्यूएशन\b": "evaluation",
        r"\bembeddings?\b": "embedding",
        r"\bएम्बेडिंग्स?\b": "embedding",
        r"\bएल एल एम\b": "LLM",
        r"\bएलएलएम\b": "LLM",
        r"\bचैटबॉट\b": "chatbot",
        r"\bचैट बॉट\b": "chatbot",
        r"\bretrieval\b": "retrieval",
        r"\bरिट्रीवल\b": "retrieval",
        r"\bgeneration\b": "generation",
        r"\bजनरेशन\b": "generation",
        r"\bआर ए जी\b": "RAG",
        r"\bआरएजी\b": "RAG",
        r"\bएजेंट्स\b": "agents",
        r"\bएजेंट\b": "agent",
        r"\bसेफ्टी\b": "safety",
        r"\bसुरक्षा\b": "safety",
    }

    for pattern, replacement in replacements.items():
        transcript = re.sub(
            pattern,
            replacement,
            transcript,
            flags=re.IGNORECASE,
        )

    return transcript


# ============================================================
# 6. PROCESS YOUTUBE VIDEO
# ============================================================

def process_youtube_video(video_url):
    global vector_store, current_video_id
    global transcript_chunks, chat_history

    video_id = extract_video_id(video_url)

    if not video_id:
        raise ValueError(
            "Invalid YouTube URL. Please enter a valid video link."
        )

    print("\n" + "=" * 60)
    print("PROCESSING YOUTUBE VIDEO")
    print("Video ID:", video_id)
    print("=" * 60)

    raw_transcript = fetch_transcript(video_id)
    transcript = clean_transcript(raw_transcript)

    if not transcript:
        raise ValueError("No usable transcript text was found.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.create_documents(
        [transcript],
        metadatas=[{"source": video_id}],
    )

    for index, document in enumerate(chunks):
        document.metadata["chunk_id"] = index

    if not chunks:
        raise ValueError(
            "The transcript could not be divided into chunks."
        )

    print("Transcript length:", len(transcript))
    print("Number of chunks:", len(chunks))
    print("Transcript preview:", transcript[:500])

    new_vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store = new_vector_store
    current_video_id = video_id
    transcript_chunks = chunks

    # Start a fresh conversation for the newly loaded video.
    chat_history = []

    print("Vector store created successfully.")
    print("Conversation history cleared for the new video.")
    print("Active video ID:", current_video_id)
    print("=" * 60)

    return {
        "video_id": video_id,
        "chunks": len(chunks),
    }


# ============================================================
# 7. BUILD RETRIEVAL QUERIES
# ============================================================

def build_retrieval_queries(question):
    queries = [question]
    q = question.lower()

    if any(term in q for term in [
        "main topic", "about this video", "video about",
        "summarize", "summary", "overview",
    ]):
        queries.append(
            "main subject central idea key concepts explained in the video"
        )

    if any(term in q for term in [
        "agent evaluation", "agent-based evaluation",
        "agent based evaluation", "agent evaluations",
        "agent-based evaluations",
    ]):
        queries.append(
            "agent evaluations agent-based evaluations evaluating AI agents"
        )

    if any(term in q for term in [
        "safety evaluation", "safety-based evaluation",
        "safety based evaluation", "safety evaluations",
    ]):
        queries.append(
            "safety evaluations safety testing LLM safety evaluation"
        )

    if "rag" in q and "evaluation" in q:
        queries.append(
            "RAG-specific evaluations retrieval augmented generation evaluation"
        )

    if any(term in q for term in [
        "operational evaluation", "operational evaluations",
    ]):
        queries.append(
            "operational evaluations monitoring applications after deployment"
        )

    if any(term in q for term in [
        "playlist", "topics covered", "topics are covered", "what topics",
    ]):
        queries.append(
            "topics discussed in the video and their order"
        )

    if "golden dataset" in q:
        queries.append(
            "golden dataset curated evaluation examples and rubrics"
        )

    if "chunk size" in q or "chunk overlap" in q:
        queries.append(
            "chunk size chunk overlap splitting text into chunks"
        )

    if "retrieval" in q or "semantic search" in q:
        queries.append(
            "retrieval semantic search relevant documents"
        )

    if "embedding" in q:
        queries.append(
            "embeddings represent text as numerical vectors"
        )

    if any(term in q for term in [
        "main topic", "main idea", "central idea", "summarize",
        "summary", "overview",
    ]):
        queries.append(
            "introduction conclusion purpose and central message"
        )

    return list(dict.fromkeys(queries))


# ============================================================
# 8. KEYWORD RELEVANCE SCORING
# ============================================================

STOPWORDS = {
    "the", "is", "are", "was", "were", "what", "why",
    "how", "when", "where", "who", "does", "do", "did",
    "a", "an", "and", "or", "of", "to", "in", "on",
    "for", "from", "with", "about", "this", "that",
    "it", "its", "be", "been", "being", "based", "say",
    "said", "mention", "mentions", "mentioned", "video",
    "transcript", "tell", "me", "please", "can", "could",
    "would", "should", "explain", "describe", "give",
    "using", "use", "used", "their", "there", "they",
    "them", "these", "those", "into", "than", "then",
    "also", "some", "such", "which", "while", "each",
    "has", "have", "had", "will", "not", "only",
    "doesnt", "dont",

    "और", "का", "की", "के", "में", "है", "हैं",
    "था", "थे", "थी", "को", "से", "पर", "यह",
    "वह", "एक", "जो", "तो", "भी", "लिए", "कर",
    "करता", "करते", "करने", "हो", "होता", "होते",
    "रहा", "रहे", "रही", "अगर", "लेकिन", "या",
    "जब", "तक", "इस", "उस", "अपने", "अपनी",
}


def tokenize(text):
    words = re.findall(
        r"[^\W_]+",
        text.lower(),
        flags=re.UNICODE,
    )

    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def keyword_relevance_score(question, chunk_text):
    question_words = tokenize(question)
    chunk_words = tokenize(chunk_text)

    if not question_words or not chunk_words:
        return 0.0

    overlap = question_words.intersection(chunk_words)
    score = len(overlap) / len(question_words)

    question_lower = question.lower()
    chunk_lower = chunk_text.lower()

    phrase_bonuses = {
        "agent-based evaluations": [
            "agent-based evaluations", "agent based evaluations",
            "agent evaluation", "agent evaluations",
        ],
        "safety-based evaluations": [
            "safety-based evaluations", "safety based evaluations",
            "safety evaluations", "safety evaluation",
        ],
        "rag-specific evaluations": [
            "rag-specific evaluations", "rag specific evaluations",
            "rag evaluations",
        ],
        "golden dataset": [
            "golden dataset", "gold dataset",
        ],
        "operational evaluations": [
            "operational evaluations", "operational evaluation",
        ],
    }

    for key, phrases in phrase_bonuses.items():
        if key in question_lower and any(
            phrase in chunk_lower for phrase in phrases
        ):
            score += 1.0

    return score


# ============================================================
# 9. ANSWER QUESTIONS USING RAG + CONVERSATION HISTORY
# ============================================================

def answer_question(question):
    global chat_history

    if vector_store is None or current_video_id is None:
        return "Please process a YouTube video before asking a question."

    question = question.strip()

    if not question:
        return "Please enter a question."

    history_before_question = chat_history[-MAX_HISTORY_MESSAGES:]

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history_before_question
    )

    follow_up_terms = [
        "that", "it", "they", "them", "this",
        "explain", "simpler", "simplify", "why",
        "how about", "more detail", "example",
        "elaborate", "what about",
    ]

    is_follow_up = (
        len(question.split()) <= 8
        or any(term in question.lower() for term in follow_up_terms)
    )

    retrieval_question = question

    if is_follow_up and history_text:
        retrieval_question = (
            f"{question}\n\n"
            f"Previous conversation:\n{history_text}"
        )

    retrieval_queries = build_retrieval_queries(retrieval_question)

    print("\n" + "=" * 60)
    print("--- RETRIEVAL DEBUG ---")
    print("Active video ID:", current_video_id)
    print("Question:", question)
    print("Follow-up detected:", is_follow_up)
    print("Retrieval queries:", retrieval_queries)

    candidates = {}

    # A. SEMANTIC SEARCH
    for query in retrieval_queries:
        try:
            results = vector_store.similarity_search_with_score(
                query,
                k=min(RETRIEVAL_CANDIDATES, len(transcript_chunks)),
                filter={"source": current_video_id},
            )

            for document, distance in results:
                chunk_id = document.metadata.get("chunk_id")

                if chunk_id is None:
                    continue

                if document.metadata.get("source") != current_video_id:
                    continue

                distance = float(distance)

                if chunk_id not in candidates:
                    candidates[chunk_id] = {
                        "document": document,
                        "distance": distance,
                        "keyword_score": 0.0,
                    }
                elif distance < candidates[chunk_id]["distance"]:
                    candidates[chunk_id]["distance"] = distance

        except Exception as error:
            print("Semantic retrieval error:", error)

    print("Semantic candidates:", len(candidates))

    # B. KEYWORD SEARCH
    keyword_matches = []

    for document in transcript_chunks:
        if document.metadata.get("source") != current_video_id:
            continue

        score = max(
            (
                keyword_relevance_score(query, document.page_content)
                for query in retrieval_queries
            ),
            default=0.0,
        )

        if score > 0:
            keyword_matches.append((score, document))

    keyword_matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    print("Keyword-matched chunks:", len(keyword_matches))

    for score, document in keyword_matches[:TOP_K]:
        chunk_id = document.metadata.get("chunk_id")

        if chunk_id is None:
            continue

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "document": document,
                "distance": float("inf"),
                "keyword_score": score,
            }
        else:
            candidates[chunk_id]["keyword_score"] = score

    print("Unique combined candidates:", len(candidates))

    # C. RANK CANDIDATES
    def candidate_rank(item):
        distance = item["distance"]
        keyword_score = item["keyword_score"]

        if distance == float("inf"):
            return (1, -keyword_score, 0)

        return (
            0,
            distance - (2.0 * keyword_score),
            -keyword_score,
        )

    ranked_candidates = sorted(
        candidates.values(),
        key=candidate_rank,
    )

    selected = ranked_candidates[:TOP_K]

    print("Selected chunks:", len(selected))

    context_parts = []

    for item in selected:
        document = item["document"]
        chunk_id = document.metadata.get("chunk_id")

        print(
            "\nChunk ID:", chunk_id,
            "| Distance:", item["distance"],
            "| Keyword score:", round(item["keyword_score"], 3),
            "| Source:", document.metadata.get("source"),
        )
        print("Chunk preview:", document.page_content[:500])

        context_parts.append(
            f"[Chunk {chunk_id}]\n{document.page_content}"
        )

    if not context_parts:
        print("No relevant transcript chunks were selected.")
        return FALLBACK_ANSWER

    context = "\n\n".join(context_parts)

    # D. ASK GEMINI
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "Missing GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question", "history"],
        template="""
You are a YouTube transcript question-answering assistant.

Answer using the supplied transcript context only.

RULES:

1. The transcript may be in Hindi while the question is in English.
   Understand its meaning and answer in simple English.

2. Use the transcript context as evidence. Do not invent facts or use
   outside knowledge.

3. For broad questions about the video, summarize the central topic
   supported by the transcript.

4. For specific questions, answer directly when the context supports it.

5. Distinguish between a topic that is explained and one that is only
   briefly mentioned.

6. If the transcript says a topic will be covered in another video,
   do not invent an explanation that is not present in the context.

7. Use recent conversation history to understand follow-up questions
   such as "that", "it", "why", and "can you explain it?"

8. When asked to explain something in simpler words:
   - Rephrase the previous answer using easier words.
   - Use short sentences.
   - Explain one idea at a time.
   - Do not simply repeat the same answer.

9. When asked for a simple example:
   - First identify the concept the user is asking about from the
     conversation history.
   - Give a concrete example supported by the transcript.
   - Explain how the example works in simple steps.
   - Do not merely repeat the name of the example.

10. For handwritten-digit recognition, if supported by the supplied
    context, explain that an image is fed into the network, the
    network processes pixel values, and the output neurons represent
    digits from 0 to 9. The neuron with the highest activation
    indicates the network's prediction. Do not claim the prediction
    is always correct.

11. Keep answers focused and easy to understand. Prefer a short
    paragraph or a few bullet points.

12. If the supplied transcript context does not contain enough
    evidence to answer, respond exactly:
    I don't know based on the provided transcript.

13. Do not mention chunk IDs unless the user asks for them.

RECENT CONVERSATION:
{history}

TRANSCRIPT CONTEXT:
{context}

CURRENT USER QUESTION:
{question}

ANSWER:
""",
    )

    chain = prompt_template | llm

    response = chain.invoke({
        "context": context,
        "question": question,
        "history": history_text or "No previous conversation.",
    })

    answer = response.content

    if isinstance(answer, list):
        answer = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in answer
        )

    answer = str(answer).strip()
    answer = answer or FALLBACK_ANSWER

    # Save this question and answer for the next turn.
    chat_history.append({
        "role": "user",
        "content": question,
    })

    chat_history.append({
        "role": "assistant",
        "content": answer,
    })

    # Keep only the latest three question-answer pairs.
    chat_history = chat_history[-MAX_HISTORY_MESSAGES:]

    print("\nGemini answer:", answer)
    print("Saved conversation messages:", len(chat_history))
    print("=" * 60)

    return answer


# ============================================================
# 10. VIDEO SUMMARY FEATURE
# ============================================================

def generate_video_summary():
    if not transcript_chunks or current_video_id is None:
        raise ValueError("Please process a YouTube video first.")

    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "Missing GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
        )

    # Reuse the transcript chunks already created for this video.
    transcript_text = "\n".join(
        document.page_content
        for document in transcript_chunks
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
    )

    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""
You are a helpful YouTube video summarizer.

Summarize the following video transcript in simple English.

Include:
1. Main topic
2. Key points explained in the video
3. Important concepts
4. Short conclusion

Rules:
- Use only information supported by the transcript.
- Do not invent facts.
- Use clear headings and bullet points.
- Keep the summary useful and easy to read.

TRANSCRIPT:
{transcript}

SUMMARY:
""",
    )

    summary_chain = prompt | llm

    response = summary_chain.invoke({
        "transcript": transcript_text
    })

    summary = response.content

    if isinstance(summary, list):
        summary = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in summary
        )

    summary = str(summary).strip()

    if not summary:
        raise ValueError("The AI could not generate a summary.")

    return summary


# ============================================================
# 11. FRONTEND ROUTES
# ============================================================

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Backend is running",
        "video_loaded": current_video_id is not None,
        "current_video_id": current_video_id,
        "chunks": len(transcript_chunks),
    })


# ============================================================
# 13. PROCESS VIDEO API
# ============================================================

@app.route("/api/process-video", methods=["POST"])
def process_video_route():
    try:
        data = request.get_json(silent=True) or {}
        video_url = data.get("url", "")

        if not isinstance(video_url, str) or not video_url.strip():
            return jsonify({
                "status": "error",
                "message": "Please provide a YouTube URL.",
            }), 400

        video_url = video_url.strip()

        # Validate before trying to retrieve the transcript.
        video_id = extract_video_id(video_url)

        if not video_id:
            return jsonify({
                "status": "error",
                "message": "Invalid YouTube URL. Please enter a valid video link.",
            }), 400

        result = process_youtube_video(video_url)

        return jsonify({
            "status": "success",
            "message": "Video processed successfully.",
            "video_id": result["video_id"],
            "chunks": result["chunks"],
        })

    except Exception as error:
        # Detailed error stays in the backend terminal for debugging.
        print("Video processing error:", repr(error))

        error_type = type(error).__name__.lower()
        error_details = str(error).lower()

        transcript_error = (
            "transcript" in error_type
            or "transcript" in error_details
            or "live stream recording is not available" in error_details
            or "no captions" in error_details
            or "subtitles" in error_details
        )

        if transcript_error:
            message = (
                "We couldn't retrieve a transcript for this video. "
                "It may be unavailable, or captions may not be accessible. "
                "Please try another YouTube video."
            )
        elif isinstance(error, ValueError):
            message = str(error)
        else:
            message = (
                "We couldn't process this video. "
                "Please check the URL and try again."
            )

        return jsonify({
            "status": "error",
            "message": message,
        }), 500


# ============================================================
# 14. ASK QUESTION API
# ============================================================

@app.route("/api/ask", methods=["POST"])
def ask_question_route():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "status": "error",
                "message": "Please enter a question.",
            }), 400

        if vector_store is None:
            return jsonify({
                "status": "error",
                "message": "Please process a YouTube video first.",
            }), 400

        answer = answer_question(question)

        return jsonify({
            "status": "success",
            "question": question,
            "answer": answer,
        })

    except Exception as error:
        print("Question answering error:", error)

        return jsonify({
            "status": "error",
            "message": str(error),
        }), 500


# ============================================================
# 15. VIDEO SUMMARY API
# ============================================================

@app.route("/api/summary", methods=["POST"])
def video_summary_route():
    try:
        if vector_store is None or current_video_id is None:
            return jsonify({
                "status": "error",
                "message": "Please process a YouTube video first.",
            }), 400

        summary = generate_video_summary()

        return jsonify({
            "status": "success",
            "video_id": current_video_id,
            "summary": summary,
        })

    except Exception as error:
        print("Video summary error:", error)

        return jsonify({
            "status": "error",
            "message": str(error),
        }), 500


# ============================================================
# 16. START FLASK SERVER
# ============================================================

if __name__ == "__main__":
    print("\nStarting YouTube RAG Chatbot backend...")
    print("Frontend:", FRONTEND_DIR)
    print("Backend: http://127.0.0.1:5000")
    print("Health check: http://127.0.0.1:5000/api/health")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
