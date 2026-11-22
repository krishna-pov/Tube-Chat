import os
import re
from urllib.parse import urlparse, parse_qs

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="YouTube Transcript Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
      .hero { padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.28);
              border-radius: 18px; margin-bottom: 1.2rem;
              background: linear-gradient(115deg, rgba(120,90,220,.16), rgba(40,130,180,.08)); }
      .hero h1 { margin: 0 0 .35rem 0; font-size: 2.15rem; }
      .hero p { margin: 0; color: #b8bdc8; font-size: 1rem; }
      .info-card { padding: 1rem; border: 1px solid rgba(128,128,128,.28);
                   border-radius: 14px; margin-bottom: .8rem; }
      .muted { color: #9ca3af; font-size: .9rem; }
      div[data-testid="stChatMessage"] { border-radius: 12px; }
      .stButton > button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎬 YouTube Transcript Chatbot</h1>
      <p>Watch a video, ask questions, and explore its transcript with AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def extract_video_id(value: str) -> str:
    """Accept a YouTube URL or a bare video ID and return the video ID."""
    value = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = parsed.netloc.lower().replace("www.", "")

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            video_id = parts[1] if len(parts) > 1 and parts[0] in {"embed", "shorts", "live"} else ""
    else:
        video_id = ""

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Enter a valid YouTube video URL or 11-character video ID.")
    return video_id


@st.cache_resource(show_spinner=False)
def build_chain(video_id: str):
    """Fetch transcript, index it in FAISS, and return the RAG chain."""
    api = YouTubeTranscriptApi()
    fetched_transcript = api.fetch(video_id, languages=["en"])
    transcript = " ".join(snippet.text for snippet in fetched_transcript)
    if not transcript.strip():
        raise ValueError("The transcript is empty.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant answering questions about a YouTube video.\n"
            "Use only the provided transcript context. Do not invent details.\n"
            "Give a clear, reasonably detailed answer. If the context is insufficient, "
            "say that the transcript does not provide enough information.\n\n"
            "Transcript context:\n{context}\n\nQuestion: {question}"
        ),
        input_variables=["question", "context"],
    )

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    return parallel_chain | prompt | model | StrOutputParser(), len(chunks)


# ---------- Session state ----------
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "video_title" not in st.session_state:
    st.session_state.video_title = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = None

# ---------- Video setup ----------
with st.expander("➕ Load a YouTube video", expanded=not bool(st.session_state.video_id)):
    with st.form("video_setup_form"):
        title_input = st.text_input(
            "Video title (for display)",
            value=st.session_state.video_title,
            placeholder="e.g. Interview with a former Navy SEAL",
        )
        url_input = st.text_input(
            "YouTube video URL or video ID",
            value=st.session_state.video_id,
            placeholder="https://www.youtube.com/watch?v=xxxxxxxxxxx",
        )
        load_video = st.form_submit_button("Load video and transcript", type="primary", use_container_width=True)

    if load_video:
        if not url_input.strip():
            st.error("Enter a YouTube video URL or video ID.")
        elif not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is missing. Add it to your .env file before running the app.")
        else:
            try:
                new_video_id = extract_video_id(url_input)
                with st.spinner("Fetching transcript and building the searchable index..."):
                    new_chain, new_chunk_count = build_chain(new_video_id)
                if new_video_id != st.session_state.video_id:
                    st.session_state.messages = []
                st.session_state.video_id = new_video_id
                st.session_state.video_title = title_input.strip()
                st.session_state.chain = new_chain
                st.session_state.chunk_count = new_chunk_count
                st.success("Transcript loaded. You can start chatting!")
            except Exception as exc:
                st.error("I couldn't load this video. Check that the URL is valid and English captions are available.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")

# ---------- Main content ----------
if st.session_state.video_id:
    title = st.session_state.video_title or "Your YouTube video"
    st.subheader(title)
    left_col, main_col = st.columns([0.85, 1.65], gap="large")

    with left_col:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Video details")
        st.caption("Video ID")
        st.code(st.session_state.video_id, language=None)
        st.caption("Transcript index")
        st.write(f"✅ Ready · {st.session_state.chunk_count} chunks")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("#### 💡 Try asking")
        suggestions = [
            "Summarize this video",
            "What are the key takeaways?",
            "Who are the main people discussed?",
            "What are the most important events?",
        ]
        for i, suggestion in enumerate(suggestions):
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()
        st.markdown("<p class='muted'>Answers are generated from the video transcript and may not cover details that are not mentioned in it.</p>", unsafe_allow_html=True)

    with main_col:
        st.markdown("#### ▶️ Watch the video")
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

    st.markdown("---")
    st.markdown("### 💬 Chat with this video")
    st.caption("Ask follow-up questions. Your conversation stays visible while this session is open.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending_question = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask anything about the video transcript...")
    question_to_answer = question or pending_question

    if question_to_answer:
        st.session_state.messages.append({"role": "user", "content": question_to_answer})
        with st.chat_message("user"):
            st.markdown(question_to_answer)

        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript and writing an answer..."):
                try:
                    answer = st.session_state.chain.invoke(question_to_answer)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as exc:
                    st.error(f"I couldn't generate an answer: {exc}")

    if st.session_state.messages:
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()
else:
    st.info("Load a YouTube video above to display the player, prepare its transcript, and start asking questions.")
    st.markdown("#### How it works")
    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("**1. Add a video**")
        st.write("Paste a YouTube URL or video ID.")
    with step2:
        st.markdown("**2. Prepare transcript**")
        st.write("The transcript is split, embedded, and indexed for retrieval.")
    with step3:
        st.markdown("**3. Ask questions**")
        st.write("The chatbot retrieves relevant transcript chunks to answer your questions.")
