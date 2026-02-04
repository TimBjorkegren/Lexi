import os
from openai import OpenAI
import streamlit as st
from PyPDF2 import PdfReader
import docx
import numpy as np
from dotenv import load_dotenv
import base64
import tempfile
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

speech_file_path = "output.wav"

# QDRANT
def does_collection_exist(collection_name):
    collections = qdrant.get_collections().collections
    existing = [c.name for c in collections]
    if collection_name not in existing:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE
            )
        )

def collection_name_from_file(filename):
    name = filename.lower().replace(" ", "_").replace(".", "_")
    return f"doc_{name}"
    
def upload_to_qdrant(chunks, embeddings, document_name):
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            payload={"text": chunk, "document_name": document_name}
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    qdrant.upsert(
        collection_name=document_name,
        points=points
    )

def search_qdrant(question, collection_name, top_k=5):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_embedding = response.data[0].embedding

    results = qdrant.query_points(
        collection_name=collection_name,
        prefetch=[],
        query=query_embedding,
        limit=top_k
    )

    texts = []
    for matches in results.points:
        texts.append(matches.payload["text"])

    
    return texts

# FILE
def read_file(file):
    text = ""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        st.error("Filformatet stöds inte!")
    return text

def chunk_text(text, chunksize=800):
    chunks = []
    for i in range(0, len(text), chunksize):
        chunks.append(text[i:i + chunksize])
    return chunks

def create_embeddings(chunks, client):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

# -------------------------------------------AUDIO----------------------------

def autoplay_audio_with_bubble(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()

    html = f"""
    <style>
      .bubble {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #6fb1ff, #1e3c72);
        margin: 20px auto;
        transition: transform 0.05s linear;
        box-shadow: 0 0 20px rgba(0,123,255,0.6);
      }}
    </style>

    <div class="bubble" id="bubble"></div>

    <audio id="audio" autoplay>
      <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>

    <script>
      const audio = document.getElementById("audio");
      const bubble = document.getElementById("bubble");

      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioContext();
      const source = ctx.createMediaElementSource(audio);
      const analyser = ctx.createAnalyser();

      analyser.fftSize = 256;
      source.connect(analyser);
      analyser.connect(ctx.destination);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      function animate() {{
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {{
          sum += dataArray[i];
        }}
        let volume = sum / dataArray.length;
        let scaleVal = 1 + volume / 300;
        bubble.style.transform = "scale(" + scaleVal + ")";
        requestAnimationFrame(animate);
      }}

      audio.onplay = () => {{
        if (ctx.state === "suspended") ctx.resume();
        animate();
      }};
    </script>
    """
    st.components.v1.html(html, height=130)



def speech_to_text(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    return transcript.text

# --------------------------------AI------------------------------------

def ask_ai(question, relevant_chunks, speech_file_path):
    response = client.chat.completions.create(
    model="gpt-4.1-nano-2025-04-14",
    messages= [
        {"role": "system", "content": f"Du är en hjälpsam {role}. Svara endast baserat på informationen som skickas i användarmeddelandet."},
        {"role": "user", "content": f"KONTEXT:\n{relevant_chunks}\n\nFRÅGA:\n{question}"}
        ],
        max_tokens=500
    )

    response_tts = response.choices[0].message.content

    with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="coral",
    input=response_tts,
    instructions="Speak in a cheerful and positive tone.",
    ) as tts_response:
        tts_response.stream_to_file(speech_file_path)
    return response_tts

# ---------------------------------UI-----------------------------
st.set_page_config(page_title="Lexa AI", layout="centered")

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_audio_bytes" not in st.session_state: st.session_state.last_audio_bytes = None
if "last_uploaded_collection" not in st.session_state: st.session_state.last_uploaded_collection = None

with st.sidebar:
    st.title("inställningar")
    role = st.selectbox("AI Role", ["Lärare", "Jurist", "Detektiv", "Sammanfattare"])

    st.divider()

    uploaded_file = st.file_uploader("Välj fil format", type=["txt", "pdf", "docx"])
    if uploaded_file:
        with st.spinner("Bearbetar dokument..."):
            document_text = read_file(uploaded_file)
            chunks = chunk_text(document_text)
            embeddings = create_embeddings(chunks, client)
            collection_name = collection_name_from_file(uploaded_file.name)
            does_collection_exist(collection_name)
            upload_to_qdrant(chunks, embeddings, collection_name)
            st.session_state.last_uploaded_collection = collection_name
            st.success("Uppladning klar")

    
    collections = qdrant.get_collections().collections
    collection_names = [c.name for c in collections if c.name.startswith("doc_")]
    selected_collection = st.selectbox("Välj dokument", collection_names)


st.title("Lexi")

audio = st.audio_input("Ställ en fråga genom mikrofon")
if audio and audio.getvalue() != st.session_state.last_audio_bytes:
    st.session_state.last_audio_bytes = audio.getvalue()
    if selected_collection:
        with st.spinner("Tänker..."):
           question = speech_to_text(audio)
           relevant_chunks = search_qdrant(question, selected_collection)
           answer = ask_ai(question, relevant_chunks, role)
           st.session_state.chat_history.append({"user": question, "bot": answer})
           st.rerun()
    else:
        st.warning("Ladda upp eller välj ett dokument först")      

st.markdown("""
    <style>
    .user-msg { background-color: #555555; color: white; padding: 10px 15px; border-radius: 15px; margin-bottom: 10px; width: fit-content; margin-left: auto; }
    .bot-msg { background-color: #0b3d91; color: white; padding: 10px 15px; border-radius: 15px; margin-bottom: 10px; width: fit-content; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)


for chat in st.session_state.chat_history:
    st.markdown(f'<div class="user-msg"><b>Du:</b> {chat["user"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-msg"><b>Lexa:</b> {chat["bot"]}</div>', unsafe_allow_html=True)

if st.session_state.chat_history:
    autoplay_audio_with_bubble(speech_file_path)

if question := st.chat_input("Skriv din fråga här"):
    if selected_collection:
        relevant_chunks = search_qdrant(question, selected_collection)
        answer = ask_ai(question, relevant_chunks, role)
        st.session_state.chat_history.append({"user": question, "bot": answer})
        st.rerun()
    else:
        st.error("Ladda upp ett dokument först")