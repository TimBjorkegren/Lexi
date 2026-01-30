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
        collection_name=""document_name"",
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
st.title("Dokument chattbott")
st.write("Ladda upp ett dokument och fråga om innehållet")

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_audio_bytes" not in st.session_state: st.session_state.last_audio_bytes = None
if "last_uploaded_collection" not in st.session_state: st.session_state.last_uploaded_collection = None

collections = qdrant.get_collections().collections
collection_names = [c.name for c in collections if c.name.startswith("doc_")]

uploaded_file = st.file_uploader("Välj fil format", type=["txt", "pdf", "docx"])
role = st.selectbox("Välj en roll för AI:n", ["Lärare", "Jurist", "Detektiv", "Sammanfattare"])

if uploaded_file:
    document_text = read_file(uploaded_file)
    chunks = chunk_text(document_text)
    embeddings = create_embeddings(chunks, client)
    collection_name = collection_name_from_file(uploaded_file.name)
    does_collection_exist(collection_name)
    upload_to_qdrant(chunks, embeddings, collection_name)

    st.session_state.last_uploaded_collection = collection_name
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    
# SELECTBOX
    selected_index = 0
    if st.session_state.last_uploaded_collection in collection_names:
        selected_index = collection_names.index(st.session_state.last_uploaded_collection)

    selected_collection = st.selectbox(
    "Välj ett dokument som redan är uppladdat",
    collection_names,
    index=selected_index,
    key="selected_collection"
    )

#AUDIO INPUT
    audio = st.audio_input("Ställ en fråga genom mikrofon")
    if audio is not None:
       current_audio_bytes = audio.getvalue()

       if current_audio_bytes != st.session_state.last_audio_bytes:
           st.session_state.last_audio_bytes = current_audio_bytes
           question = speech_to_text(audio)
           relevant_chunks = search_qdrant(question, selected_collection)
           answer = ask_ai(question, relevant_chunks, speech_file_path)
           autoplay_audio_with_bubble(speech_file_path)

           st.session_state.chat_history.append(
               {"user": question, "bot": answer}
           )

#TEXT INPUT
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Ställ en fråga angående dokumentet", key="question_input")
        send_button = st.form_submit_button("Send")

    if send_button and question:
        relevants_chunks = search_qdrant(question, selected_collection)
        answer = ask_ai(question, relevants_chunks, speech_file_path)
        autoplay_audio_with_bubble(speech_file_path)
        st.session_state.chat_history.append({"user": question, "bot": answer})


# CSS för chattruta
st.markdown(
    """
    <style>
    .chat-container {
        border: 2px solid #ccc;
        border-radius: 10px;
        padding: 10px;
        height: 400px;
        overflow-y: auto;
        background-color: transparent;
    }
    .user-msg {
        background-color: #555555;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 70%;
        word-wrap: break-word;
        margin-left: auto;
        margin-bottom: 10px;
    }
    .bot-msg {
        background-color: #0b3d91;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 70%;
        word-wrap: break-word;
        margin-right: auto;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

chat_html = '<div class="chat-container">'
for chat in st.session_state.chat_history:
    chat_html += f'<div class="user-msg"><strong>You:</strong> {chat["user"]}</div>'
    chat_html += f'<div class="bot-msg"><strong>Lexa:</strong> {chat["bot"]}</div>'
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)