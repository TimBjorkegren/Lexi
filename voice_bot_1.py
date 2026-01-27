import os
from openai import OpenAI
import streamlit as st
from PyPDF2 import PdfReader
import docx
import numpy as np
from dotenv import load_dotenv

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

def setup_qdrant():
    qdrant.recreate_collection(
    collection_name="documents",
    vectors_config= VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)
    
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
        collection_name="documents",
        points=points
    )

def search_qdrant(question, top_k=5):
    # 1️⃣ Skapa embedding för frågan
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_embedding = np.array(response.data[0].embedding, dtype=float)
    print(f"\nQuery embedding length: {len(query_embedding)} Sample: {query_embedding[:5]}")  # DEBUG

    # 2️⃣ Hämta alla points från Qdrant
    scroll_response = qdrant.scroll(
        collection_name="documents",
        limit=1000,
        with_vectors=True
    )
    points = scroll_response[0]  # själva listan med points

    print(f"Number of points fetched: {len(points)}")  # DEBUG

    # 3️⃣ Filtrera bort points utan vector och plocka ut text och document_name
    valid_points = [
        (np.array(point.vector, dtype=float), point.payload.get("text", ""), point.payload.get("document_name", "Okänt dokument"))
        for point in points
        if point.vector is not None
    ]
    print(f"Valid points with vectors: {len(valid_points)}")  # DEBUG

    # 4️⃣ Beräkna cosine similarity mellan frågan och varje chunk
    similarities = []
    for vec, text, doc_name in valid_points:
        sim = np.dot(query_embedding, vec) / (np.linalg.norm(query_embedding) * np.linalg.norm(vec))
        print(f"Similarity for document '{doc_name}': {sim:.4f} Text start: {text[:50]}")  # DEBUG
        similarities.append((sim, text, doc_name))

    # 5️⃣ Sortera på högsta likhet
    similarities.sort(reverse=True, key=lambda x: x[0])

    # 6️⃣ Returnera top-k mest relevanta
    top_chunks = [{"text": text, "document_name": doc_name} for _, text, doc_name in similarities[:top_k]]

    print("\nTop-k matches returned:")
    for i, chunk in enumerate(top_chunks):
        print(f"{i+1}: {chunk['document_name']} | {chunk['text'][:50]}")

    return top_chunks






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



def ask_ai(question, relevant_chunks):
    context_chunks = [
        f"[{chunk['document_name']}] {chunk['text']}" 
        for chunk in relevant_chunks 
        if chunk["text"].strip() != ""
    ]
    
    if not context_chunks:
        return "Inga relevanta textbitar hittades i dokumentet."

    context = "\n\n".join(context_chunks)
    
    response = client.chat.completions.create(
    model="gpt-4.1-nano-2025-04-14",
    messages= [
        {"role": "system", "content": "Du är en hjälpsam AI-assistent. Svara endast baserat på informationen som skickas i användarmeddelandet."},
        {"role": "user", "content": f"KONTEXT:\n{context}\n\nFRÅGA:\n{question}"}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content


st.title("Dokument chattbott")
st.write("Ladda upp ett dokument och fråga om innehållet")

uploaded_file = st.file_uploader("Välj fil format", type=["txt", "pdf", "docx"])

if uploaded_file:
    document_text = read_file(uploaded_file)

    chunks = chunk_text(document_text)
    embeddings = create_embeddings(chunks, client)

    setup_qdrant()
    upload_to_qdrant(chunks, embeddings, uploaded_file.name)

    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.success("File was successfully uploaded")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    
    question = st.text_input("Ställ en fråga angående dokumentet")

    if st.button("Send") and question:
        relevants_chunks = search_qdrant(question)
        for i in relevants_chunks:
            print(i["document_name"], i["text"])

        answer = ask_ai(question, relevants_chunks)

        st.session_state.chat_history.append({"user": question, "bot": answer})


    for chat in st.session_state.chat_history:
        st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: flex-end;
            margin-bottom: 10px;
        ">
            <div style="
                background-color: #555555;
                padding: 10px 15px;
                border-radius: 15px;
                max-width: 70%;
                word-wrap: break-word;
            ">
                <strong>You:</strong> {chat['user']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
        st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: flex-start;
            margin-bottom: 10px;
        ">
            <div style="
                background-color: #0b3d91;
                padding: 10px 15px;
                border-radius: 15px;
                max-width: 70%;
                word-wrap: break-word;
            ">
                <strong>Lexa:</strong> {chat['bot']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )