import os
from dotenv import load_dotenv
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()  # Carga las variables del archivo .env

# ====== Config ======
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_EMB = "sentence-transformers/all-MiniLM-L6-v2"  # dim=384
LLM_MODEL = "llama3-8b-8192"

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY no configurada"); st.stop()
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY no configurada"); st.stop()

# Crear el cliente de Groq para comunicación directa con la API
# Nota: Aquí usamos el cliente nativo de Groq, no LangChain
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    st.sidebar.success("✅ Cliente Pinecone conectado exitosamente")
    client = Groq(api_key=GROQ_API_KEY)
    st.sidebar.success("✅ Cliente Groq conectado exitosamente")
except Exception as e:
    st.sidebar.error(f"❌ Error al conectar con Groq: {str(e)}")
    st.stop()

@st.cache_resource
def get_embedder():
    m = SentenceTransformer(MODEL_EMB)
    return m, m.get_sentence_embedding_dimension()

def embed(texts, model):
    vecs = model.encode(texts, batch_size=32, convert_to_numpy=False)
    return [v.tolist() for v in vecs]

def retrieve(index_name: str, query: str, top_k: int = 5, filter_doc_id: str = None):
    idx = pc.Index(index_name)
    model, _ = get_embedder()
    qv = embed([query], model)[0]
    pine_filter = {"doc_id": {"$eq": filter_doc_id}} if filter_doc_id else None
    res = idx.query(vector=qv, top_k=top_k, include_metadata=True, filter=pine_filter)
    matches = res.get("matches", [])
    # ordenar por score desc
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches

def build_prompt(user_q: str, contexts: list):
    joined = []
    for m in contexts:
        cid = m["id"]
        score = round(m["score"], 4)
        text = m["metadata"]["text"]
        joined.append(f"[{cid} | score={score}]\n{text}")
    context_block = "\n\n---\n".join(joined) if joined else "N/A"

    system = (
        "Eres un asistente que responde SOLO con la información del contexto proporcionado.\n"
        "Si algo no está en el contexto, di claramente que no está en el CV.\n"
        "Responde en español, claro y conciso. Incluye una breve lista de citas con los IDs de los chunks usados."
    )
    user = (
        f"Pregunta del usuario:\n{user_q}\n\n"
        f"Contexto (fragmentos del CV):\n{context_block}\n\n"
        "Instrucción: redacta la respuesta usando únicamente el contexto."
    )
    return system, user

def ask_llm(system_msg: str, user_msg: str):
    chat = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.7,
        max_tokens=1000,
        messages=[
            {"role":"system","content":system_msg},
            {"role":"user","content":user_msg}
        ]
    )
    return chat.choices[0].message.content

# ====== UI ======
st.set_page_config(page_title="Chatbot RAG CV", page_icon="🤖", layout="centered")
st.title("🤖 Chatbot RAG sobre tu CV (Pinecone + Groq)")

index_name = st.text_input("Nombre del índice de Pinecone", value="cv-floro-384")
doc_id = st.text_input("Filtrar por doc_id (opcional)", value="cv-floro")
top_k = st.slider("Top K", 1, 10, 5)

st.markdown("Escribe una pregunta sobre tu CV (p.ej., *¿Qué experiencia tengo con Next.js?*).")
q = st.text_input("Tu pregunta", "")

if st.button("Consultar") and q.strip():
    with st.spinner("Buscando contexto en Pinecone..."):
        matches = retrieve(index_name, q.strip(), top_k=top_k, filter_doc_id=doc_id.strip() or None)
    if not matches:
        st.warning("No se hallaron fragmentos relevantes en el índice.")
    else:
        system_msg, user_msg = build_prompt(q.strip(), matches)
        with st.spinner("Generando respuesta con Groq..."):
            answer = ask_llm(system_msg, user_msg)

        st.markdown("### 🧠 Respuesta")
        st.write(answer)

        st.markdown("### 📌 Fragmentos citados")
        for m in matches:
            st.markdown(f"- **{m['id']}** (score={round(m['score'],4)})")
        with st.expander("Ver textos de los fragmentos"):
            for m in matches:
                st.markdown(f"**{m['id']}** — {m['metadata'].get('len', '?')} chars")
                st.write(m["metadata"]["text"])
                st.markdown("---")
