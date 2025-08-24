import os
import re
from typing import List, Dict, Tuple
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Carga las variables del archivo .env

# ========= CONFIG =========
# ====== Config ======
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_EMB = "sentence-transformers/all-MiniLM-L6-v2"  # dim=384
LLM_MODEL = "llama3-8b-8192"

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY no configurada"); st.stop()
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY no configurada"); st.stop()

# Inicializar Pinecone y Groq
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    st.sidebar.success("✅ Cliente Pinecone conectado exitosamente")
    groq = Groq(api_key=GROQ_API_KEY)
    st.sidebar.success("✅ Cliente Groq conectado exitosamente")
except Exception as e:
    st.sidebar.error(f"❌ Error al conectar con Groq: {str(e)}")
    st.stop()

@st.cache_resource
def get_embedder():
    m = SentenceTransformer(MODEL_EMB)
    return m, m.get_sentence_embedding_dimension()

def embed(texts: List[str]) -> List[List[float]]:
    model, _ = get_embedder()
    vecs = model.encode(texts, batch_size=32, convert_to_numpy=False)
    return [v.tolist() for v in vecs]

# ========= AGENTES (1 índice por CV) =========
AGENTS: Dict[str, Dict] = {
    "floro": {
        "aliases": [r"\bfloro\b", r"\bflorentino\b", r"\byo\b", r"\bmi\s+cv\b", r"\bflorito\b", r"\bflori\b", r"\barias\b"],
        "index": "cv-floro-384",
        "doc_id": "cv-floro",
    },
    "german": {
        "aliases": [r"\bgerman\b", r"\bger\b", r"\bborto\b", r"\bgermán\b", r"\bbortolotti\b", r"\bbortoloti\b"],
        "index": "cv-german-384",
        "doc_id": "cv-german",
    },
    # puedo agregar más agentes aquí...
}
# El "default_agent" apunta a mi CV (floro).
DEFAULT_AGENT_KEY = "floro"

# Compila patrones de RegEx una sola vez
COMPILED = {
    k: [re.compile(pat, flags=re.I) for pat in cfg["aliases"]]
    for k, cfg in AGENTS.items()
}

# ========= DECISOR (Conditional Edge) =========
def decidir_agentes(query: str) -> List[str]:
    """Devuelve lista de agent_keys mencionados en la query. Si ninguno, devuelve [DEFAULT_AGENT_KEY]."""
    q = query or ""
    q = re.sub(r"\s+", " ", q.lower()).strip()
    encontrados = []
    for key, pats in COMPILED.items():
        if any(p.search(q) for p in pats):
            encontrados.append(key)
    # eliminar duplicados preservando orden
    vistos = set()
    final = [x for x in encontrados if not (x in vistos or vistos.add(x))]
    return final or [DEFAULT_AGENT_KEY]

# ========= RETRIEVAL por agente =========
def retrieve_from_agent(agent_key: str, query: str, top_k: int = 4) -> List[Dict]:
    agent = AGENTS[agent_key]
    idx = pc.Index(agent["index"])
    qv = embed([query])[0]
    pine_filter = {"doc_id": {"$eq": agent.get("doc_id")}} if agent.get("doc_id") else None
    res = idx.query(vector=qv, top_k=top_k, include_metadata=True, filter=pine_filter)
    matches = res.get("matches", [])
    # anota el agente para trazabilidad
    for m in matches:
        m["agent"] = agent_key
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches

def retrieve_multi(agents: List[str], query: str, top_k_per_agent: int = 4) -> List[Dict]:
    all_matches = []
    for a in agents:
        try:
            all_matches.extend(retrieve_from_agent(a, query, top_k=top_k_per_agent))
        except Exception as e:
            all_matches.append({
                "id": f"{a}::ERROR",
                "score": 0.0,
                "metadata": {"text": f"Error consultando índice de {a}: {e}", "doc_id": AGENTS[a].get("doc_id","")},
                "agent": a
            })
    # orden global por score
    all_matches.sort(key=lambda m: m["score"], reverse=True)
    return all_matches

# ========= PROMPTING =========
def build_prompt(user_q: str, matches: List[Dict], per_agent_limit: int = 4) -> Tuple[str, str]:
    """
    Si hay varios agentes, construye contexto separado por agente.
    Limita a per_agent_limit fragmentos por agente para no sobrecargar el prompt.
    """
    # agrupar por agente
    by_agent: Dict[str, List[Dict]] = {}
    for m in matches:
        by_agent.setdefault(m["agent"], []).append(m)

    blocks = []
    for agent, lst in by_agent.items():
        sl = lst[:per_agent_limit]
        lines = []
        for m in sl:
            cid = m["id"]; score = round(m.get("score", 0.0), 4)
            txt = (m.get("metadata", {}) or {}).get("text", "")
            lines.append(f"[{agent} | {cid} | score={score}]\n{txt}")
        blocks.append(f"### Contexto de {agent}\n" + "\n\n".join(lines))

    context = "\n\n---\n\n".join(blocks) if blocks else "N/A"

    system = (
        "Eres un asistente que responde SOLO con el contexto provisto (RAG sobre CVs).\n"
        "Responde en español, con precisión y sin inventar.\n"
        "Si hay varios agentes, responde por secciones dejando claro a quién corresponde cada dato.\n"
        "Incluye una lista breve de citas usando el formato [agente | chunk-id]."
    )
    user = f"Pregunta: {user_q}\n\n{context}\n\nInstrucción: Redacta la respuesta usando únicamente el contexto. Si algo no está en el/los CV/s, dilo."
    return system, user

def ask_llm(system_msg: str, user_msg: str) -> str:
    chat = groq.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.7,
        max_tokens=1000,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
    )
    return chat.choices[0].message.content

# ========= UI =========
st.set_page_config(page_title="TP3 - Chatbot de Agentes (RAG + Pinecone)", page_icon="🕵️", layout="centered")
st.title("🕵️ Chatbot de Agentes — CVs con RAG (Pinecone)")

st.markdown("Consulta por una o varias personas. Si no nombrás a nadie, responde tu **agente por defecto**.")

query = st.text_input("Escribí tu pregunta", placeholder="¿Qué experiencia tiene Germán como Procurador? ¿Y Floro en MySQL?")
top_k_agent = st.slider("Top K por agente", 1, 10, 4)

# debug opcional
with st.expander("Ver configuración de agentes"):
    st.json({k: {"index": v["index"], "doc_id": v.get("doc_id"), "aliases": v["aliases"]} for k, v in AGENTS.items()})
    st.write(f"Agente por defecto: **{DEFAULT_AGENT_KEY}**")

if st.button("Consultar") and query.strip():
    agents = decidir_agentes(query)
    st.info(f"Agentes detectados: **{', '.join(agents)}**")

    with st.spinner("Buscando contexto..."):
        matches = retrieve_multi(agents, query, top_k_per_agent=top_k_agent)

    if not matches:
        st.warning("No se hallaron fragmentos relevantes.")
    else:
        system_msg, user_msg = build_prompt(query, matches, per_agent_limit=top_k_agent)
        with st.spinner("Generando respuesta con LLM..."):
            answer = ask_llm(system_msg, user_msg)

        st.markdown("## 🧠 Respuesta")
        st.write(answer)

        st.markdown("## 📎 Citas / Fragmentos")
        for m in matches:
            st.markdown(f"- **{m['agent']}** · `{m['id']}` · score={round(m.get('score',0.0),4)}")
        with st.expander("Ver texto de los fragmentos"):
            for m in matches:
                st.markdown(f"**{m['agent']}** — `{m['id']}`")
                st.write((m.get("metadata",{}) or {}).get("text",""))
                st.markdown("---")
