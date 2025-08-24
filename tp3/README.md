# Procesamiento de Lenguaje Natural II - CEIA - FIUBA
# TP3 — Chatbot de **Agentes** con RAG (Pinecone + Groq + Streamlit)
# Alumno: Florentino Arias

Video Demo: [Video tp3 GoogleDrive](https://drive.google.com/file/d/1sHpF_Wok0KOSmpzt0NBA0i8CJBQFDjR3/view?usp=sharing)

Sistema que, dado una consulta, **detecta a qué persona(s)** se refiere (por **RegEx**),
recupera contexto desde **Pinecone** (**1 índice por CV**) y genera la respuesta con un LLM
(**Groq / Llama 3**), **citando** los fragmentos utilizados.

> Si no se menciona a nadie, responde el **agente por defecto (alumno)**.  
> Si se mencionan **varias personas**, trae contexto de **cada** una y responde acorde.

---

## 🚀 Demo rápida

```shell
# 1) instalar deps
pip install -U pinecone-client sentence-transformers streamlit groq python-docx pypdf docx2txt

# 2) variables de entorno
# Poner en archivo .env
PINECONE_API_KEY="..."
GROQ_API_KEY="..."

# 3) ingestar cada CV (1 índice por CV; el script crea el índice si no existe)
# Acepta docx y pdf
python ingestar_cv_pinecone.py --file ./cv_floro.docx --index cv-floro-384 --doc_id cv-floro
python ingestar_cv_pinecone.py --file ./cv_german.docx   --index cv-german-384   --doc_id cv-german

# 4) configurar agentes en chatbot_rag_agentes.py (bloque AGENTS)

# 5) correr el chatbot y preguntarle
streamlit run chatbot_rag_agentes.py

