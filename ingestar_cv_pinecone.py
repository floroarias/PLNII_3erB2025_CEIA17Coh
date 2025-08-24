import os
from dotenv import load_dotenv
import re
import time
from typing import List, Dict
# import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()  # Carga las variables del archivo .env

# (opcionales) para leer docx/pdf
try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# ====== Config Pinecone ======
def configurar_pinecone():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY no configurada")
    pc = Pinecone(api_key=api_key)
    print(f"✅ Pinecone configurado")
    return pc

def asegurar_indice(nombre_indice: str, dimension: int, metrica: str = "cosine", pc: Pinecone = None):
    indices = pc.list_indexes().names()
    if nombre_indice in indices:
        print(f"ℹ️ Índice '{nombre_indice}' ya existe")
        return
    pc.create_index(
        name=nombre_indice,
        dimension=dimension,
        metric=metrica,
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(f"⏳ Creando índice '{nombre_indice}'...")
    while nombre_indice not in pc.list_indexes().names():
        time.sleep(1)
    print(f"✅ Índice '{nombre_indice}' listo")


# ====== Embeddings ======
class Embeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Modelo {model_name} (dim={self.dim})")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, batch_size=32, convert_to_numpy=False)
        return [v.tolist() for v in vecs]


# ====== Lectura y chunking ======
def leer_texto(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".txt":
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if ext == ".docx":
        if not docx2txt:
            raise RuntimeError("Instala python-docx o docx2txt para leer .docx")
        return docx2txt.process(path) or ""
    if ext == ".pdf":
        if not PdfReader:
            raise RuntimeError("Instala pypdf para leer .pdf")
        reader = PdfReader(path)
        out = []
        for page in reader.pages:
            out.append(page.extract_text() or "")
        return "\n".join(out)
    raise ValueError("Formato no soportado. Usa .txt, .docx o .pdf")

def limpiar_texto(t: str) -> str:
    t = t.replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_por_palabras(texto: str, target_palabras=180, overlap_palabras=30) -> List[str]:
    # Split en párrafos y luego “pegar” hasta ~target_palabras
    paras = [p.strip() for p in texto.split("\n\n") if p.strip()]
    chunks = []
    actual = []
    count = 0
    for p in paras:
        palabras = p.split()
        if count + len(palabras) <= target_palabras or not actual:
            actual.append(p)
            count += len(palabras)
        else:
            chunks.append("\n\n".join(actual))
            # solapamiento
            keep = " ".join(" ".join(actual).split()[-overlap_palabras:])
            actual = [keep, p]
            count = len(keep.split()) + len(palabras)
    if actual:
        chunks.append("\n\n".join(actual))
    # Fallback si el CV es muy corto
    if not chunks:
        chunks = [texto]
    return chunks


# ====== Upsert ======
def upsert_cv(
    index_name: str,
    doc_id: str,
    chunks: List[str],
    emb: Embeddings,
    extra_metadata: Dict = None,
    pc: Pinecone = None
):
    idx = pc.Index(index_name)
    vectors = []
    texts = [c for c in chunks]
    vecs = emb.embed_batch(texts)

    for i, (text, v) in enumerate(zip(texts, vecs)):
        vectors.append({
            "id": f"{doc_id}::chunk-{i:04d}",
            "values": v,
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": i,
                "len": len(text),
                "text": text,
                **(extra_metadata or {})
            }
        })
    # upsert en lotes seguros
    B = 100
    for i in range(0, len(vectors), B):
        idx.upsert(vectors=vectors[i:i+B])
    stats = idx.describe_index_stats()
    print(f"🎉 Ingestado '{doc_id}' → {len(chunks)} chunks | total={stats.get('total_vector_count', '¿?')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingestar un CV a Pinecone (1 índice por CV)")
    parser.add_argument("--file", required=True, help="Ruta al CV (.txt/.docx/.pdf)")
    parser.add_argument("--index", required=True, help="Nombre del índice (ej: cv-floro-384)")
    parser.add_argument("--doc_id", required=True, help="ID lógico del documento (ej: cv-floro)")
    args = parser.parse_args()

    pc = configurar_pinecone()
    emb = Embeddings("sentence-transformers/all-MiniLM-L6-v2")
    asegurar_indice(args.index, dimension=emb.dim, metrica="cosine", pc=pc)

    raw = leer_texto(args.file)
    texto = limpiar_texto(raw)
    chunks = chunk_por_palabras(texto, target_palabras=180, overlap_palabras=30)

    upsert_cv(args.index, args.doc_id, chunks, emb, extra_metadata={"tipo": "cv"}, pc=pc)
