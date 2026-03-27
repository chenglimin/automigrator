import os
import json
import re
import html
import xml.etree.ElementTree as ET
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ======================================================
# 配置
# ======================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# ======================================================
# Embedding 模型（唯一模型）
# ======================================================
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed(texts):
    return embed_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

# ======================================================
# 工具函数
# ======================================================
def clean_html(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()

def batch_add(collection, documents, metadatas, embeddings, ids, batch_size=1000):
    total = len(documents)
    for i in range(0, total, batch_size):
        collection.add(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )
        print(f"  -> 已写入 {min(i + batch_size, total)}/{total}")

# ======================================================
# 1. 解析 Intel XML
# ======================================================
def load_intel_xml(xml_path):
    docs = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for ins in root.findall("intrinsic"):
        name = ins.get("name")
        tech = ins.get("tech", "Unknown")

        desc = ins.findtext("description", "")
        ret = ins.find("return")
        ret_type = ret.get("type") if ret is not None else "void"

        params = []
        for p in ins.findall("parameter"):
            params.append(f"{p.get('type')} {p.get('varname')}")

        signature = f"{ret_type} {name}({', '.join(params)})"

        content = (
            f"Intrinsic: {name}\n"
            f"Signature: {signature}\n"
            f"Description: {desc}\n"
            f"Technology: {tech}"
        )

        docs.append((content, {
            "name": name,
            "tech": tech,
            "raw_desc": desc
        }))

    return docs

# ======================================================
# 2. 解析 RVV JSON
# ======================================================
def load_rvv_json(json_path):
    docs = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        items = data if isinstance(data, list) else data.get("intrinsics", [])

    for item in items:
        raw_name = item.get("函数名", "unknown")
        clean_name = raw_name.replace("_", " ").replace("|", " ")

        raw_cat = item.get("Category", ["General"])[0]
        category = raw_cat.replace("|", " ").replace("_", " ")

        desc = clean_html(item.get("Description", ""))
        instr = clean_html(item.get("Instruction", ""))
        op = clean_html(item.get("Operation", ""))
        arch = item.get("Architecture", "Unknown")

        content = (
            f"Intrinsic Name: {clean_name}\n"
            f"Architecture: {arch}\n"
            f"Category: {category}\n"
            f"Description: {desc}\n"
            f"Instruction: {instr}\n"
            f"Operation:\n{op}"
        )

        docs.append((content, {
            "name": raw_name,
            "search_name": clean_name,
            "category": category,
            "architecture": arch
        }))

    return docs

# ======================================================
# 3. 构建 Chroma 向量库
# ======================================================
if __name__ == "__main__":
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # ---------- x86 ----------
    x86_path = DATA_DIR / "intel_intrinsics-2.xml"
    if x86_path.exists():
        x86_docs = load_intel_xml(x86_path)
        col = client.get_or_create_collection("x86")

        texts = [d[0] for d in x86_docs]
        metas = [d[1] for d in x86_docs]
        ids = [f"x86_{i}" for i in range(len(texts))]

        embeddings = embed(texts)

        batch_add(
            collection=col,
            documents=texts,
            metadatas=metas,
            embeddings=embeddings,
            ids=ids
        )

        print(f"✅ x86 写入 {len(texts)} 条")

    # ---------- RVV ----------
    rvv_path = DATA_DIR / "riscv_intrinsics_combined.json"
    if rvv_path.exists():
        rvv_docs = load_rvv_json(rvv_path)
        col = client.get_or_create_collection("rvv")

        texts = [d[0] for d in rvv_docs]
        metas = [d[1] for d in rvv_docs]
        ids = [f"rvv_{i}" for i in range(len(texts))]

        embeddings = embed(texts)

        batch_add(
            collection=col,
            documents=texts,
            metadatas=metas,
            embeddings=embeddings,
            ids=ids
        )

        print(f"✅ RVV 写入 {len(texts)} 条")

        # ---------- Category 路由 ----------
        categories = sorted({m["category"] for m in metas})
        cat_col = client.get_or_create_collection("category")

        cat_col.add(
            documents=[f"Category: {c}" for c in categories],
            metadatas=[{"category_name": c} for c in categories],
            embeddings=embed(categories),
            ids=[f"cat_{i}" for i in range(len(categories))]
        )
        print(f"✅ Category 写入 {len(categories)} 类")
