import re
import sys
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET
from datetime import datetime

# ======================================================
# 路径配置
# ======================================================
BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INTEL_XML_PATH = BASE_DIR / "data" / "intel_intrinsics-2.xml"

# ======================================================
# 输出管理（核心新增）
# ======================================================
class OutputLogger:
    def __init__(self, intrinsic: str):
        safe_name = re.sub(r"[^\w\-_.]", "_", intrinsic)
        self.filename = f"{safe_name}.txt"
        self.filepath = RESULTS_DIR / self.filename
        self.fp = open(self.filepath, "w", encoding="utf-8")

        # 终端只打印一次
        print(f"\n📄 输出文件: {self.filename}")
        print(f"📂 相对路径: results/{self.filename}\n")

        self.log(f"# RAG Query Result")
        self.log(f"# Intrinsic: {intrinsic}")
        self.log(f"# Time: {datetime.now()}")
        self.log("=" * 80)

    def log(self, msg=""):
        self.fp.write(str(msg) + "\n")

    def close(self):
        self.fp.close()

# ======================================================
# XML 查询
# ======================================================
def lookup_x86_from_xml(intrinsic_name: str):
    if not INTEL_XML_PATH.exists():
        return None

    key = intrinsic_name.strip().lower()

    try:
        tree = ET.parse(INTEL_XML_PATH)
        root = tree.getroot()
    except Exception:
        return None

    for intr in root.iter("intrinsic"):
        name = intr.get("name")
        if name and name.lower() == key:
            desc = intr.findtext("description", default="").strip()
            return {"name": name, "raw_desc": desc}

    return None

# ======================================================
# Embedding
# ======================================================
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed(text: str):
    return embed_model.encode([text], normalize_embeddings=True)[0].tolist()

# ======================================================
# 规则引擎
# ======================================================
class InstructionMapper:
    def __init__(self):
        self.direct_rules = {
            "add": ["add", "sum", "accumulate", "count"],
            "sub": ["sub", "diff", "minus"],
            "mul": ["mul", "product", "square"],
            "div": ["div", "rem"],
            "min": ["min"],
            "max": ["max"],
            "abs": ["abs"],
            "neg": ["neg", "sub"],
            "load": ["load", "le", "gather"],
            "store": ["store", "se", "scatter"],
            "cmp": ["compare", "seq", "sgt", "sge", "slt", "sle", "ne", "eq"],
            "and": ["and"],
            "xor": ["xor"],
            "or":  ["or"],
            "not": ["not"],
            "sll": ["shift", "sll"],
            "srl": ["shift", "srl"],
            "sra": ["shift", "sra"],
            "cvt": ["conversion", "cvt", "widen", "narrow"],
        }

        self.complex_rules = {
            "unpack": ("slide gather interleave", ["permutation", "slide", "gather", "merge"]),
            "shuffle": ("gather permutation index", ["permutation", "gather", "shuffle", "compress"]),
            "alignr": ("slideup slidedown extract", ["permutation", "slide"]),
            "hadd": ("slide reduction fold", ["fold", "reduction", "slide", "sum"]),
            "avg": ("add shift averaging", ["fixed-point", "add", "shift"]),
            "blend": ("merge mask", ["permutation", "merge", "mask"]),
            "gather": ("indexed load", ["memory", "indexed", "gather"]),
        }

    def analyze(self, x86_name):
        name_lower = x86_name.lower()

        for key, (boost, keywords) in self.complex_rules.items():
            if key in name_lower:
                return "COMPLEX", boost, keywords

        for key in sorted(self.direct_rules, key=len, reverse=True):
            if key in name_lower:
                return "DIRECT", None, self.direct_rules[key]

        return "UNKNOWN", None, []

# ======================================================
# 加载 Chroma
# ======================================================
def load_dbs():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return (
        client.get_collection("x86"),
        client.get_collection("rvv"),
        client.get_collection("category"),
    )

# ======================================================
# 主查询逻辑（核心修改）
# ======================================================
def query_rag_system(instruction, x86_col, rvv_col, cat_col, mapper):
    logger = OutputLogger(instruction)

    try:
        logger.log(f"🔎 查询指令: {instruction}")
        logger.log("=" * 60)

        xml_hit = lookup_x86_from_xml(instruction)

        if xml_hit:
            x86_name = xml_hit["name"]
            x86_desc = xml_hit["raw_desc"]
            logger.log("✅ 命中 Intel XML（精确）")
        else:
            logger.log("⚠️ XML 未命中，回退到 Chroma")
            x86_res = x86_col.query(
                query_embeddings=[embed(instruction)],
                n_results=5
            )
            meta = x86_res["metadatas"][0][0]
            x86_name = meta.get("name", instruction)
            x86_desc = meta.get("raw_desc", "")

        logger.log(f"[x86] {x86_name}")
        logger.log(x86_desc)

        strategy, boost, keywords = mapper.analyze(x86_name)
        clean_name = x86_name.replace("_", " ")

        query_text = f"{clean_name} {boost or ''} {x86_desc}"

        cat_res = cat_col.query(
            query_embeddings=[embed(query_text)],
            n_results=4,
            include=["metadatas", "distances"]
        )

        categories = [m["category_name"] for m in cat_res["metadatas"][0]]
        logger.log(f"\n🎯 分类: {categories}")

        all_hits = []

        for cat in categories:
            rvv_res = rvv_col.query(
                query_embeddings=[embed(query_text)],
                n_results=3,
                where={"category": cat},
                include=["documents", "metadatas", "distances"]
            )
            for doc, meta, dist in zip(
                rvv_res["documents"][0],
                rvv_res["metadatas"][0],
                rvv_res["distances"][0]
            ):
                meta["_score"] = dist
                all_hits.append((doc, meta))

        all_hits.sort(key=lambda x: x[1]["_score"])

        logger.log("\n✅ 最终推荐:")
        for i, (doc, meta) in enumerate(all_hits):
            logger.log(f"\n--- #{i+1} ---")
            logger.log(f"Name: {meta['name']}")
            logger.log(f"Category: {meta['category']}")
            logger.log(doc[:1500])

    finally:
        logger.close()

# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    x86_col, rvv_col, cat_col = load_dbs()
    mapper = InstructionMapper()

    if len(sys.argv) > 1:
        query_rag_system(sys.argv[1].strip(), x86_col, rvv_col, cat_col, mapper)
    else:
        while True:
            q = input("输入 intrinsic (q退出): ").strip()
            if q == "q":
                break
            if q:
                query_rag_system(q, x86_col, rvv_col, cat_col, mapper)
