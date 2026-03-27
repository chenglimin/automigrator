from pathlib import Path
import json
import sys
import re
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# ======================================================
# 路径配置
# ======================================================
BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RVV_JSON_PATH = BASE_DIR / "data" / "riscv_intrinsics_combined.json"

# ======================================================
# 输出管理
# ======================================================
class OutputLogger:
    def __init__(self, func_name: str):
        safe_name = re.sub(r"[^\w\-_.]", "_", func_name)
        self.filename = f"{safe_name}.txt"
        self.filepath = RESULTS_DIR / self.filename
        self.fp = open(self.filepath, "w", encoding="utf-8")

        # 终端只提示一次
        print(f"\n📄 输出文件: {self.filename}")
        print(f"📂 相对路径: results/{self.filename}\n")

        self.log("# RVV Query Result")
        self.log(f"# Function: {func_name}")
        self.log(f"# Time: {datetime.now()}")
        self.log("=" * 80)

    def log(self, msg=""):
        self.fp.write(str(msg) + "\n")

    def close(self):
        self.fp.close()

# ======================================================
# Embedding
# ======================================================
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed(text: str):
    return embed_model.encode(
        [text],
        normalize_embeddings=True
    )[0].tolist()

# ======================================================
# 加载 Chroma
# ======================================================
def load_dbs():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return (
        client.get_collection("category"),
        client.get_collection("rvv"),
    )

# ======================================================
# 加载 RVV JSON
# ======================================================
def load_rvv_json():
    if not RVV_JSON_PATH.exists():
        raise FileNotFoundError(f"❌ RVV JSON 不存在: {RVV_JSON_PATH}")
    with open(RVV_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ======================================================
# JSON 精准查询（仅函数名）
# ======================================================
def lookup_rvv_json_exact(func_name: str, rvv_json):
    key = func_name.strip().lower()
    for item in rvv_json:
        if item.get("函数名", "").lower() == key:
            return item
    return None

# ======================================================
# 主查询逻辑
# ======================================================
def query_rvv_system(
    func_name: str,
    rvv_json,
    cat_col,
    rvv_col,
    top_k_categories: int = 3,
    top_k_per_category: int = 3
):
    logger = OutputLogger(func_name)

    try:
        logger.log(f"🔎 RVV 查询: {func_name}")
        logger.log("=" * 70)

        # --------------------------------------------------
        # 1️⃣ JSON 精准查询
        # --------------------------------------------------
        rvv_info = lookup_rvv_json_exact(func_name, rvv_json)

        if not rvv_info:
            logger.log("❌ JSON 中未找到该 RVV intrinsic（仅支持精准函数名）")
            return

        logger.log("✅ 命中 RVV JSON")
        logger.log(f"函数名: {rvv_info.get('函数名')}")
        logger.log(f"Architecture: {rvv_info.get('Architecture')}")
        logger.log(f"Category(JSON): {rvv_info.get('Category')}")

        # --------------------------------------------------
        # 构造 query
        # --------------------------------------------------
        json_categories = rvv_info.get("Category", [])
        if not json_categories:
            logger.log("❌ JSON 中无 Category，无法继续")
            return

        cat_query_text = " ".join(json_categories)

        rvv_query_text = " ".join([
            rvv_info.get("函数名", ""),
            rvv_info.get("Description", ""),
            rvv_info.get("Instruction", ""),
        ])

        # --------------------------------------------------
        # 2️⃣ Category 向量查询
        # --------------------------------------------------
        logger.log(f"\n📡 Category 向量检索（Top-{top_k_categories}）")

        cat_res = cat_col.query(
            query_embeddings=[embed(cat_query_text)],
            n_results=top_k_categories,
            include=["metadatas", "distances"]
        )

        top_categories = []

        for meta, dist in zip(cat_res["metadatas"][0], cat_res["distances"][0]):
            cat_name = meta["category_name"]
            top_categories.append(cat_name)
            logger.log(f"   - {cat_name:<30} (Dist: {dist:.4f})")

        if not top_categories:
            logger.log("❌ 未命中任何 Category")
            return

        # --------------------------------------------------
        # 3️⃣ RVV 检索（分类约束）
        # --------------------------------------------------
        logger.log("\n🚀 RVV 检索（分类约束）")

        all_results = []

        for cat in top_categories:
            rvv_res = rvv_col.query(
                query_embeddings=[embed(rvv_query_text)],
                n_results=top_k_per_category,
                where={"category": cat},
                include=["documents", "metadatas", "distances"]
            )

            docs = rvv_res["documents"][0]
            metas = rvv_res["metadatas"][0]
            dists = rvv_res["distances"][0]

            if not docs:
                continue

            logger.log(f"\n📂 分类: {cat}")

            for doc, meta, dist in zip(docs, metas, dists):
                meta["_score"] = dist
                all_results.append((doc, meta))
                logger.log(f"   - {meta['name']} (Dist: {dist:.4f})")

        if not all_results:
            logger.log("❌ RVV 库无命中")
            return

        # --------------------------------------------------
        # 4️⃣ 全局排序 + 去重
        # --------------------------------------------------
        logger.log("\n✅ 全局排序 + 去重")

        all_results.sort(key=lambda x: x[1]["_score"])

        final = []
        seen = set()

        for doc, meta in all_results:
            name = meta.get("name")
            if name in seen:
                continue
            seen.add(name)
            final.append((doc, meta))

        # --------------------------------------------------
        # 输出最终结果
        # --------------------------------------------------
        for i, (doc, meta) in enumerate(final):
            logger.log(f"\n--- 推荐 {i+1} ---")
            logger.log(f"Name: {meta['name']}")
            logger.log(f"Category: {meta['category']}")
            logger.log(f"Dist: {meta['_score']:.4f}")

            if "Operation:" in doc:
                logger.log("Logic:")
                logger.log(doc.split("Operation:")[-1].strip())
            else:
                logger.log("Info:")
                logger.log(doc[:800])

    finally:
        logger.close()

# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    rvv_json = load_rvv_json()
    cat_col, rvv_col = load_dbs()

    if len(sys.argv) > 1:
        query_rvv_system(sys.argv[1], rvv_json, cat_col, rvv_col)
        sys.exit(0)

    while True:
        q = input("\n输入 RVV 函数名（q 退出）: ").strip()
        if q == "q":
            break
        if q:
            query_rvv_system(q, rvv_json, cat_col, rvv_col)
