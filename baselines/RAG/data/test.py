import json
import os
from collections import Counter

def inspect_categories(filepath):
    # 1. 自动寻找文件 (如果不在当前目录)
    if not os.path.exists(filepath):
        possible_paths = [
            f"data/{filepath}", 
            f"../{filepath}",
            "/openhands/code/data/riscv_intrinsics_combined.json",
            "/root/RAG_AVX_to_RVV/RAG/data/riscv_intrinsics_combined.json"
        ]
        found = False
        for p in possible_paths:
            if os.path.exists(p):
                filepath = p
                found = True
                break
        if not found:
            print(f"❌ 错误: 在当前目录及常见路径下未找到 {filepath}")
            return

    print(f"正在读取文件: {filepath} ...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 兼容数据结构 (List 或 Dict)
    items = data if isinstance(data, list) else data.get("intrinsics", [])
    print(f"✅ 成功加载 {len(items)} 条指令数据。正在分析 Category...\n")

    # 2. 统计 Category
    # 我们不仅统计有哪些，还统计每个 Category 下有多少条指令
    cat_counter = Counter()
    example_map = {} # 存一个例子，方便看长什么样

    for item in items:
        # 获取 Category 字段，通常是列表 ["Memory|Load"]
        cats = item.get('Category', [])
        
        # 如果是 None 或空
        if not cats:
            cat_counter["[No Category]"] += 1
            continue

        for cat in cats:
            cat_counter[cat] += 1
            if cat not in example_map:
                example_map[cat] = item.get('函数名', 'unknown')

    # 3. 打印结果
    print(f"{'='*15} Category 分类汇总 ({len(cat_counter)} 种) {'='*15}")
    print(f"{'Count':<8} | {'Category Name':<40} | {'Example Intrinsic'}")
    print("-" * 80)

    # 按名称排序输出
    for cat, count in sorted(cat_counter.items()):
        example = example_map.get(cat, "")
        print(f"{count:<8} | {cat:<40} | {example}")

if __name__ == "__main__":
    inspect_categories("riscv_intrinsics_combined.json")