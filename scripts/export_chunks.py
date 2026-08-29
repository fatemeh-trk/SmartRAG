import chromadb
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / "database" / "chroma"

client = chromadb.PersistentClient(path=str(db_path))

collection = client.get_collection(name="company_docs_v2")

data = collection.get(
    include=["documents", "metadatas"]
)

chunks = []

for i in range(len(data["ids"])):
    chunks.append({
        "chunk_id": data["ids"][i],
        "chunk_index": data["metadatas"][i]["chunk_index"],
        "file_name": data["metadatas"][i]["file_name"],
        "text": data["documents"][i]
    })

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ {len(chunks)} chunks exported")
print("📄 Output: chunks.json")
