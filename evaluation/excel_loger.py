import json
import os

import pandas as pd
import chromadb
from sentence_transformers import CrossEncoder

CHROMA_PATH=r"D:\TechHive\SRC\database\chroma"
# ============================================================
# CONFIG
# ============================================================

CHUNKS_JSON = "chunks.json"
QUESTIONS_CSV = "retrieval_evaluation_updated.csv"

OUTPUT_XLSX = "reranker_analysis.xlsx"

COLLECTION_NAME = "company_docs_v2"

INITIAL_TOP_K = 20

RERANK_TOP_N = 7


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# ============================================================
# LOAD RERANKER
# ============================================================

def load_reranker():
    print("Loading CrossEncoder...")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )

    print("CrossEncoder loaded.")

    return reranker


# ============================================================
# CONNECT TO CHROMA
# ============================================================

def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    print("\n========== CHROMA DEBUG ==========")
    print("Chroma path:", CHROMA_PATH)

    collections = client.list_collections()

    print("Available collections:")
    for c in collections:
        print(" -", c.name)

    print("==================================\n")

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# ============================================================
# EXPECTED CHUNK PARSER
# ============================================================

def parse_expected_chunks(value):
    """
    Converts values such as:

        493-496
        441-452,463-468
        21-25,29-40

    into a set of integer chunk indexes.
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value or value.lower() == "not found":
        return set()

    chunks = set()

    parts = value.split(",")

    for part in parts:

        part = part.strip()

        if "-" in part:

            start, end = part.split("-", 1)

            try:
                start = int(start.strip())
                end = int(end.strip())

                for i in range(start, end + 1):
                    chunks.add(i)

            except ValueError:
                continue

        else:

            try:
                chunks.add(int(part))

            except ValueError:
                continue

    return chunks


# ============================================================
# FIND CHUNK INDEX
# ============================================================

def get_chunk_index(metadata):
    value = metadata.get("chunk_index")

    try:
        return int(value)
    except (TypeError, ValueError):
        return value


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("\n========================================")
    print("RERANKER SCORE ANALYSIS")
    print("========================================\n")

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    if not os.path.exists(QUESTIONS_CSV):
        raise FileNotFoundError(
            f"CSV not found: {QUESTIONS_CSV}"
        )

    questions_df = pd.read_csv(
        QUESTIONS_CSV
    )

    print(
        f"Loaded {len(questions_df)} questions."
    )

    # --------------------------------------------------------
    # Load reranker
    # --------------------------------------------------------

    reranker = load_reranker()

    # --------------------------------------------------------
    # Load Chroma
    # --------------------------------------------------------

    collection = load_collection()

    print(
        f"Collection loaded: {COLLECTION_NAME}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    score_rows = []

    # --------------------------------------------------------
    # Process every question
    # --------------------------------------------------------

    for row_number, row in questions_df.iterrows():

        question_id = str(row["ID"])
        question = str(row["Question"])

        expected_chunks = parse_expected_chunks(
            row["Expected_chunks"]
        )

        print("\n----------------------------------------")
        print(f"ID: {question_id}")
        print(f"Question: {question}")
        print(f"Expected: {sorted(expected_chunks)}")

        # ----------------------------------------------------
        # Negative question
        # ----------------------------------------------------

        if str(row["Type"]).lower() == "negative":

            print("Negative question -> skipped.")

            continue

        # ----------------------------------------------------
        # Initial vector retrieval
        # ----------------------------------------------------

        results = collection.query(
            query_texts=[question],
            n_results=INITIAL_TOP_K
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        ids = results.get(
            "ids",
            [[]]
        )[0]

        if not documents:
            print("No documents retrieved.")

            continue

        # ----------------------------------------------------
        # CrossEncoder
        # ----------------------------------------------------

        pairs = [
            (question, document)
            for document in documents
        ]

        scores = reranker.predict(
            pairs
        )

        # ----------------------------------------------------
        # Sort by reranker score
        # ----------------------------------------------------

        ranked = sorted(
            zip(
                scores,
                range(len(documents))
            ),
            key=lambda x: float(x[0]),
            reverse=True
        )

        # ----------------------------------------------------
        # Save ALL reranker candidates
        # ----------------------------------------------------

        for rank, (score, index) in enumerate(
            ranked,
            start=1
        ):

            metadata = metadatas[index]

            chunk_index = get_chunk_index(
                metadata
            )

            is_expected = (
                chunk_index in expected_chunks
            )

            score_rows.append({

                "ID":
                    question_id,

                "Question":
                    question,

                "Initial_Rank":
                    rank,

                "Chunk":
                    chunk_index,

                "Reranker_Score":
                    float(score),

                "Expected":
                    is_expected,

                "Distance":
                    distances[index],

                "Chroma_ID":
                    ids[index],

                "File":
                    metadata.get(
                        "file_name",
                        ""
                    )

            })

        # ----------------------------------------------------
        # Debug output
        # ----------------------------------------------------

        print("\nRERANKER RESULTS:")

        for rank, (score, index) in enumerate(
            ranked,
            start=1
        ):

            metadata = metadatas[index]

            chunk_index = get_chunk_index(
                metadata
            )

            expected = (
                "YES"
                if chunk_index in expected_chunks
                else "NO"
            )

            print(
                f"{rank:2d} | "
                f"Chunk {str(chunk_index):>5} | "
                f"Score {float(score):8.4f} | "
                f"Expected: {expected}"
            )

        # ----------------------------------------------------
        # Final Top 7
        # ----------------------------------------------------

        print("\nFINAL TOP 7:")

        for rank, (score, index) in enumerate(
            ranked[:RERANK_TOP_N],
            start=1
        ):

            metadata = metadatas[index]

            chunk_index = get_chunk_index(
                metadata
            )

            expected = (
                "YES"
                if chunk_index in expected_chunks
                else "NO"
            )

            print(
                f"{rank:2d} | "
                f"Chunk {str(chunk_index):>5} | "
                f"Score {float(score):8.4f} | "
                f"Expected: {expected}"
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    scores_df = pd.DataFrame(
        score_rows
    )

    if scores_df.empty:

        print(
            "\nNo reranker scores generated."
        )

        return

    # --------------------------------------------------------
    # Additional analysis
    # --------------------------------------------------------

    summary_rows = []

    for question_id, group in scores_df.groupby(
        "ID"
    ):

        relevant = group[
            group["Expected"] == True
        ]

        irrelevant = group[
            group["Expected"] == False
        ]

        if not relevant.empty:
            max_relevant = relevant[
                "Reranker_Score"
            ].max()

            min_relevant = relevant[
                "Reranker_Score"
            ].min()
        else:
            max_relevant = None
            min_relevant = None

        if not irrelevant.empty:
            max_irrelevant = irrelevant[
                "Reranker_Score"
            ].max()

            min_irrelevant = irrelevant[
                "Reranker_Score"
            ].min()
        else:
            max_irrelevant = None
            min_irrelevant = None

        if (
            min_relevant is not None
            and max_irrelevant is not None
        ):
            score_gap = (
                min_relevant
                - max_irrelevant
            )
        else:
            score_gap = None

        top7 = group[
            group["Initial_Rank"] <= RERANK_TOP_N
        ]

        top7_relevant_count = int(
            top7["Expected"].sum()
        )

        summary_rows.append({

            "ID":
                question_id,

            "Max_Relevant_Score":
                max_relevant,

            "Min_Relevant_Score":
                min_relevant,

            "Max_Irrelevant_Score":
                max_irrelevant,

            "Min_Irrelevant_Score":
                min_irrelevant,

            "Relevant_vs_Irrelevant_Gap":
                score_gap,

            "Relevant_in_Top7":
                top7_relevant_count,

            "Total_Expected":
                int(
                    group["Expected"].sum()
                )

        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # SAVE EXCEL
    # ========================================================

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl"
    ) as writer:

        scores_df.to_excel(
            writer,
            sheet_name="Reranker_Scores",
            index=False
        )

        summary_df.to_excel(
            writer,
            sheet_name="Score_Summary",
            index=False
        )

    print("\n========================================")
    print("DONE")
    print("========================================")

    print(
        f"\nOutput: {OUTPUT_XLSX}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
