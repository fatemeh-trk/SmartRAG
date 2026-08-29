import time
import re
import ast
from pathlib import Path
import sys
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "source"
sys.path.append(str(SRC_DIR))

from models.embeddings import load_embedder
from models.reranker import load_reranker

from storage.chroma import data_base

from retrieval.search import (
    search,
    process_retrieval_results
)

from retrieval.reranker import rerank

INPUT_CSV = BASE_DIR / "retrieval_evaluation_updated.csv"
OUTPUT_XLSX = BASE_DIR / "retrieval_evaluation_results.xlsx"

SEARCH_TOP_K = 20
FINAL_TOP_K = 7
RETRIEVAL_THRESHOLD = 21


# ============================================================
# EXPECTED CHUNKS PARSER
# ============================================================

def parse_expected_chunks(value):
    """
    Supports formats such as:

        12
        12,13,14
        12-15
        [12, 13, 14]
        "12,13,14"
        "12-15,20"
        NaN / empty

    Returns a set of integer chunk indexes.
    """

    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    # Try Python-list format
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            result = set()

            for item in parsed:
                if isinstance(item, int):
                    result.add(item)

                elif isinstance(item, float) and item.is_integer():
                    result.add(int(item))

                elif isinstance(item, str):
                    result.update(
                        parse_expected_chunks(item)
                    )

            return result

        if isinstance(parsed, int):
            return {parsed}

        if isinstance(parsed, float) and parsed.is_integer():
            return {int(parsed)}

    except Exception:
        pass

    # Normalize separators
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace(";", ",")
    text = text.replace("/", ",")
    
    result = set()

    # Find ranges and individual numbers
    tokens = re.split(r"[,\s]+", text)

    for token in tokens:

        token = token.strip()

        if not token:
            continue

        # Range: 12-16
        range_match = re.fullmatch(
            r"(\d+)\s*-\s*(\d+)",
            token
        )

        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))

            if start <= end:
                result.update(range(start, end + 1))
            else:
                result.update(range(end, start + 1))

            continue

        # Single integer
        number_match = re.fullmatch(
            r"\d+",
            token
        )

        if number_match:
            result.add(int(token))

    return result


# ============================================================
# RETRIEVED RESULTS
# ============================================================

def get_retrieved_items(results, reranker_scores=None):
    """
    Converts final Chroma/reranker result into a clean list.

    Rank is the FINAL rank after reranking.
    """

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    items = []

    for i in range(len(documents)):

        metadata = metadatas[i] if i < len(metadatas) else {}

        chunk_index = metadata.get("chunk_index")

        try:
            chunk_index = int(chunk_index)
        except Exception:
            pass

        item = {
            "rank": i + 1,
            "chunk_id": ids[i] if i < len(ids) else None,
            "chunk_index": chunk_index,
            "distance": (
                distances[i]
                if i < len(distances)
                else None
            ),
            "file_name": metadata.get(
                "file_name",
                ""
            ),
            "text": documents[i],
        }

        if reranker_scores is not None and i < len(reranker_scores):
            item["reranker_score"] = float(
                reranker_scores[i]
            )
        else:
            item["reranker_score"] = None

        items.append(item)

    return items


# ============================================================
# METRICS
# ============================================================

def hit_at_k(retrieved, expected, k):

    if not expected:
        return None

    top_k = retrieved[:k]

    return int(
        any(
            item["chunk_index"] in expected
            for item in top_k
        )
    )


def precision_at_k(retrieved, expected, k):

    if not expected:
        return None

    top_k = retrieved[:k]

    if not top_k:
        return 0.0

    relevant = sum(
        item["chunk_index"] in expected
        for item in top_k
    )

    return relevant / k


def recall_at_k(retrieved, expected, k):

    if not expected:
        return None

    if not expected:
        return 0.0

    top_k = retrieved[:k]

    relevant = sum(
        item["chunk_index"] in expected
        for item in top_k
    )

    return relevant / len(expected)


def reciprocal_rank(retrieved, expected, k):

    if not expected:
        return None

    for item in retrieved[:k]:

        if item["chunk_index"] in expected:
            return 1.0 / item["rank"]

    return 0.0


def dcg_at_k(retrieved, expected, k):

    score = 0.0

    for rank, item in enumerate(
        retrieved[:k],
        start=1
    ):

        relevance = int(
            item["chunk_index"] in expected
        )

        score += (
            relevance /
            __import__("math").log2(rank + 1)
        )

    return score


def ndcg_at_k(retrieved, expected, k):

    if not expected:
        return None

    actual_dcg = dcg_at_k(
        retrieved,
        expected,
        k
    )

    ideal_retrieved = [
        {
            "chunk_index": chunk
        }
        for chunk in list(expected)[:k]
    ]

    ideal_dcg = 0.0

    for rank, _ in enumerate(
        ideal_retrieved,
        start=1
    ):

        ideal_dcg += (
            1.0 /
            __import__("math").log2(rank + 1)
        )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ============================================================
# SINGLE QUESTION EVALUATION
# ============================================================

def evaluate_question(
    row,
    embedder,
    reranker,
    collection,
):

    question = str(
        row["Question"]
    ).strip()

    expected = parse_expected_chunks(
        row.get("Expected_chunks", "")
    )

    # --------------------------------------------------------
    # TOTAL RETRIEVAL LATENCY
    # --------------------------------------------------------

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # 1. EMBEDDING + VECTOR SEARCH
    # --------------------------------------------------------

    search_start = time.perf_counter()

    search_results = search(
        question,
        embedder,
        collection,
        top_k=SEARCH_TOP_K
    )

    search_latency = (
        time.perf_counter() - search_start
    ) * 1000

    # --------------------------------------------------------
    # BEST DISTANCE BEFORE THRESHOLD
    # --------------------------------------------------------

    original_distances = (
        search_results.get(
            "distances",
            [[]]
        )[0]
    )

    best_distance = (
        min(original_distances)
        if original_distances
        else None
    )

    initial_retrieved_count = len(
        search_results.get(
            "ids",
            [[]]
        )[0]
    )

    # --------------------------------------------------------
    # 2. THRESHOLD
    # --------------------------------------------------------

    threshold_start = time.perf_counter()

    filtered_results = process_retrieval_results(
        search_results,
        RETRIEVAL_THRESHOLD
    )

    threshold_latency = (
        time.perf_counter() - threshold_start
    ) * 1000

    threshold_count = len(
        filtered_results.get(
            "ids",
            [[]]
        )[0]
    )

    # --------------------------------------------------------
    # 3. RERANK
    # --------------------------------------------------------

    rerank_start = time.perf_counter()

    final_results = rerank(
        question,
        filtered_results,
        reranker,
        top_n=FINAL_TOP_K
    )

    rerank_latency = (
        time.perf_counter() - rerank_start
    ) * 1000

    total_latency = (
        time.perf_counter() - total_start
    ) * 1000

    # --------------------------------------------------------
    # FINAL RETRIEVED ITEMS
    # --------------------------------------------------------

    retrieved = get_retrieved_items(
        final_results
    )

    retrieved_indices = [
        item["chunk_index"]
        for item in retrieved
    ]

    # --------------------------------------------------------
    # RELEVANT RANKS
    # --------------------------------------------------------

    relevant_ranks = [
        item["rank"]
        for item in retrieved
        if item["chunk_index"] in expected
    ]

    # --------------------------------------------------------
    # QUERY TYPE
    # --------------------------------------------------------

    if expected:
        query_type = "In-scope"
    else:
        query_type = "Negative/Out-of-scope"

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    if expected:

        top1_hit = hit_at_k(
            retrieved,
            expected,
            1
        )

        top3_hit = hit_at_k(
            retrieved,
            expected,
            3
        )

        top5_hit = hit_at_k(
            retrieved,
            expected,
            5
        )

        precision7 = precision_at_k(
            retrieved,
            expected,
            7
        )

        recall7 = recall_at_k(
            retrieved,
            expected,
            7
        )

        mrr7 = reciprocal_rank(
            retrieved,
            expected,
            7
        )

        ndcg7 = ndcg_at_k(
            retrieved,
            expected,
            7
        )

        negative_no_retrieval = None

    else:

        top1_hit = None
        top3_hit = None
        top5_hit = None

        precision7 = None
        recall7 = None
        mrr7 = None
        ndcg7 = None

        negative_no_retrieval = int(
            len(retrieved) == 0
        )

    # --------------------------------------------------------
    # MAIN RESULT
    # --------------------------------------------------------

    result = {

        "ID":
            row.get("ID", ""),

        "Type":
            row.get("Type", ""),

        "Question":
            question,

        "Expected_chunks":
            row.get(
                "Expected_chunks",
                ""
            ),

        "Expected_chunk_count":
            len(expected),

        "Retrieved_Chunks":
            ",".join(
                str(x)
                for x in retrieved_indices
            ),

        "Retrieved_Chunk_Count":
            len(retrieved),

        "Relevant_Ranks":
            ",".join(
                str(x)
                for x in relevant_ranks
            ),

        "Initial_Search_TopK":
            SEARCH_TOP_K,

        "Initial_Retrieved_Count":
            initial_retrieved_count,

        "Threshold":
            RETRIEVAL_THRESHOLD,

        "After_Threshold_Count":
            threshold_count,

        "Final_TopK":
            FINAL_TOP_K,

        "Best_Distance":
            best_distance,

        "Top1_Hit":
            top1_hit,

        "Top3_Hit":
            top3_hit,

        "Top5_Hit":
            top5_hit,

        "Precision@7":
            precision7,

        "Recall@7":
            recall7,

        "MRR@7":
            mrr7,

        "NDCG@7":
            ndcg7,

        "Negative_No_Retrieval":
            negative_no_retrieval,

        "Query_Type":
            query_type,

        "Search_Latency_ms":
            search_latency,

        "Threshold_Latency_ms":
            threshold_latency,

        "Rerank_Latency_ms":
            rerank_latency,

        "Total_Retrieval_Latency_ms":
            total_latency,

        "Status":
            "OK",
    }

    return result, retrieved


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"\nInput CSV:")
    print(INPUT_CSV)

    print(f"\nOutput Excel:")
    print(OUTPUT_XLSX)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_CSV}"
        )

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"\nLoaded {len(df)} evaluation questions."
    )

    required_columns = [
        "Question",
        "Expected_chunks"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedder = load_embedder()

    print(
        "Embedding model loaded."
    )

    print(
        "\nLoading reranker..."
    )

    reranker = load_reranker()

    print(
        "Reranker loaded."
    )

    # --------------------------------------------------------
    # LOAD CHROMA
    # --------------------------------------------------------

    print(
        "\nLoading Chroma database..."
    )

    collection = data_base()

    collection_count = collection.count()

    print(
        f"Chroma collection contains "
        f"{collection_count} chunks."
    )

    if collection_count == 0:

        raise RuntimeError(
            "\nChroma database is empty. "
            "Please add the document before running evaluation."
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    results_rows = []
    details_rows = []

    total_questions = len(df)

    print("\n")
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        question = str(
            row["Question"]
        ).strip()

        print(
            f"\n[{position}/{total_questions}] "
            f"{question}"
        )

        try:

            result, retrieved = evaluate_question(
                row,
                embedder,
                reranker,
                collection
            )

            results_rows.append(
                result
            )

            expected = parse_expected_chunks(
                row["Expected_chunks"]
            )

            # ----------------------------------------------
            # DETAILED TOP-7 RESULTS
            # ----------------------------------------------

            for item in retrieved:

                details_rows.append({

                    "ID":
                        row.get("ID", ""),

                    "Question":
                        question,

                    "Rank":
                        item["rank"],

                    "Chunk_ID":
                        item["chunk_id"],

                    "Chunk_Index":
                        item["chunk_index"],

                    "Distance":
                        item["distance"],

                    "Reranker_Score":
                        item["reranker_score"],

                    "Is_Expected":
                        int(
                            item["chunk_index"]
                            in expected
                        ),

                    "File_Name":
                        item["file_name"],

                    "Text":
                        item["text"],
                })

            print(
                "Expected:",
                sorted(expected)
            )

            print(
                "Retrieved:",
                [
                    item["chunk_index"]
                    for item in retrieved
                ]
            )

            if result["MRR@7"] is not None:

                print(
                    f"MRR@7: "
                    f"{result['MRR@7']:.4f} | "
                    f"NDCG@7: "
                    f"{result['NDCG@7']:.4f}"
                )

            print(
                f"Latency: "
                f"{result['Total_Retrieval_Latency_ms']:.2f} ms"
            )

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

            results_rows.append({

                "ID":
                    row.get("ID", ""),

                "Type":
                    row.get("Type", ""),

                "Question":
                    question,

                "Expected_chunks":
                    row.get(
                        "Expected_chunks",
                        ""
                    ),

                "Status":
                    f"ERROR: {e}",
            })

    # --------------------------------------------------------
    # DATAFRAMES
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results_rows
    )

    details_df = pd.DataFrame(
        details_rows
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    successful = results_df[
        results_df["Status"] == "OK"
    ]

    in_scope = successful[
        successful["Query_Type"] == "In-scope"
    ]

    negative = successful[
        successful["Query_Type"]
        == "Negative/Out-of-scope"
    ]

    summary_rows = [

        {
            "Metric":
                "Total Questions",

            "Value":
                len(df)
        },

        {
            "Metric":
                "Successful Questions",

            "Value":
                len(successful)
        },

        {
            "Metric":
                "In-scope Questions",

            "Value":
                len(in_scope)
        },

        {
            "Metric":
                "Negative/Out-of-scope Questions",

            "Value":
                len(negative)
        },

        {
            "Metric":
                "Hit@1",

            "Value":
                (
                    in_scope["Top1_Hit"].mean()
                    if len(in_scope)
                    else None
                )
        },

        {
            "Metric":
                "Hit@3",

            "Value":
                (
                    in_scope["Top3_Hit"].mean()
                    if len(in_scope)
                    else None
                )
        },

        {
            "Metric":
                "Hit@5",

            "Value":
                (
                    in_scope["Top5_Hit"].mean()
                    if len(in_scope)
                    else None
                )
        },

        {
            "Metric":
                "Precision@7",

            "Value":
                (
                    in_scope["Precision@7"].mean()
                    if len(in_scope)
                    else None
                )
        },

        {
            "Metric":
                "Recall@7",

            "Value":
                (
                    in_scope["Recall@7"].mean()
                    if len(in_scope)
                    else None
                )
        },

        {
            "Metric":
                "MRR@7",

            "Value":
                (
                    in_scope["MRR@7"].mean()
                    if len(in_scope)
                    else None
                )
        },

        {
            "Metric":
                "NDCG@7",

            "Value":
                (
                    in_scope["NDCG@7"].mean()
                    if len(in_scope)
                    else None
                )
        },

        {
            "Metric":
                "Negative No-Retrieval Rate",

            "Value":
                (
                    negative[
                        "Negative_No_Retrieval"
                    ].mean()
                    if len(negative)
                    else None
                )
        },

        {
            "Metric":
                "Average Search Latency (ms)",

            "Value":
                (
                    successful[
                        "Search_Latency_ms"
                    ].mean()
                    if len(successful)
                    else None
                )
        },

        {
            "Metric":
                "Average Rerank Latency (ms)",

            "Value":
                (
                    successful[
                        "Rerank_Latency_ms"
                    ].mean()
                    if len(successful)
                    else None
                )
        },

        {
            "Metric":
                "Average Total Retrieval Latency (ms)",

            "Value":
                (
                    successful[
                        "Total_Retrieval_Latency_ms"
                    ].mean()
                    if len(successful)
                    else None
                )
        },
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # SAVE EXCEL
    # --------------------------------------------------------

    print(
        "\nSaving results..."
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        results_df.to_excel(
            writer,
            sheet_name="Question_Results",
            index=False
        )

        details_df.to_excel(
            writer,
            sheet_name="Retrieved_Details",
            index=False
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n{OUTPUT_XLSX}"
    )

    if len(in_scope):

        print("\nOverall In-scope Metrics:")

        print(
            f"Hit@1     : "
            f"{in_scope['Top1_Hit'].mean():.4f}"
        )

        print(
            f"Hit@3     : "
            f"{in_scope['Top3_Hit'].mean():.4f}"
        )

        print(
            f"Hit@5     : "
            f"{in_scope['Top5_Hit'].mean():.4f}"
        )

        print(
            f"Precision@7: "
            f"{in_scope['Precision@7'].mean():.4f}"
        )

        print(
            f"Recall@7  : "
            f"{in_scope['Recall@7'].mean():.4f}"
        )

        print(
            f"MRR@7     : "
            f"{in_scope['MRR@7'].mean():.4f}"
        )

        print(
            f"NDCG@7   : "
            f"{in_scope['NDCG@7'].mean():.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
