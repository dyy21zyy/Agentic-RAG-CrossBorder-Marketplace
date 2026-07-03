from pathlib import Path

from crossborder_agentic_rag.evaluation.datasets import load_eval_jsonl


GOLDEN_QUERIES_PATH = Path("data/eval/golden_queries_30.jsonl")
ALLOWED_SOURCE_TYPES = {"trademark", "patent", "litigation"}
ALLOWED_PARTITIONS = {"trademark_db", "patent_db", "litigation_db"}
ALLOWED_TOOLS = {
    "trademark_search_tool",
    "patent_search_tool",
    "litigation_search_tool",
    "duckdb_lookup_tool",
    "graph_rag_tool",
}
FORBIDDEN_TERMS = {
    "policy",
    "policy_search_tool",
    "date_calculation_tool",
    "expiration_tool",
    "memory_tool",
}


def test_golden_queries_30_template_loads_and_matches_schema() -> None:
    assert GOLDEN_QUERIES_PATH.exists()

    examples = load_eval_jsonl(GOLDEN_QUERIES_PATH)
    assert len(examples) == 30
    assert [ex.query_id for ex in examples] == [f"Q{i:03d}" for i in range(1, 31)]

    for ex in examples:
        assert set(ex.expected_source_types) <= ALLOWED_SOURCE_TYPES
        assert set(ex.expected_partitions) <= ALLOWED_PARTITIONS
        assert set(ex.expected_tools) <= ALLOWED_TOOLS
        assert ex.relevant_doc_ids == []
        assert ex.relevant_chunk_ids == []
        assert ex.relevance_grades == {}

        serialized = str(ex).lower()
        for forbidden_term in FORBIDDEN_TERMS:
            assert forbidden_term not in serialized
