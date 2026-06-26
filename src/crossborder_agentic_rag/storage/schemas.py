"""DuckDB schema definitions for structured exact lookup."""

from __future__ import annotations

REQUIRED_TABLES = {
    "trademarks",
    "trademark_classes",
    "trademark_goods_services",
    "patents",
    "litigation_cases",
    "litigation_documents",
    "litigation_names",
    "litigation_patents",
    "load_audit",
}

ROW_COUNT_TABLES = [
    "trademarks",
    "trademark_classes",
    "trademark_goods_services",
    "patents",
    "litigation_cases",
    "litigation_documents",
    "litigation_names",
    "litigation_patents",
    "load_audit",
]

CREATE_TABLE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS trademarks (
        doc_id TEXT PRIMARY KEY,
        serial_number TEXT,
        registration_number TEXT,
        word_mark TEXT,
        filing_date TEXT,
        registration_date TEXT,
        status_code TEXT,
        source_file TEXT,
        source_path TEXT,
        metadata_json TEXT,
        content TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trademark_classes (
        doc_id TEXT,
        serial_number TEXT,
        registration_number TEXT,
        word_mark TEXT,
        nice_class TEXT,
        class_description TEXT,
        source_file TEXT,
        source_path TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trademark_goods_services (
        doc_id TEXT,
        serial_number TEXT,
        registration_number TEXT,
        word_mark TEXT,
        nice_class TEXT,
        goods_services TEXT,
        source_file TEXT,
        source_path TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS patents (
        doc_id TEXT PRIMARY KEY,
        patent_id TEXT,
        patent_number TEXT,
        brief_summary TEXT,
        claims TEXT,
        detail_description TEXT,
        drawing_description TEXT,
        source_file TEXT,
        source_path TEXT,
        metadata_json TEXT,
        content TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS litigation_cases (
        doc_id TEXT PRIMARY KEY,
        case_row_id TEXT,
        case_number TEXT,
        district_id TEXT,
        court_name TEXT,
        case_name TEXT,
        case_cause TEXT,
        jurisdictional_basis TEXT,
        date_filed TEXT,
        date_closed TEXT,
        date_last_filed TEXT,
        source_files_json TEXT,
        metadata_json TEXT,
        content TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS litigation_documents (
        doc_id TEXT,
        case_row_id TEXT,
        case_number TEXT,
        district_id TEXT,
        doc_number TEXT,
        short_description TEXT,
        long_description TEXT,
        doc_date_filed TEXT,
        doc_date_uploaded TEXT,
        document_url TEXT,
        source_file TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS litigation_names (
        doc_id TEXT,
        case_row_id TEXT,
        case_number TEXT,
        district_id TEXT,
        party_type TEXT,
        name TEXT,
        name_long TEXT,
        source_file TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS litigation_patents (
        doc_id TEXT,
        case_row_id TEXT,
        case_number TEXT,
        district_id TEXT,
        patent TEXT,
        patent_number TEXT,
        patent_doc_type TEXT,
        case_type TEXT,
        date_filed TEXT,
        case_name TEXT,
        source_file TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS load_audit (
        source_type TEXT,
        documents_seen INTEGER,
        documents_loaded INTEGER,
        rows_inserted INTEGER,
        loaded_at TEXT,
        notes TEXT
    )
    """,
]
