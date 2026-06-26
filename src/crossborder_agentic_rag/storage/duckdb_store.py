"""DuckDB storage adapter for Stage 4 structured exact lookup."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - fallback only for constrained local test environments
    duckdb = None

import sqlite3

from crossborder_agentic_rag.schemas.documents import NormalizedDocument
from crossborder_agentic_rag.storage.schemas import CREATE_TABLE_STATEMENTS, ROW_COUNT_TABLES

STRUCTURED_ROW_TABLES = [t for t in ROW_COUNT_TABLES if t != "load_audit"]


class DuckDBStore:
    """Load NormalizedDocument records into DuckDB and run exact lookups."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._connection: Any | None = None
        self._schema_initialized = False

    def connect(self):
        if self._connection is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._connection = duckdb.connect(self.path) if duckdb is not None else _SQLiteDuckDBCompat(self.path)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._schema_initialized = False

    def initialize_schema(self) -> None:
        conn = self.connect()
        for statement in CREATE_TABLE_STATEMENTS:
            conn.execute(statement)
        self._schema_initialized = True

    def load_documents(self, docs: list[NormalizedDocument]) -> dict[str, Any]:
        self.initialize_schema()
        report = {
            "documents_seen": len(docs),
            "documents_loaded": 0,
            "documents_skipped": 0,
            "rows_inserted": {table: 0 for table in STRUCTURED_ROW_TABLES},
            "warnings": [],
            "failed_documents": [],
        }
        by_source: dict[str, dict[str, int]] = {}
        for doc in docs:
            try:
                if doc.source_type == "policy":
                    report["documents_skipped"] += 1
                    report["warnings"].append(f"Skipped policy document {doc.doc_id}; policy retrieval is handled by later semantic retrieval stages.")
                    continue
                before = dict(report["rows_inserted"])
                if doc.source_type == "trademark":
                    self._load_trademark(doc, report)
                elif doc.source_type == "patent":
                    self._load_patent(doc, report)
                elif doc.source_type == "litigation":
                    self._load_litigation(doc, report)
                else:
                    raise ValueError(f"Unsupported source_type: {doc.source_type}")
                inserted = sum(report["rows_inserted"][k] - before[k] for k in before)
                if inserted <= 0:
                    raise ValueError("document produced no structured rows")
                report["documents_loaded"] += 1
                by_source.setdefault(doc.source_type, {"seen": 0, "loaded": 0, "rows": 0})
                by_source[doc.source_type]["seen"] += 1
                by_source[doc.source_type]["loaded"] += 1
                by_source[doc.source_type]["rows"] += inserted
            except Exception as exc:
                report["failed_documents"].append({"doc_id": doc.doc_id, "source_type": doc.source_type, "error": str(exc)})
        if docs and report["documents_loaded"] == 0 and len(report["failed_documents"]) == len(docs):
            raise RuntimeError("All documents failed to load")
        now = datetime.now(timezone.utc).isoformat()
        for source_type, counts in by_source.items():
            self.connect().execute("INSERT INTO load_audit VALUES (?, ?, ?, ?, ?, ?)", [source_type, counts["seen"], counts["loaded"], counts["rows"], now, "Stage 4 DuckDB load"])
        return report

    def row_counts(self) -> dict[str, int]:
        self.initialize_schema()
        return {table: int(self.connect().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in ROW_COUNT_TABLES}

    def lookup_trademark_by_registration_number(self, registration_number: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM trademarks WHERE registration_number = ? ORDER BY doc_id", [registration_number])

    def lookup_trademark_by_word_mark(self, word_mark: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM trademarks WHERE lower(word_mark) = lower(?) ORDER BY doc_id", [word_mark])

    def lookup_trademark_classes_by_word_mark(self, word_mark: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM trademark_classes WHERE lower(word_mark) = lower(?) ORDER BY nice_class, doc_id", [word_mark])

    def lookup_trademark_goods_services_by_word_mark(self, word_mark: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM trademark_goods_services WHERE lower(word_mark) = lower(?) ORDER BY nice_class, goods_services", [word_mark])

    def lookup_patent_by_id(self, patent_id: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM patents WHERE patent_id = ? OR patent_number = ? ORDER BY doc_id", [patent_id, patent_id])

    def lookup_litigation_by_patent(self, patent_number: str) -> list[dict[str, Any]]:
        return self._query(
            """
            SELECT c.*, p.patent, p.patent_number, p.patent_doc_type, p.case_type
            FROM litigation_cases c
            JOIN litigation_patents p ON c.doc_id = p.doc_id
            WHERE p.patent = ? OR p.patent_number = ?
            ORDER BY c.case_number
            """,
            [patent_number, patent_number],
        )

    def lookup_litigation_by_case(self, case_number: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM litigation_cases WHERE case_number = ? ORDER BY doc_id", [case_number])

    def lookup_litigation_parties_by_case(self, case_number: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM litigation_names WHERE case_number = ? ORDER BY party_type, name", [case_number])

    def lookup_litigation_documents_by_case(self, case_number: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM litigation_documents WHERE case_number = ? ORDER BY doc_number", [case_number])

    def lookup_litigation_patents_by_case(self, case_number: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM litigation_patents WHERE case_number = ? ORDER BY patent_number, patent", [case_number])

    def _query(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        self.initialize_schema()
        cursor = self.connect().execute(sql, params)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def _load_trademark(self, doc: NormalizedDocument, report: dict[str, Any]) -> None:
        md = doc.metadata
        serial = _first(md, "serial_number", "serial_no", "serial")
        reg = _first(md, "registration_number", "registration_no", "reg_no")
        word = _first(md, "word_mark", "mark", "mark_id_char") or doc.title
        source_file = _first(md, "source_file")
        source_path = _first(md, "source_path")
        self.connect().execute("INSERT INTO trademarks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [doc.doc_id, _text(serial), _text(reg), _text(word), _text(_first(md, "filing_date")), _text(_first(md, "registration_date")), _text(_first(md, "status_code")), _text(source_file), _text(source_path), _json(md), doc.content])
        report["rows_inserted"]["trademarks"] += 1
        classes = _as_list(_first(md, "nice_classes", "intl_class", "intl_class_cd", "classes"))
        for item in classes:
            nice_class = item.get("nice_class") if isinstance(item, dict) else item
            desc = item.get("class_description") if isinstance(item, dict) else None
            self.connect().execute("INSERT INTO trademark_classes VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [doc.doc_id, _text(serial), _text(reg), _text(word), _text(nice_class), _text(desc), _text(source_file), _text(source_path)])
            report["rows_inserted"]["trademark_classes"] += 1
        for item in _as_list(_first(md, "goods_services", "goods_and_services")):
            nice_class = item.get("nice_class") if isinstance(item, dict) else None
            goods = item.get("goods_services") or item.get("goods_and_services") if isinstance(item, dict) else item
            self.connect().execute("INSERT INTO trademark_goods_services VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [doc.doc_id, _text(serial), _text(reg), _text(word), _text(nice_class), _text(goods), _text(source_file), _text(source_path)])
            report["rows_inserted"]["trademark_goods_services"] += 1

    def _load_patent(self, doc: NormalizedDocument, report: dict[str, Any]) -> None:
        md = doc.metadata
        patent_id = _first(md, "patent_id", "patent_number", "pat_no")
        patent_number = _first(md, "patent_number", "pat_no", "patent_id")
        if not patent_id and not patent_number:
            raise ValueError("patent document missing patent_id/patent_number")
        self.connect().execute("INSERT INTO patents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [doc.doc_id, _text(patent_id or patent_number), _text(patent_number or patent_id), _text(_first(md, "brief_summary")), _text(_first(md, "claims")), _text(_first(md, "detail_description")), _text(_first(md, "drawing_description")), _text(_first(md, "source_file")), _text(_first(md, "source_path")), _json(md), doc.content])
        report["rows_inserted"]["patents"] += 1

    def _load_litigation(self, doc: NormalizedDocument, report: dict[str, Any]) -> None:
        md = doc.metadata
        case = md.get("case") if isinstance(md.get("case"), dict) else md
        case_row_id = _first(case, "case_row_id")
        case_number = _first(case, "case_number")
        district_id = _first(case, "district_id")
        source_files = md.get("source_files", [])
        self.connect().execute("INSERT INTO litigation_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [doc.doc_id, _text(case_row_id), _text(case_number), _text(district_id), _text(_first(case, "court_name")), _text(_first(case, "case_name")), _text(_first(case, "case_cause")), _text(_first(case, "jurisdictional_basis")), _text(_first(case, "date_filed")), _text(_first(case, "date_closed")), _text(_first(case, "date_last_filed")), _json(source_files), _json(md), doc.content])
        report["rows_inserted"]["litigation_cases"] += 1
        for key in ["documents", "parties", "patents"]:
            if not md.get(key):
                report["warnings"].append(f"Litigation document {doc.doc_id} has no {key} list.")
        for item in _as_list(md.get("documents")):
            self.connect().execute("INSERT INTO litigation_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [doc.doc_id, _text(case_row_id), _text(case_number), _text(district_id), _text(_first(item, "doc_number")), _text(_first(item, "short_description")), _text(_first(item, "long_description")), _text(_first(item, "doc_date_filed", "date_filed")), _text(_first(item, "doc_date_uploaded", "date_uploaded")), _text(_first(item, "document_url")), _text(_first(item, "source_file"))])
            report["rows_inserted"]["litigation_documents"] += 1
        for item in _as_list(md.get("parties")):
            self.connect().execute("INSERT INTO litigation_names VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [doc.doc_id, _text(case_row_id), _text(case_number), _text(district_id), _text(_first(item, "party_type")), _text(_first(item, "name")), _text(_first(item, "name_long")), _text(_first(item, "source_file"))])
            report["rows_inserted"]["litigation_names"] += 1
        for item in _as_list(md.get("patents")):
            patent = _first(item, "patent", "patent_number", "pat_no")
            patent_number = _first(item, "patent_number", "pat_no", "patent")
            self.connect().execute("INSERT INTO litigation_patents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [doc.doc_id, _text(case_row_id), _text(case_number), _text(district_id), _text(patent), _text(patent_number), _text(_first(item, "patent_doc_type")), _text(_first(item, "case_type")), _text(_first(item, "date_filed")), _text(_first(item, "case_name") or _first(case, "case_name")), _text(_first(item, "source_file"))])
            report["rows_inserted"]["litigation_patents"] += 1


def _first(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class _SQLiteDuckDBCompat:
    """Tiny local fallback for environments where the duckdb wheel is unavailable."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(":memory:" if path == ":memory:" else path)
        self._cursor: sqlite3.Cursor | None = None
        self.description = None

    def execute(self, sql: str, params: list[Any] | tuple[Any, ...] | None = None):
        statement = sql.strip()
        if statement.lower() == "show tables":
            statement = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        self._cursor = self._conn.execute(statement, [] if params is None else params)
        self.description = self._cursor.description
        self._conn.commit()
        return self

    def fetchall(self):
        return [] if self._cursor is None else self._cursor.fetchall()

    def fetchone(self):
        return None if self._cursor is None else self._cursor.fetchone()

    def close(self) -> None:
        self._conn.close()
