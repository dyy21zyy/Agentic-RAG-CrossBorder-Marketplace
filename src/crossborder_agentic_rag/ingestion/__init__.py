"""Ingestion interfaces for source documents."""
from crossborder_agentic_rag.ingestion.io_utils import read_documents_jsonl, write_documents_jsonl, write_report
from crossborder_agentic_rag.ingestion.litigation_parser import parse_litigation_csv_directory
from crossborder_agentic_rag.ingestion.patent_parser import parse_patent_tsv_directory
from crossborder_agentic_rag.ingestion.policy_parser import parse_policy_directory
from crossborder_agentic_rag.ingestion.trademark_parser import parse_trademark_xml_directory

__all__ = ["parse_trademark_xml_directory", "parse_patent_tsv_directory", "parse_litigation_csv_directory", "parse_policy_directory", "read_documents_jsonl", "write_documents_jsonl", "write_report"]
