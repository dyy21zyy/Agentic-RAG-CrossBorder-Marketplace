#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compact_chunk(r: dict) -> dict:
    m = r.get("metadata") or {}
    return {
        "chunk_id": r.get("chunk_id") or r.get("id") or "",
        "doc_id": r.get("doc_id") or "",
        "source_type": r.get("source_type") or "",
        "source_subtype": r.get("source_subtype") or "",
        "title": r.get("title") or "",
        "content": r.get("content") or "",
        "metadata": {
            "word_mark": m.get("word_mark"),
            "serial_number": m.get("serial_number"),
            "registration_number": m.get("registration_number"),
            "nice_class": m.get("nice_class"),
            "nice_classes": m.get("nice_classes"),
            "goods_services": m.get("goods_services"),
            "patent_id": m.get("patent_id"),
            "claim_number": m.get("claim_number"),
            "case_number": m.get("case_number"),
            "case_title": m.get("case_title") or m.get("title"),
            "doc_number": m.get("doc_number"),
            "party_name": m.get("party_name"),
        },
    }


def add_limited(group: dict, chunk: dict, limit_per_subtype: int = 8):
    sub = chunk["source_subtype"]
    xs = group["chunks_by_subtype"][sub]
    if len(xs) < limit_per_subtype:
        xs.append(chunk)


def first(xs):
    return xs[0] if xs else None


def unique_chunks(chunks):
    seen = set()
    out = []
    for c in chunks:
        if not c:
            continue
        cid = c["chunk_id"]
        if cid and cid not in seen:
            out.append(c)
            seen.add(cid)
    return out


def claim_no(c: dict) -> str:
    m = c.get("metadata") or {}
    if m.get("claim_number") is not None:
        return str(m.get("claim_number"))
    text = (c.get("title") or "") + " " + (c.get("content") or "")
    m2 = re.search(r"\bclaim\s+(\d+)\b", text, flags=re.I)
    if m2:
        return m2.group(1)
    m3 = re.search(r"claim-\d+-(\d+)", c.get("chunk_id") or "")
    if m3:
        return m3.group(1)
    return "1"


def extract_case_number(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\b\d+:\d{2,4}-cv-\d{2,6}\b", text)
    return m.group(0) if m else None


def clean_label(s: str, fallback: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s if s else fallback


def is_good_word_mark(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False

    low = s.lower()

    bad_phrases = [
        "the mark consists",
        "the mark comprises",
        "the mark includes",
        "stylized wording",
        "stylized word",
        "design forming",
        "appears below",
        "appears above",
        "the wording",
        "color is not claimed",
        "drawing",
        "literal element",
        "description of mark",
    ]

    if any(x in low for x in bad_phrases):
        return False

    # 太长的通常不是正常商标名，而是图形/设计描述
    if len(s) > 80:
        return False

    # 词太多也不像普通 word mark
    if len(s.split()) > 10:
        return False

    # 标点太多也过滤
    punct_count = sum(1 for ch in s if ch in ".,;:“”\"()[]{}")
    if punct_count >= 5:
        return False

    return True


def make_record(
    *,
    rid: str,
    query: str,
    query_type: str,
    task_type: str,
    core_chunks: list[dict],
    support_chunks: list[dict],
    target_entities: dict,
    expected_source_types: list[str],
    expected_source_subtypes: list[str],
    gold_answer_key_points: list[str],
    hard_negative_chunk_ids: list[str] | None = None,
):
    core_chunks = unique_chunks(core_chunks)
    support_chunks = [c for c in unique_chunks(support_chunks) if c["chunk_id"] not in {x["chunk_id"] for x in core_chunks}]

    # relevant_chunk_ids 用于兼容现有 evaluator：核心证据 + 强辅助证据都计入相关。
    relevant = unique_chunks(core_chunks + support_chunks)
    grades = {}
    for c in core_chunks:
        grades[c["chunk_id"]] = 3
    for c in support_chunks:
        grades[c["chunk_id"]] = 2

    relevant_doc_ids = sorted({c["doc_id"] for c in relevant if c.get("doc_id")})

    return {
        "id": rid,
        "query": query,
        "query_type": query_type,
        "task_type": task_type,
        "expected_route": "agentic_hybrid_retrieval",
        "expected_answer_type": f"{query_type}_answer",
        "expected_source_types": expected_source_types,
        "expected_source_subtypes": expected_source_subtypes,
        "target_entities": target_entities,
        "relevant_doc_ids": relevant_doc_ids,
        "relevant_chunk_ids": [c["chunk_id"] for c in relevant],
        "strict_relevant_chunk_ids": [c["chunk_id"] for c in core_chunks],
        "acceptable_relevant_chunk_ids": [c["chunk_id"] for c in relevant],
        "supporting_chunk_ids": [c["chunk_id"] for c in support_chunks],
        "relevance_grades": grades,
        "hard_negative_chunk_ids": hard_negative_chunk_ids or [],
        "must_contain_any": [],
        "gold_answer": "A good answer should retrieve and cite the graded evidence chunks, cover the requested entities and subtypes, and avoid legal advice.",
        "gold_answer_key_points": gold_answer_key_points,
        "metadata": {
            "source": "v2_multigold_auto",
            "gold_count": len(relevant),
            "strict_gold_count": len(core_chunks),
            "supporting_gold_count": len(support_chunks),
        },
    }


def sample_n(xs, n, rng):
    xs = list(xs)
    rng.shuffle(xs)
    return xs[: min(n, len(xs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/ip_evidence_chunks_full_optimized_fixed.jsonl")
    ap.add_argument("--out", default="data/eval/chunk_grounded_eval_v2_multigold_300.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    chunks_path = Path(args.chunks)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tm = defaultdict(lambda: {"chunks_by_subtype": defaultdict(list), "word_mark": None, "serial_number": None})
    patent = defaultdict(lambda: {"chunks_by_subtype": defaultdict(list), "patent_id": None})
    lit = defaultdict(lambda: {"chunks_by_subtype": defaultdict(list), "case_number": None, "case_title": None})

    source_counter = Counter()
    subtype_counter = Counter()

    print(f"Loading chunks from {chunks_path} ...")

    for r in load_jsonl(chunks_path):
        c = compact_chunk(r)
        st = c["source_type"]
        sub = c["source_subtype"]
        source_counter[st] += 1
        subtype_counter[sub] += 1

        m = c["metadata"]

        if st == "trademark":
            serial = m.get("serial_number") or c["doc_id"].replace("trademark:", "")
            g = tm[serial]
            g["serial_number"] = serial
            g["word_mark"] = g["word_mark"] or m.get("word_mark") or (c["title"].split()[0] if c["title"] else serial)
            add_limited(g, c)

        elif st == "patent":
            pid = m.get("patent_id") or c["doc_id"].replace("patent:", "")
            g = patent[pid]
            g["patent_id"] = pid
            add_limited(g, c, limit_per_subtype=12)

        elif st == "litigation":
            doc_id = c["doc_id"] or c["chunk_id"].split(":")[0]
            g = lit[doc_id]
            add_limited(g, c, limit_per_subtype=8)

            text_for_case = " ".join([
                str(m.get("case_number") or ""),
                c.get("title") or "",
                c.get("content") or "",
                c.get("chunk_id") or "",
                c.get("doc_id") or "",
            ])
            case_number = m.get("case_number") or extract_case_number(text_for_case)
            if case_number:
                g["case_number"] = g["case_number"] or case_number

            if not g["case_title"]:
                title = c.get("title") or ""
                title = re.sub(r"\s+(summary|timeline|docket.*|party.*|patent.*)$", "", title, flags=re.I).strip()
                g["case_title"] = title or c["doc_id"]

    print("Loaded source types:", dict(source_counter))
    print("Loaded subtypes:", dict(subtype_counter))

    records = []
    used_queries = set()

    def add_record(rec):
        if not rec:
            return
        if rec["query"] in used_queries:
            return
        if len(rec["relevant_chunk_ids"]) < 2:
            return
        records.append(rec)
        used_queries.add(rec["query"])

    # Candidate groups
    tm_profile_groups = [
        g for g in tm.values()
        if first(g["chunks_by_subtype"].get("trademark_identity"))
        and first(g["chunks_by_subtype"].get("trademark_class"))
        and first(g["chunks_by_subtype"].get("trademark_goods_services"))
        and g.get("word_mark")
        and is_good_word_mark(g.get("word_mark"))
    ]

    patent_multi_groups = [
        g for g in patent.values()
        if len(g["chunks_by_subtype"].get("patent_claim", [])) >= 2
        and g.get("patent_id")
    ]

    lit_overview_groups = []
    for g in lit.values():
        subs = g["chunks_by_subtype"]
        available = [
            s for s in [
                "litigation_case_summary",
                "litigation_timeline",
                "litigation_docket",
                "litigation_party",
                "litigation_patent",
            ]
            if first(subs.get(s))
        ]
        if len(available) >= 3:
            lit_overview_groups.append(g)

    print("Candidate trademark profile groups:", len(tm_profile_groups))
    print("Candidate patent multi-claim groups:", len(patent_multi_groups))
    print("Candidate litigation overview groups:", len(lit_overview_groups))

    # 1) 50 trademark profile
    for i, g in enumerate(sample_n(tm_profile_groups, 50, rng), start=1):
        subs = g["chunks_by_subtype"]
        word = clean_label(g["word_mark"], g["serial_number"])
        serial = g["serial_number"]
        core = [
            first(subs["trademark_identity"]),
            first(subs["trademark_class"]),
            first(subs["trademark_goods_services"]),
        ]
        rec = make_record(
            rid=f"V2_TM_PROFILE_{i:04d}",
            query=f"Retrieve trademark profile evidence for {word}, including identity, Nice class, and goods/services.",
            query_type="trademark",
            task_type="multi_evidence_profile",
            core_chunks=core,
            support_chunks=[],
            target_entities={"word_mark": word, "serial_number": serial},
            expected_source_types=["trademark"],
            expected_source_subtypes=["trademark_identity", "trademark_class", "trademark_goods_services"],
            gold_answer_key_points=[
                f"word mark is {word}",
                "identity or registration evidence",
                "Nice class evidence",
                "goods/services evidence",
            ],
        )
        add_record(rec)

    # 2) 40 trademark focused but still multi-gold
    focus_templates = [
        ("trademark_identity", "Find trademark identity and registration evidence for {word}, with supporting Nice class and goods/services context."),
        ("trademark_class", "Which Nice class evidence is associated with the trademark {word}, with supporting identity and goods/services context?"),
        ("trademark_goods_services", "Find trademark goods and services evidence for {word}, with supporting identity and Nice class context."),
    ]

    for i, g in enumerate(sample_n(tm_profile_groups, 40, rng), start=1):
        focus_sub, template = focus_templates[(i - 1) % len(focus_templates)]
        subs = g["chunks_by_subtype"]
        word = clean_label(g["word_mark"], g["serial_number"])
        serial = g["serial_number"]
        core = [first(subs[focus_sub])]
        support = []
        for s in ["trademark_identity", "trademark_class", "trademark_goods_services"]:
            if s != focus_sub:
                support.append(first(subs[s]))
        rec = make_record(
            rid=f"V2_TM_FOCUS_{i:04d}",
            query=template.format(word=word),
            query_type="trademark",
            task_type=f"focused_{focus_sub}",
            core_chunks=core,
            support_chunks=support,
            target_entities={"word_mark": word, "serial_number": serial},
            expected_source_types=["trademark"],
            expected_source_subtypes=[focus_sub, "trademark_identity", "trademark_class", "trademark_goods_services"],
            gold_answer_key_points=[f"target trademark is {word}", focus_sub.replace("_", " ")],
        )
        add_record(rec)

    # 3) 40 patent generic multi-claim
    for i, g in enumerate(sample_n(patent_multi_groups, 40, rng), start=1):
        pid = g["patent_id"]
        claims = g["chunks_by_subtype"]["patent_claim"][:5]
        core = claims[: min(3, len(claims))]
        support = claims[3:5]
        rec = make_record(
            rid=f"V2_PATENT_MULTI_{i:04d}",
            query=f"Retrieve multiple patent claim evidence chunks for patent {pid}.",
            query_type="patent",
            task_type="multi_claim_evidence",
            core_chunks=core,
            support_chunks=support,
            target_entities={"patent_id": pid},
            expected_source_types=["patent"],
            expected_source_subtypes=["patent_claim"],
            gold_answer_key_points=[f"patent id is {pid}", "multiple claim evidence chunks"],
        )
        add_record(rec)

    # 4) 30 patent exact claim with same-patent support
    for i, g in enumerate(sample_n(patent_multi_groups, 30, rng), start=1):
        pid = g["patent_id"]
        claims = g["chunks_by_subtype"]["patent_claim"][:5]
        target = claims[0]
        cno = claim_no(target)
        support = claims[1:4]
        rec = make_record(
            rid=f"V2_PATENT_EXACT_{i:04d}",
            query=f"Find claim {cno} evidence for patent {pid}, and include supporting claims from the same patent.",
            query_type="patent",
            task_type="exact_claim_with_support",
            core_chunks=[target],
            support_chunks=support,
            target_entities={"patent_id": pid, "claim_number": cno},
            expected_source_types=["patent"],
            expected_source_subtypes=["patent_claim"],
            gold_answer_key_points=[f"patent id is {pid}", f"claim {cno}", "supporting claims from the same patent"],
        )
        add_record(rec)

    # 5) 60 litigation overview multi-evidence
    for i, g in enumerate(sample_n(lit_overview_groups, 60, rng), start=1):
        subs = g["chunks_by_subtype"]
        label = g.get("case_number") or clean_label(g.get("case_title"), "the litigation case")
        core = []
        for s in ["litigation_case_summary", "litigation_timeline", "litigation_docket", "litigation_party", "litigation_patent"]:
            c = first(subs.get(s))
            if c:
                core.append(c)
        core = core[:5]
        rec = make_record(
            rid=f"V2_LIT_OVERVIEW_{i:04d}",
            query=f"Retrieve litigation overview evidence for case {label}, including summary, timeline or docket, parties, and asserted patents when available.",
            query_type="litigation",
            task_type="multi_evidence_litigation_overview",
            core_chunks=core,
            support_chunks=[],
            target_entities={"case_number": g.get("case_number"), "case_title": g.get("case_title")},
            expected_source_types=["litigation"],
            expected_source_subtypes=list({c["source_subtype"] for c in core}),
            gold_answer_key_points=["case summary", "timeline or docket", "party evidence", "asserted patent evidence if available"],
        )
        add_record(rec)

    # 6) 40 litigation focused but multi-gold
    lit_focus_subs = ["litigation_docket", "litigation_party", "litigation_patent", "litigation_timeline"]
    lit_focus_candidates = []
    for g in lit_overview_groups:
        for s in lit_focus_subs:
            if first(g["chunks_by_subtype"].get(s)):
                lit_focus_candidates.append((g, s))

    for i, (g, focus_sub) in enumerate(sample_n(lit_focus_candidates, 40, rng), start=1):
        subs = g["chunks_by_subtype"]
        label = g.get("case_number") or clean_label(g.get("case_title"), "the litigation case")
        core = [first(subs[focus_sub])]
        support = []
        for s in ["litigation_case_summary", "litigation_timeline", "litigation_docket", "litigation_party", "litigation_patent"]:
            if s != focus_sub and first(subs.get(s)):
                support.append(first(subs[s]))
            if len(support) >= 2:
                break

        natural = {
            "litigation_docket": "docket",
            "litigation_party": "party",
            "litigation_patent": "asserted patent",
            "litigation_timeline": "timeline",
        }[focus_sub]

        rec = make_record(
            rid=f"V2_LIT_FOCUS_{i:04d}",
            query=f"Find {natural} evidence for litigation case {label}, with supporting case context.",
            query_type="litigation",
            task_type=f"focused_{focus_sub}",
            core_chunks=core,
            support_chunks=support,
            target_entities={"case_number": g.get("case_number"), "case_title": g.get("case_title")},
            expected_source_types=["litigation"],
            expected_source_subtypes=[focus_sub] + [c["source_subtype"] for c in support],
            gold_answer_key_points=[natural, "supporting case context"],
        )
        add_record(rec)

    # 7) 20 cross-source marketplace risk screening
    tm_sample = sample_n(tm_profile_groups, 20, rng)
    patent_sample = sample_n(patent_multi_groups, 20, rng)
    lit_sample = sample_n(lit_overview_groups, 20, rng)

    for i, (tg, pg, lg) in enumerate(zip(tm_sample, patent_sample, lit_sample), start=1):
        tsubs = tg["chunks_by_subtype"]
        pclaims = pg["chunks_by_subtype"]["patent_claim"]
        lsubs = lg["chunks_by_subtype"]

        word = clean_label(tg["word_mark"], tg["serial_number"])
        pid = pg["patent_id"]
        case_label = lg.get("case_number") or clean_label(lg.get("case_title"), "the litigation case")

        core = [
            first(tsubs["trademark_identity"]),
            first(tsubs["trademark_goods_services"]),
            first(pclaims),
            first(lsubs.get("litigation_case_summary")),
        ]
        support = [
            first(tsubs["trademark_class"]),
            first(lsubs.get("litigation_patent")),
            first(lsubs.get("litigation_party")),
        ]

        rec = make_record(
            rid=f"V2_CROSS_RISK_{i:04d}",
            query=f"For marketplace IP risk screening, gather evidence for trademark {word}, patent {pid}, and litigation case {case_label}.",
            query_type="cross_source",
            task_type="cross_source_ip_risk_screening",
            core_chunks=core,
            support_chunks=support,
            target_entities={
                "word_mark": word,
                "serial_number": tg["serial_number"],
                "patent_id": pid,
                "case_number": lg.get("case_number"),
                "case_title": lg.get("case_title"),
            },
            expected_source_types=["trademark", "patent", "litigation"],
            expected_source_subtypes=["trademark_identity", "trademark_goods_services", "trademark_class", "patent_claim", "litigation_case_summary", "litigation_patent", "litigation_party"],
            gold_answer_key_points=["trademark evidence", "patent claim evidence", "litigation evidence", "avoid legal advice"],
        )
        add_record(rec)

    # 8) 20 relationship / graph-style litigation queries
    graph_candidates = []
    for g in lit_overview_groups:
        subs = g["chunks_by_subtype"]
        if first(subs.get("litigation_party")) and first(subs.get("litigation_patent")):
            graph_candidates.append(g)

    for i, g in enumerate(sample_n(graph_candidates, 20, rng), start=1):
        subs = g["chunks_by_subtype"]
        label = g.get("case_number") or clean_label(g.get("case_title"), "the litigation case")

        core = [
            first(subs.get("litigation_party")),
            first(subs.get("litigation_patent")),
        ]
        support = [
            first(subs.get("litigation_case_summary")),
            first(subs.get("litigation_timeline")),
            first(subs.get("litigation_docket")),
        ]

        rec = make_record(
            rid=f"V2_GRAPH_REL_{i:04d}",
            query=f"Show relationship evidence around litigation case {label}, including parties, asserted patents, and case context.",
            query_type="graph_relationship",
            task_type="relationship_evidence",
            core_chunks=core,
            support_chunks=support,
            target_entities={"case_number": g.get("case_number"), "case_title": g.get("case_title")},
            expected_source_types=["litigation"],
            expected_source_subtypes=["litigation_party", "litigation_patent", "litigation_case_summary", "litigation_timeline", "litigation_docket"],
            gold_answer_key_points=["party relationship evidence", "asserted patent evidence", "case context"],
        )
        add_record(rec)

    # 截断到目标 n
    records = records[: args.n]

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records)} records to {out_path}")

    task_cnt = Counter(r["task_type"] for r in records)
    qtype_cnt = Counter(r["query_type"] for r in records)
    gold_counts = [len(r["relevant_chunk_ids"]) for r in records]
    strict_counts = [len(r["strict_relevant_chunk_ids"]) for r in records]

    print("\nquery_type distribution:")
    for k, v in qtype_cnt.most_common():
        print(k, v)

    print("\ntask_type distribution:")
    for k, v in task_cnt.most_common():
        print(k, v)

    print("\ngold count stats:")
    print("min =", min(gold_counts) if gold_counts else None)
    print("max =", max(gold_counts) if gold_counts else None)
    print("avg =", round(sum(gold_counts) / len(gold_counts), 4) if gold_counts else None)
    print("single_gold_count =", sum(1 for x in gold_counts if x == 1))

    print("\nstrict gold count stats:")
    print("min =", min(strict_counts) if strict_counts else None)
    print("max =", max(strict_counts) if strict_counts else None)
    print("avg =", round(sum(strict_counts) / len(strict_counts), 4) if strict_counts else None)

    print("\nfirst 3 examples:")
    for r in records[:3]:
        print(json.dumps(r, ensure_ascii=False, indent=2)[:2000])
        print("---")


if __name__ == "__main__":
    main()
