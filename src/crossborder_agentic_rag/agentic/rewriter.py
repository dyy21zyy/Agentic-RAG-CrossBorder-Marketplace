def rewrite_for_scenario(query: str, scope: list[str]) -> dict[str, str]:
    q = " ".join(query.split())
    out: dict[str, str] = {}
    if "trademark" in scope:
        out["trademark"] = f"brand logo goods services {q}"
    if "patent" in scope:
        out["patent"] = f"technical features patent claims {q}"
    if "litigation" in scope:
        out["litigation"] = f"litigation case asserted patent {q}"
    return out
