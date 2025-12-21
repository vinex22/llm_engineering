#!/usr/bin/env python3
"""Bulk-convert course notebooks from OpenAI API-key auth to Azure OpenAI Entra ID auth.

This is intentionally conservative:
- Only touches notebooks whose code uses OpenAI() without base_url.
- Skips notebooks already using AzureOpenAI / azure_ad_token_provider.
- Skips notebooks that appear to use Ollama/OpenRouter/etc via OpenAI(base_url=...).

Run:
  python scripts/convert_notebooks_to_azure_entra.py

Optional:
  python scripts/convert_notebooks_to_azure_entra.py --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable


AZURE_ENDPOINT_DEFAULT = "https://ai221212sweeden.openai.azure.com/"
DEPLOYMENT_DEFAULT = "gpt-5.2"
SCOPE = "https://cognitiveservices.azure.com/.default"
API_VERSION_DEFAULT = "2025-01-01-preview"


@dataclass
class ConvertResult:
    path: str
    changed: bool
    reason: str


def iter_candidate_notebooks() -> list[str]:
    patterns = [
        "week[0-9]*/**/*.ipynb",
        "setup/**/*.ipynb",
        "extras/**/*.ipynb",
        "*.ipynb",
    ]

    paths: set[str] = set()
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            p = p.replace("\\\\", "/")
            paths.add(p)

    def is_excluded(p: str) -> bool:
        low = p.lower()
        return "community-contributions/" in low or "community_contributions/" in low

    return sorted([p for p in paths if not is_excluded(p)])


def _cell_source_to_str(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    if isinstance(src, str):
        return src
    return ""


def _str_to_cell_source(original: Any, new_text: str) -> Any:
    # Preserve Jupyter's typical list-of-lines format if present.
    if isinstance(original, list):
        # Keep line endings. Splitlines(True) preserves them.
        return new_text.splitlines(True)
    return new_text


def notebook_looks_azure_already(all_code: str) -> bool:
    return (
        "AzureOpenAI" in all_code
        or "azure_ad_token_provider" in all_code
        or "get_bearer_token_provider" in all_code
    )


def notebook_looks_non_azure_provider(all_code: str) -> bool:
    # Very common pattern for Ollama/OpenRouter/etc.
    if re.search(r"OpenAI\s*\(.*base_url\s*=", all_code, flags=re.IGNORECASE | re.DOTALL):
        return True
    return False


def notebook_looks_like_openai_key_flow(all_code: str) -> bool:
    return (
        "OPENAI_API_KEY" in all_code
        or re.search(r"\bOpenAI\s*\(", all_code) is not None
        or re.search(r"from\s+openai\s+import\s+OpenAI\b", all_code) is not None
    )


def ensure_imports(text: str) -> str:
    # Replace `from openai import OpenAI` with `AzureOpenAI`.
    text2 = re.sub(
        r"^\s*from\s+openai\s+import\s+OpenAI\s*$",
        "from openai import AzureOpenAI",
        text,
        flags=re.MULTILINE,
    )

    # Handle `from openai import OpenAI, ...` cases.
    text2 = re.sub(
        r"^\s*from\s+openai\s+import\s+OpenAI\s*,",
        "from openai import AzureOpenAI,",
        text2,
        flags=re.MULTILINE,
    )

    # If they already import AzureOpenAI, don't add.
    if re.search(r"^\s*from\s+openai\s+import\s+AzureOpenAI\b", text2, flags=re.MULTILINE):
        if not re.search(
            r"^\s*from\s+azure\.identity\s+import\s+DefaultAzureCredential\s*,\s*get_bearer_token_provider\s*$",
            text2,
            flags=re.MULTILINE,
        ):
            # Insert azure.identity import right after the openai import.
            text2 = re.sub(
                r"(^\s*from\s+openai\s+import\s+AzureOpenAI.*$)",
                r"\1\nfrom azure.identity import DefaultAzureCredential, get_bearer_token_provider",
                text2,
                flags=re.MULTILINE,
                count=1,
            )

    return text2


def build_azure_init_block(var_name: str) -> str:
    return (
        f"endpoint = os.getenv(\"ENDPOINT_URL\", \"{AZURE_ENDPOINT_DEFAULT}\")\n"
        f"deployment = os.getenv(\"DEPLOYMENT_NAME\", \"{DEPLOYMENT_DEFAULT}\")\n"
        "token_provider = get_bearer_token_provider(\n"
        "    DefaultAzureCredential(),\n"
        f"    \"{SCOPE}\",\n"
        ")\n"
        f"{var_name} = AzureOpenAI(\n"
        "    azure_endpoint=endpoint,\n"
        "    azure_ad_token_provider=token_provider,\n"
        f"    api_version=os.getenv(\"OPENAI_API_VERSION\", \"{API_VERSION_DEFAULT}\"),\n"
        ")\n"
    )


def replace_openai_client_init(text: str) -> tuple[str, bool]:
    """Replace simple OpenAI() client creation with AzureOpenAI Entra init.

    Only replaces calls that do NOT contain base_url.
    """

    # Match simple assignments: client = OpenAI() or openai = OpenAI(api_key=...)
    # Capture variable name and the call args.
    pattern = re.compile(
        r"^(?P<var>[A-Za-z_]\w*)\s*=\s*OpenAI\s*\((?P<args>[^\)]*)\)\s*$",
        flags=re.MULTILINE,
    )

    changed = False

    def repl(m: re.Match[str]) -> str:
        nonlocal changed
        args = m.group("args") or ""
        if re.search(r"\bbase_url\b", args):
            return m.group(0)
        # Some notebooks pass organization/project; treat as key-flow and override.
        changed = True
        return build_azure_init_block(m.group("var")).rstrip("\n")

    new_text = pattern.sub(repl, text)
    return new_text, changed


def replace_literal_model_with_deployment(text: str) -> tuple[str, bool]:
    # Replace model="..." with model=deployment when it looks like a literal.
    # Avoid touching already-variable models.
    model_pat = re.compile(r"\bmodel\s*=\s*([\"\'])(?P<val>[^\"\']+)\1")

    changed = False

    def repl(m: re.Match[str]) -> str:
        nonlocal changed
        val = m.group("val")
        if val.lower() in {"deployment", "deployment_name"}:
            return m.group(0)
        # If they already use an Azure deployment name literal, it's still fine to set deployment.
        changed = True
        return "model=deployment"

    return model_pat.sub(repl, text), changed


def convert_notebook(path: str, dry_run: bool) -> ConvertResult:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        return ConvertResult(path, False, "unexpected notebook format")

    all_code = "\n".join(
        _cell_source_to_str(c) for c in cells if c.get("cell_type") == "code"
    )

    if not notebook_looks_like_openai_key_flow(all_code):
        return ConvertResult(path, False, "no OpenAI usage detected")

    if notebook_looks_azure_already(all_code):
        return ConvertResult(path, False, "already Azure/Entra auth")

    if notebook_looks_non_azure_provider(all_code):
        return ConvertResult(path, False, "uses OpenAI(base_url=...) (likely non-Azure provider)")

    changed_any = False

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        src_original = cell.get("source", "")
        text = _cell_source_to_str(cell)
        if not text.strip():
            continue

        text2 = ensure_imports(text)
        if text2 != text:
            changed_any = True
            text = text2

        text2, changed = replace_openai_client_init(text)
        if changed:
            changed_any = True
            text = text2

        # If we inserted deployment variable, make sure model literals use it.
        if "deployment = os.getenv(\"DEPLOYMENT_NAME\"" in text:
            text2, changed = replace_literal_model_with_deployment(text)
            if changed:
                changed_any = True
                text = text2

        if changed_any:
            cell["source"] = _str_to_cell_source(src_original, text)

    if changed_any:
        all_code_after = "\n".join(
            _cell_source_to_str(c) for c in cells if c.get("cell_type") == "code"
        )
        needs_os_import = (
            "os.getenv(" in all_code_after
            and re.search(r"^\s*import\s+os\b", all_code_after, flags=re.MULTILINE) is None
        )
        if needs_os_import:
            # Add `import os` to the first non-empty code cell.
            for cell in cells:
                if cell.get("cell_type") != "code":
                    continue
                src_original = cell.get("source", "")
                text = _cell_source_to_str(cell)
                if not text.strip():
                    continue
                cell["source"] = _str_to_cell_source(src_original, f"import os\n{text}")
                break

    if not changed_any:
        return ConvertResult(path, False, "no safe transformations applied")

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")

    return ConvertResult(path, True, "converted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Do not write files")
    args = parser.parse_args()

    paths = iter_candidate_notebooks()

    results: list[ConvertResult] = []
    for p in paths:
        results.append(convert_notebook(p, dry_run=args.dry_run))

    changed = [r for r in results if r.changed]
    skipped = [r for r in results if not r.changed]

    print(f"total={len(results)} changed={len(changed)} skipped={len(skipped)}")
    for r in results:
        if r.changed:
            print(f"CHANGED: {r.path}")

    # Show a few skip reasons to help refine rules.
    summary: dict[str, int] = {}
    for r in skipped:
        summary[r.reason] = summary.get(r.reason, 0) + 1
    if summary:
        print("\nSkip reasons:")
        for k, v in sorted(summary.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {v:>3}  {k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
