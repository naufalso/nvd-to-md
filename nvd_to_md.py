#!/usr/bin/env python3
"""Download NVD CVE records, convert to Markdown and push to Hugging Face datasets.

This script iterates over yearly NVD CVE JSON feeds, transforms each entry to a
compact Markdown representation and optionally uploads the result as a dataset
to the Hugging Face Hub. All configuration is done via command line arguments
with environment variable fallbacks for convenience.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import io
import json
import os
import textwrap
from typing import Dict, List

import requests
from datasets import Dataset, Features, Value
from huggingface_hub import HfApi
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NVD_URL_FMT = (
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download NVD CVE records and create a dataset"
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face authentication token",
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_REPO"),
        help="Dataset repository id, e.g. 'username/nvd-cve-markdown'",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=int(os.getenv("START_YEAR", "2002")),
        help="First year to download (default 2002)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=int(os.getenv("END_YEAR", str(datetime.datetime.utcnow().year))),
        help="Last year to download (default current year)",
    )
    parser.add_argument(
        "--save-jsonl",
        help="Path to save records as JSON Lines",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push the dataset to the Hugging Face Hub",
    )
    args = parser.parse_args()
    if not args.no_push and (not args.hf_token or not args.repo_id):
        parser.error("HF token and repo id are required unless --no-push is set")
    return args

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
session = requests.Session()
session.headers.update({"User-Agent": "NVD-CVE-MD/1.0 (https://huggingface.co)"})

def download_feed(year: int) -> Dict:
    """Download and return the JSON feed for a given year."""
    url = NVD_URL_FMT.format(year=year)
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
        return json.load(gz)


def extract_markdown(item: Dict) -> tuple[str, List[Dict], List[str]]:
    """Convert one CVE item to Markdown and return CVSS metrics and weaknesses.

    The returned CVSS metrics are a list of all available versions instead of
    only the most recent one.
    """
    meta = item["cve"]["CVE_data_meta"]
    cve_id = meta["ID"]

    # ---------- Description / dates ----------
    desc_en = next(
        (d["value"] for d in item["cve"]["description"]["description_data"] if d["lang"] == "en"),
        "No description",
    )
    published = item.get("publishedDate", "")
    modified = item.get("lastModifiedDate", "")

    # ---------- CVSS metrics (gather all available) ----------
    cvss: List[Dict] = []
    impact = item.get("impact", {})
    v3 = impact.get("baseMetricV3", {})
    v2 = impact.get("baseMetricV2", {})
    if v3:
        c = v3.get("cvssV3", {})
        cvss.append(
            {
                "version": c.get("version", "3.x"),
                "score": c.get("baseScore"),
                "severity": c.get("baseSeverity"),
                "vector": c.get("vectorString"),
            }
        )
    if v2:
        c = v2.get("cvssV2", {})
        cvss.append(
            {
                "version": c.get("version", "2.0"),
                "score": c.get("baseScore"),
                "severity": v2.get("severity", "UNKNOWN"),
                "vector": c.get("vectorString"),
            }
        )

    # ---------- Weaknesses (CWE mapping) ----------
    weaknesses: List[str] = []
    for pdata in item["cve"]["problemtype"]["problemtype_data"]:
        for d in pdata.get("description", []):
            if d.get("lang") == "en" and d.get("value"):
                weaknesses.append(d["value"])

    # ---------- References ----------
    refs = [r["url"] for r in item["cve"]["references"]["reference_data"] if "url" in r]

    # ---------- Markdown assembly ----------
    md_lines = [
        f"# {cve_id}",
        "",
        f"**Published:** {published}  ",
        f"**Last Modified:** {modified}",
    ]

    # Description first
    md_lines += ["", "## Description", textwrap.fill(desc_en, width=100)]

    # Then CVSS metrics
    if cvss:
        md_lines += ["", "## CVSS"]
        for metric in cvss:
            md_lines += [
                f"### Version {metric['version']}",
                f"*Base Score*: **{metric['score']}** ({metric['severity']})  ",
                f"*Vector*: `{metric['vector']}`",
                "",
            ]
        if md_lines[-1] == "":
            md_lines.pop()

    # Finally weaknesses
    if weaknesses:
        md_lines += ["", "## Weaknesses"] + [f"- {w}" for w in weaknesses]

    if refs:
        md_lines += ["", "## References"] + [f"- {u}" for u in refs]

    return "\n".join(md_lines), cvss, weaknesses


# ---------------------------------------------------------------------------
# Main ETL
# ---------------------------------------------------------------------------

def build_records(start_year: int, end_year: int) -> List[Dict]:
    records: List[Dict] = []
    for year in tqdm(range(start_year, end_year + 1), desc="Years"):
        feed = download_feed(year)
        for item in feed.get("CVE_Items", []):
            cve_id = item["cve"]["CVE_data_meta"]["ID"]
            md, cvss, weaknesses = extract_markdown(item)
            records.append(
                {
                    "id": cve_id,
                    "text": md,
                    "publishedDate": item.get("publishedDate", ""),
                    "lastModifiedDate": item.get("lastModifiedDate", ""),
                    "cvss": cvss,
                    "weaknesses": weaknesses,
                }
            )
    return records


def build_dataset(records: List[Dict]) -> Dataset:
    features = Features(
        {
            "id": Value("string"),
            "text": Value("string"),
            "publishedDate": Value("string"),
            "lastModifiedDate": Value("string"),
            "cvss": Value("string"),  # stored as JSON string
            "weaknesses": Value("string"),  # stored as JSON string
        }
    )
    dataset = Dataset.from_list(records, features=features)
    dataset = dataset.map(
        lambda x: {
            "cvss": json.dumps(x["cvss"]),
            "weaknesses": json.dumps(x["weaknesses"]),
        }
    )
    return dataset


def push_dataset(dataset: Dataset, hf_token: str, repo_id: str) -> None:
    api = HfApi(token=hf_token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive; hub returns 409
        print("Repo creation skipped / already exists:", exc)
    print("Pushing dataset to Hub… this can take a while.")
    dataset.push_to_hub(repo_id, token=hf_token, split="train")
    print(f"✅ Done! View it at https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    args = parse_args()
    records = build_records(args.start_year, args.end_year)
    if args.save_jsonl:
        with open(args.save_jsonl, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
    dataset = build_dataset(records)
    if not args.no_push:
        push_dataset(dataset, args.hf_token, args.repo_id)


if __name__ == "__main__":
    main()
