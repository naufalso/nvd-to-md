#!/usr/bin/env python3
"""Download NVD CVE records, convert to Markdown and push to Hugging Face datasets.

This script iterates over all yearly NVD CVE JSON feeds, transforms each entry to a
compact Markdown representation and uploads the result as a dataset to the
Hugging Face Hub.

Environment variables required:
    HF_TOKEN  – Hugging Face authentication token
    HF_REPO   – Dataset repository id, e.g. "username/nvd-cve-markdown"
Optional variables:
    START_YEAR – first year to download (default 2002)
    END_YEAR   – last year to download (default current year)
"""
from __future__ import annotations

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
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("HF_REPO")  # e.g. "yourname/nvd-cve-markdown"
START_YR = int(os.getenv("START_YEAR", "2002"))
END_YR = int(os.getenv("END_YEAR", str(datetime.datetime.utcnow().year)))

assert HF_TOKEN and REPO_ID, "Set HF_TOKEN and HF_REPO environment variables"

NVD_URL_FMT = (
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
)

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


def extract_markdown(item: Dict) -> str:
    """Convert one CVE item to a Markdown formatted string."""
    meta = item["cve"]["CVE_data_meta"]
    cve_id = meta["ID"]
    desc_en = next(
        (d["value"] for d in item["cve"]["description"]["description_data"]
         if d["lang"] == "en"),
        "No description",
    )
    published = item.get("publishedDate", "")
    modified = item.get("lastModifiedDate", "")
    refs = [
        r["url"] for r in item["cve"]["references"]["reference_data"] if "url" in r
    ]

    md_lines = [
        f"# {cve_id}",
        "",
        f"**Published:** {published}  ",
        f"**Last Modified:** {modified}",
        "",
        "## Description",
        textwrap.fill(desc_en, width=100),
    ]

    if refs:
        md_lines.extend(["", "## References"])
        md_lines.extend([f"- {u}" for u in refs])

    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# Main ETL
# ---------------------------------------------------------------------------

def build_records(start_year: int, end_year: int) -> List[Dict]:
    records: List[Dict] = []
    for year in tqdm(range(start_year, end_year + 1), desc="Years"):
        feed = download_feed(year)
        for item in feed.get("CVE_Items", []):
            cve_id = item["cve"]["CVE_data_meta"]["ID"]
            md = extract_markdown(item)
            records.append(
                {
                    "id": cve_id,
                    "text": md,
                    "publishedDate": item.get("publishedDate", ""),
                    "lastModifiedDate": item.get("lastModifiedDate", ""),
                    "cvssMetricV31": item.get("impact", {}).get("baseMetricV3", {}),
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
            "cvssMetricV31": Value("string"),  # store as JSON string
        }
    )
    dataset = Dataset.from_list(records, features=features)
    dataset = dataset.map(
        lambda x: {"cvssMetricV31": json.dumps(x["cvssMetricV31"])}
    )
    return dataset


def push_dataset(dataset: Dataset) -> None:
    api = HfApi(token=HF_TOKEN)
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive; hub returns 409
        print("Repo creation skipped / already exists:", exc)
    print("Pushing dataset to Hub… this can take a while.")
    dataset.push_to_hub(REPO_ID, token=HF_TOKEN, split="train")
    print(f"✅ Done! View it at https://huggingface.co/datasets/{REPO_ID}")


def main() -> None:
    records = build_records(START_YR, END_YR)
    dataset = build_dataset(records)
    push_dataset(dataset)


if __name__ == "__main__":
    main()
