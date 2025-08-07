# nvd-to-md

Convert [NVD CVE](https://nvd.nist.gov/vuln/data-feeds) JSON feeds into Markdown and
push the resulting dataset to the [Hugging Face Hub](https://huggingface.co/datasets).

## Usage

```bash
pip install -r requirements.txt
python nvd_to_md.py \
    --hf-token your_huggingface_token \
    --repo-id yourname/nvd-cve-markdown
```

Optional arguments:

- `--start-year` – first year to download (default `2002`)
- `--end-year` – last year to download (default current year)
- `--save-jsonl` – path to write records as JSON Lines before uploading
- `--no-push` – skip uploading the dataset to the Hugging Face Hub

The dataset uploaded to Hugging Face contains at least two columns:
- `id` – CVE identifier
- `text` – Markdown text with description and references

Additional metadata like publication dates and CVSS metrics are also included.
