# nvd-to-md

Convert [NVD CVE](https://nvd.nist.gov/vuln/data-feeds) JSON feeds into Markdown and
push the resulting dataset to the [Hugging Face Hub](https://huggingface.co/datasets).

## Usage

```bash
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
export HF_REPO=yourname/nvd-cve-markdown
python nvd_to_md.py
```

Optional environment variables:
- `START_YEAR` – first year to download (default `2002`)
- `END_YEAR` – last year to download (default current year)

The dataset uploaded to Hugging Face contains at least two columns:
- `id` – CVE identifier
- `text` – Markdown text with description and references

Additional metadata like publication dates and CVSS metrics are also included.
