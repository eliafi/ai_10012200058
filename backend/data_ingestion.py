# Student: Eli Afi Ayekpley | Index: 10012200058
# CS4241 - Introduction to Artificial Intelligence | ACity 2026
import os
import re
import json
import requests
import pandas as pd
import pdfplumber
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).parent.parent / "data")))
CSV_URL = "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
PDF_URL = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v5.pdf"
CSV_PATH = DATA_DIR / "Ghana_Election_Result.csv"
PDF_PATH = DATA_DIR / "2025_Budget_Statement.pdf"
CHUNKS_PATH = DATA_DIR / "chunks.json"

# Sliding window: 500 words per chunk, 50 word overlap.
# Justification: PDF pages contain dense policy text; fixed-size word windows
# ensure each chunk is semantically coherent and fits within embedding model
# limits, while 50-word overlap preserves cross-boundary context so retrieval
# does not lose sentences split at chunk boundaries.
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 50


def clean_text(text: str) -> str:
    """Strip null bytes, collapse whitespace, fix ligatures, remove non-ASCII."""
    text = text.replace("\x00", "")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\ufb03", "ffi").replace("\ufb04", "ffl")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_file(url: str, dest: Path, retries: int = 5) -> None:
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=180, headers=headers) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            print(f"  Saved to {dest}")
            return
        except Exception as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            if dest.exists():
                dest.unlink()
            if attempt == retries:
                raise


def build_election_summaries(df: pd.DataFrame) -> list[dict]:
    """
    Create national-level aggregate summary chunks per election year.
    These match broad queries like 'highlights', 'who won', 'results overview'
    that don't align well with individual per-region rows.
    """
    summaries = []
    votes_col = next((c for c in df.columns if "votes" in c.lower() and "%" not in c), None)
    pct_col = next((c for c in df.columns if "%" in c), None)
    year_col = next((c for c in df.columns if "year" in c.lower()), None)
    cand_col = next((c for c in df.columns if "candidate" in c.lower()), None)
    party_col = next((c for c in df.columns if "party" in c.lower()), None)

    if not all([votes_col, year_col, cand_col, party_col]):
        return summaries

    df = df.copy()
    df[votes_col] = pd.to_numeric(df[votes_col].str.replace(",", ""), errors="coerce").fillna(0)

    for year, ydf in df.groupby(year_col):
        national = (
            ydf.groupby([cand_col, party_col])[votes_col]
            .sum()
            .reset_index()
            .sort_values(votes_col, ascending=False)
        )
        total_votes = national[votes_col].sum()
        lines = [f"Ghana {year} Presidential Election — National Results Summary:"]
        for _, r in national.iterrows():
            pct = (r[votes_col] / total_votes * 100) if total_votes else 0
            lines.append(
                f"  {r[cand_col]} ({r[party_col]}): {int(r[votes_col]):,} votes ({pct:.1f}%)"
            )
        winner = national.iloc[0]
        lines.append(
            f"Winner: {winner[cand_col]} of the {winner[party_col]} party with "
            f"{int(winner[votes_col]):,} votes ({winner[votes_col]/total_votes*100:.1f}% of votes counted)."
        )
        text = clean_text(" ".join(lines))
        summaries.append({
            "id": f"csv_summary_{year}",
            "source": "Ghana_Election_Result.csv",
            "text": text,
            "metadata": {"type": "national_summary", "year": str(year)},
        })
    return summaries


def ingest_csv(path: Path) -> list[dict]:
    """Each CSV row becomes one atomic document chunk, plus national summary chunks per year."""
    df = pd.read_csv(path, dtype=str).fillna("")
    chunks = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="CSV rows"):
        text = clean_text(" | ".join(f"{col}: {val}" for col, val in row.items() if val))
        if text:
            chunks.append({
                "id": f"csv_{idx}",
                "source": "Ghana_Election_Result.csv",
                "text": text,
                "metadata": {"row": idx},
            })
    summaries = build_election_summaries(df)
    print(f"  Built {len(summaries)} national summary chunks")
    return summaries + chunks


def sliding_window_chunks(words: list[str], size: int, overlap: int, base_id: str, source: str) -> list[dict]:
    chunks = []
    step = size - overlap
    for i in range(0, max(1, len(words) - overlap), step):
        window = words[i: i + size]
        if not window:
            break
        text = clean_text(" ".join(window))
        if text:
            chunks.append({
                "id": f"{base_id}_{i}",
                "source": source,
                "text": text,
                "metadata": {"word_offset": i},
            })
    return chunks


def ingest_pdf(path: Path) -> list[dict]:
    chunks = []
    with pdfplumber.open(path) as pdf:
        all_words = []
        for page in tqdm(pdf.pages, desc="PDF pages"):
            text = page.extract_text() or ""
            all_words.extend(text.split())
        chunks = sliding_window_chunks(
            all_words, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP,
            base_id="pdf", source=path.name
        )
    return chunks


def run_ingestion() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[1/4] Downloading datasets...")
    download_file(CSV_URL, CSV_PATH)
    try:
        download_file(PDF_URL, PDF_PATH)
    except Exception as e:
        if PDF_PATH.exists():
            print(f"  WARNING: PDF download incomplete, using partial file: {e}")
        else:
            print(f"  WARNING: Could not download PDF: {e}")
            print(f"  Please manually download the PDF and save it to: {PDF_PATH}")

    print("\n[2/4] Ingesting CSV...")
    csv_chunks = ingest_csv(CSV_PATH)
    print(f"  CSV chunks: {len(csv_chunks)}")

    print("\n[3/4] Ingesting PDF...")
    if PDF_PATH.exists():
        try:
            pdf_chunks = ingest_pdf(PDF_PATH)
            print(f"  PDF chunks: {len(pdf_chunks)}")
        except Exception as e:
            print(f"  WARNING: PDF ingestion failed: {e}")
            pdf_chunks = []
    else:
        print("  Skipping PDF (file not found). Add it manually to data/ and re-run.")
        pdf_chunks = []

    all_chunks = csv_chunks + pdf_chunks
    print(f"\n[4/4] Saving {len(all_chunks)} chunks to {CHUNKS_PATH}...")
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2))
    print("  Done.")
    return all_chunks


if __name__ == "__main__":
    run_ingestion()
