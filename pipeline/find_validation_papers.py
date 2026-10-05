"""Query PMC for open-access papers that cross-reference tables and figures.

Outputs candidates.csv with metadata for papers likely to present the same
data in both tabular and graphical form.
"""

import csv
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import CANDIDATES_PATH  # noqa: E402

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DELAY = 0.12  # NCBI allows 3 req/s without API key
TARGET = 3000  # need large pool; many will be filtered by plot type

CROSS_REF = re.compile(
    r"(Table\s+\d+.{0,80}?Figure\s+\d+)"
    r"|(Figure\s+\d+.{0,80}?Table\s+\d+)",
    re.IGNORECASE,
)

# Restrict to MEDLINE-indexed biomedical journals (Nature, Cell, Science,
# JAMA, Lancet, BMJ, NEJM, etc.) — excludes engineering/CS/materials.
_LIFE_SCI = "medline[sb]"
_OA = "AND open access[filter]"

QUERIES = [
    f'("summarized in Table" AND "shown in Figure") AND {_LIFE_SCI} {_OA}',
    f'("presented in Table" AND "depicted in Figure") AND {_LIFE_SCI} {_OA}',
    f'("Table 1" AND "Figure 1" AND ("plotted" OR "depicted" OR "illustrated"))'
    f" AND {_LIFE_SCI} {_OA}",
    f'("Table 2" AND "Figure 2") AND {_LIFE_SCI} {_OA}',
    f'"systematic review"[Title] AND "forest plot" AND "Table"'
    f" AND {_LIFE_SCI} {_OA}",
    f'"meta-analysis"[Title] AND "Table" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'"randomized controlled trial"[Publication Type] AND "Table 1"'
    f' AND "Figure 1" {_OA}',
    f'("biomarker" OR "C-reactive protein" OR "interleukin")'
    f' AND "Table" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'("cohort study" AND "Table" AND "Figure"'
    f' AND ("bar chart" OR "box plot" OR "line graph")) AND {_LIFE_SCI} {_OA}',
    f'("data are shown in Table" AND "Figure") AND {_LIFE_SCI} {_OA}',
    f'("see Table" AND "see Figure") AND {_LIFE_SCI} {_OA}',
    # Targeted queries for specific life-science plot types
    f'"Kaplan-Meier" AND "Table" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'"forest plot" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"ROC curve" AND "Table" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'"box plot" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"bar chart" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"violin plot" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"scatter plot" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"heatmap" AND "Table" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'"funnel plot" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"Bland-Altman" AND "Table" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'"histogram" AND "Table" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'("dose-response" OR "dose response") AND "Table" AND "Figure"'
    f' AND {_LIFE_SCI} {_OA}',
    f'"waterfall plot" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"volcano plot" AND "Table" AND {_LIFE_SCI} {_OA}',
    # Extra queries for underrepresented types — duplicate keywords to
    # pull in more papers that explicitly mention these plot types
    f'"histogram" AND "Table 1" AND {_LIFE_SCI} {_OA}',
    f'"histogram" AND "Table 2" AND {_LIFE_SCI} {_OA}',
    f'"histogram" AND "frequency distribution" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"funnel plot" AND "publication bias" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"funnel plot" AND "meta-analysis" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"violin plot" AND "Figure" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"box plot" AND "Figure" AND "Table 1" AND {_LIFE_SCI} {_OA}',
    f'"box plot" AND "Figure" AND "Table 2" AND {_LIFE_SCI} {_OA}',
    f'"box-and-whisker" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"boxplot" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"scatter plot" AND "Figure" AND "Table 1" AND {_LIFE_SCI} {_OA}',
    f'"scatter plot" AND "correlation" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"Kaplan-Meier" AND "Table 1" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'"Kaplan-Meier" AND "Table 2" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'"survival curve" AND "Table" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'"forest plot" AND "Table 1" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'"forest plot" AND "Table 2" AND "Figure" AND {_LIFE_SCI} {_OA}',
    f'"bar graph" AND "Table" AND {_LIFE_SCI} {_OA}',
    f'"line graph" AND "Table" AND {_LIFE_SCI} {_OA}',
]

_REPLACE = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2212": "-"})


def normalise(text: str) -> str:
    return text.translate(_REPLACE)


def collect_ids() -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for qi, q in enumerate(QUERIES):
        print(f"Query {qi + 1}/{len(QUERIES)}: {q[:90]}...")
        try:
            r = requests.get(
                f"{NCBI}/esearch.fcgi",
                params={"db": "pmc", "term": q, "retmax": 200, "retmode": "json",
                        "sort": "date"},
                timeout=30,
            )
            r.raise_for_status()
            for x in r.json()["esearchresult"]["idlist"]:
                if x not in seen:
                    seen.add(x)
                    ids.append(x)
        except Exception as e:  # noqa: BLE001
            print(f"  search error: {e}")
        time.sleep(DELAY)
    return ids


BATCH_SIZE = 10  # IDs per efetch call (~10x faster than one-by-one)


def _process_article(root: ET.Element, pmcid: str) -> dict | None:
    """Extract candidate info from a single article XML element."""
    tables = root.findall(".//table-wrap")
    figs = root.findall(".//fig")
    if not tables or not figs:
        return None

    body = root.find(".//body")
    if body is None:
        return None
    txt = normalise(ET.tostring(body, encoding="unicode", method="text"))

    matches = [m for tup in CROSS_REF.findall(txt) for m in tup if m]

    # Also check table/figure captions for cross-references
    if not matches:
        for cap_parent in tables + figs:
            cap_el = cap_parent.find("caption")
            label_el = cap_parent.find("label")
            if cap_el is None:
                continue
            cap_txt = normalise(
                ET.tostring(cap_el, encoding="unicode", method="text")
            )
            label_txt = normalise(
                ET.tostring(label_el, encoding="unicode", method="text")
            ) if label_el is not None else ""
            combined = label_txt + " " + cap_txt
            cap_matches = [
                m for tup in CROSS_REF.findall(combined) for m in tup if m
            ]
            matches.extend(cap_matches)

    if not matches:
        return None

    title_el = root.find(".//article-title")
    title = (
        normalise(ET.tostring(title_el, encoding="unicode", method="text").strip())
        if title_el is not None else ""
    )
    doi_el = root.find(".//article-id[@pub-id-type='doi']")
    doi = doi_el.text.strip() if doi_el is not None and doi_el.text else ""

    return {
        "pmcid": f"PMC{pmcid}",
        "doi": doi,
        "title": title,
        "n_tables": len(tables),
        "n_figures": len(figs),
        "cross_refs": len(matches),
        "example_ref": matches[0][:120],
        "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
    }


def filter_candidates(ids: list[str], n_existing: int = 0) -> list[dict]:
    results: list[dict] = []
    n_processed = 0
    for batch_start in range(0, len(ids), BATCH_SIZE):
        if n_existing + len(results) >= TARGET:
            break
        batch = ids[batch_start:batch_start + BATCH_SIZE]
        try:
            r = requests.get(
                f"{NCBI}/efetch.fcgi",
                params={"db": "pmc", "id": ",".join(batch), "retmode": "xml"},
                timeout=30,
            )
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for article in root.findall(".//article"):
                pmcid_el = article.find(".//article-id[@pub-id-type='pmcaid']")
                if pmcid_el is None or not pmcid_el.text:
                    continue
                pmcid = pmcid_el.text.strip()
                result = _process_article(article, pmcid)
                if result:
                    results.append(result)
                    title = result["title"][:70].encode("ascii", "replace").decode()
                    print(f"  [{len(results):3d}] {result['pmcid']}: {title}")
        except Exception:  # noqa: BLE001
            # Fall back to one-by-one for failed batches
            for pmcid in batch:
                try:
                    r2 = requests.get(
                        f"{NCBI}/efetch.fcgi",
                        params={"db": "pmc", "id": pmcid, "retmode": "xml"},
                        timeout=15,
                    )
                    r2.raise_for_status()
                    art = ET.fromstring(r2.content)
                    result = _process_article(art, pmcid)
                    if result:
                        results.append(result)
                        title = result["title"][:70].encode("ascii", "replace").decode()
                        print(f"  [{len(results):3d}] {result['pmcid']}: {title}")
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(DELAY)

        n_processed += len(batch)
        if n_processed % 100 < BATCH_SIZE:
            print(f"  ... {n_processed}/{len(ids)} processed, {len(results)} hits")
        time.sleep(DELAY)
    return results


def main() -> None:
    out = CANDIDATES_PATH

    existing: list[dict] = []
    existing_pmcids: set[str] = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.append(row)
                existing_pmcids.add(row["pmcid"])
        print(f"Loaded {len(existing)} existing candidates, "
              f"need {TARGET - len(existing)} more\n")

    if len(existing) >= TARGET:
        print(f"Already at target ({len(existing)} >= {TARGET}), nothing to do.")
        return

    ids = collect_ids()
    ids = [i for i in ids if f"PMC{i}" not in existing_pmcids]
    print(f"\nCollected {len(ids)} new candidate IDs\n")

    new_results = filter_candidates(ids, n_existing=len(existing))
    all_results = existing + new_results

    if new_results:
        with open(out, "w", newline="", encoding="utf-8") as f:
            fieldnames = list((existing or new_results)[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)

    print(f"\nDone: {len(all_results)} papers saved to {out} ({len(new_results)} new)")
    if len(all_results) < TARGET:
        print(f"(Below target of {TARGET})")


if __name__ == "__main__":
    main()
