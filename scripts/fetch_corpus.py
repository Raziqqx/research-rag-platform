"""Build or download the openly licensed research corpus.

Two modes:

  python scripts/fetch_corpus.py              # download every PDF in corpus/manifest.json
                                              # into $RAG_PDF_DIR and verify sha256
  python scripts/fetch_corpus.py --discover   # (re)build the manifest from the sources

Discovery only keeps documents whose licence is verified on the source's own record:
  * World Bank Open Knowledge Repository: the item's dc.rights.uri (the DSpace record behind
    the item page) must be CC BY 3.0 IGO, and the item must be a Working Paper.
  * arXiv q-fin: the abs page (https://arxiv.org/abs/<id>) must link the CC BY 4.0 licence.
Anything that can't be confirmed is dropped. PDFs themselves are never committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pymupdf
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import get_settings  # noqa: E402

UA = {"User-Agent": "research-rag-platform corpus fetcher (non-commercial research)"}
WB_API = "https://openknowledge.worldbank.org/server/api"
# Attribution-only Creative Commons licences used by the OKR (non-commercial variants are rejected).
WB_LICENCES = {
    "https://creativecommons.org/licenses/by/3.0/igo/": "CC BY 3.0 IGO",
    "https://creativecommons.org/licenses/by/3.0/igo": "CC BY 3.0 IGO",
    "https://creativecommons.org/licenses/by/4.0/igo/": "CC BY 4.0 IGO",
    "https://creativecommons.org/licenses/by/4.0/igo": "CC BY 4.0 IGO",
}
ARXIV_OAI = "https://oaipmh.arxiv.org/oai"
ARXIV_LICENCE = "http://creativecommons.org/licenses/by/4.0/"
MIN_PAGES, MAX_PAGES = 8, 45

WB_QUERIES = [
    "inflation monetary policy",
    "sovereign debt",
    "financial inclusion",
    "climate finance",
    "remittances",
    "exchange rate",
    "banking supervision",
    "fiscal policy",
    "productivity growth",
    "trade tariffs",
    "household consumption poverty",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    resp = requests.get(url, headers=UA, timeout=120)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError(f"{url} did not return a PDF")
    dest.write_bytes(resp.content)


def page_count(path: Path) -> int:
    with pymupdf.open(path) as doc:
        return doc.page_count


# --------------------------------------------------------------------------- World Bank
def wb_candidates(query: str, size: int = 25) -> list[dict]:
    resp = requests.get(
        f"{WB_API}/discover/search/objects",
        params={"query": query, "dsoType": "ITEM", "size": size, "sort": "dc.date.issued,DESC"},
        headers=UA,
        timeout=60,
    )
    resp.raise_for_status()
    objs = resp.json()["_embedded"]["searchResult"]["_embedded"]["objects"]
    return [o["_embedded"]["indexableObject"] for o in objs]


def wb_pdf_url(item_uuid: str) -> str | None:
    bundles = requests.get(f"{WB_API}/core/items/{item_uuid}/bundles", headers=UA, timeout=60)
    bundles.raise_for_status()
    for b in bundles.json()["_embedded"]["bundles"]:
        if b["name"] != "ORIGINAL":
            continue
        bits = requests.get(b["_links"]["bitstreams"]["href"], headers=UA, timeout=60).json()
        for bit in bits["_embedded"]["bitstreams"]:
            if bit["name"].lower().endswith(".pdf"):
                return bit["_links"]["content"]["href"]
    return None


def discover_world_bank(want: int, pdf_dir: Path) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for query in WB_QUERIES:
        for item in wb_candidates(query):
            if len(out) >= want:
                return out
            m = item["metadata"]
            meta = lambda k: [v["value"] for v in m.get(k, [])]  # noqa: E731
            if item["uuid"] in seen or "Working Paper" not in meta("dc.type"):
                continue
            seen.add(item["uuid"])
            # Licence check against the repository's own record for this item.
            lic_uri = next((u for u in meta("dc.rights.uri") if u in WB_LICENCES), None)
            if lic_uri is None:
                print(f"  drop {item['handle']}: licence {meta('dc.rights.uri')}")
                continue
            url = wb_pdf_url(item["uuid"])
            if not url:
                continue
            doc_id = "wb-" + item["handle"].replace("/", "-")
            dest = pdf_dir / f"{doc_id}.pdf"
            download(url, dest)
            n = page_count(dest)
            if not MIN_PAGES <= n <= MAX_PAGES:
                dest.unlink()
                continue
            out.append(
                {
                    "doc_id": doc_id,
                    "source": "World Bank Open Knowledge Repository",
                    "title": meta("dc.title")[0],
                    "authors": meta("dc.contributor.author"),
                    "published": meta("dc.date.issued")[0],
                    "doc_type": "working_paper",
                    "topic_query": query,
                    "url": url,
                    "landing_page": f"https://openknowledge.worldbank.org/handle/{item['handle']}",
                    "licence": WB_LICENCES[lic_uri],
                    "licence_url": lic_uri,
                    "licence_evidence": f"{WB_API}/core/items/{item['uuid']} (dc.rights.uri)",
                    "pages": n,
                    "sha256": sha256_of(dest),
                }
            )
            print(f"  keep {doc_id} ({n} pages): {out[-1]['title'][:70]}")
            break  # one paper per topic query keeps the corpus topically diverse
    return out


# --------------------------------------------------------------------------- arXiv
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}


def arxiv_records(date_from: str, date_until: str) -> list[dict]:
    resp = requests.get(
        ARXIV_OAI,
        params={
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "set": "q-fin",
            "from": date_from,
            "until": date_until,
        },
        headers=UA,
        timeout=120,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    recs = []
    for rec in root.iter(f"{{{NS['oai']}}}record"):
        md = rec.find(".//a:arXiv", NS)
        if md is None:
            continue
        get = lambda tag: (md.findtext(f"a:{tag}", default="", namespaces=NS) or "").strip()  # noqa: E731
        authors = [
            " ".join(
                filter(None, [a.findtext("a:forenames", "", NS), a.findtext("a:keyname", "", NS)])
            )
            for a in md.findall("a:authors/a:author", NS)
        ]
        recs.append(
            {
                "id": get("id"),
                "title": re.sub(r"\s+", " ", get("title")),
                "categories": get("categories").split(),
                "licence": get("license"),
                "created": get("created"),
                "authors": authors,
            }
        )
    return recs


def arxiv_abs_licence_ok(arxiv_id: str) -> tuple[bool, str]:
    """Confirm CC BY 4.0 from the abs page itself; returns (ok, latest version id)."""
    html = requests.get(f"https://arxiv.org/abs/{arxiv_id}", headers=UA, timeout=60).text
    ok = "creativecommons.org/licenses/by/4.0" in html
    versions = re.findall(rf"/abs/{re.escape(arxiv_id)}(v\d+)", html)
    latest = max(versions, key=lambda v: int(v[1:])) if versions else "v1"
    return ok, arxiv_id + latest


def discover_arxiv(want: int, pdf_dir: Path) -> list[dict]:
    out: list[dict] = []
    per_cat: dict[str, int] = {}
    recs = arxiv_records("2025-03-01", "2025-03-20")
    for r in recs:
        if len(out) >= want:
            break
        primary = next((c for c in r["categories"] if c.startswith("q-fin")), None)
        if r["licence"] != ARXIV_LICENCE or primary is None or per_cat.get(primary, 0) >= 2:
            continue
        ok, versioned = arxiv_abs_licence_ok(r["id"])
        time.sleep(1)  # be polite to arXiv
        if not ok:
            print(f"  drop {r['id']}: abs page does not show CC BY 4.0")
            continue
        doc_id = "arxiv-" + r["id"].replace("/", "_")
        dest = pdf_dir / f"{doc_id}.pdf"
        url = f"https://arxiv.org/pdf/{versioned}"
        try:
            download(url, dest)
        except (requests.HTTPError, ValueError) as exc:
            print(f"  skip {r['id']}: {exc}")
            continue
        n = page_count(dest)
        if not MIN_PAGES <= n <= MAX_PAGES:
            dest.unlink()
            continue
        per_cat[primary] = per_cat.get(primary, 0) + 1
        out.append(
            {
                "doc_id": doc_id,
                "source": "arXiv",
                "title": r["title"],
                "authors": r["authors"],
                "published": r["created"],
                "doc_type": "preprint",
                "topic_query": primary,
                "url": url,
                "landing_page": f"https://arxiv.org/abs/{versioned}",
                "licence": "CC BY 4.0",
                "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                "licence_evidence": f"https://arxiv.org/abs/{r['id']} (licence link on abs page)",
                "categories": r["categories"],
                "pages": n,
                "sha256": sha256_of(dest),
            }
        )
        print(f"  keep {doc_id} ({n} pages, {primary}): {r['title'][:70]}")
    return out


# --------------------------------------------------------------------------- main
def fetch_from_manifest(manifest: list[dict], pdf_dir: Path, strict: bool) -> int:
    bad = 0
    for d in manifest:
        dest = pdf_dir / f"{d['doc_id']}.pdf"
        if not dest.exists():
            print(f"download {d['doc_id']}")
            download(d["url"], dest)
        digest = sha256_of(dest)
        if digest != d["sha256"]:
            bad += 1
            print(f"SHA256 MISMATCH {d['doc_id']}: {digest} != {d['sha256']}")
    print(f"{len(manifest) - bad}/{len(manifest)} PDFs verified in {pdf_dir}")
    return 1 if (bad and strict) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--discover", action="store_true", help="rebuild manifest from sources")
    ap.add_argument("--n-wb", type=int, default=8)
    ap.add_argument("--n-arxiv", type=int, default=12)
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    s.pdf_dir.mkdir(parents=True, exist_ok=True)
    if args.discover:
        print("World Bank OKR:")
        docs = discover_world_bank(args.n_wb, s.pdf_dir)
        print("arXiv q-fin:")
        docs += discover_arxiv(args.n_arxiv, s.pdf_dir)
        s.manifest_path.write_text(json.dumps(docs, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {len(docs)} documents to {s.manifest_path}")
        return 0
    manifest = json.loads(s.manifest_path.read_text())
    return fetch_from_manifest(manifest, s.pdf_dir, strict=not args.allow_hash_mismatch)


if __name__ == "__main__":
    raise SystemExit(main())
