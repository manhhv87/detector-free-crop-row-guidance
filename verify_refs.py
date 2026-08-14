#!/usr/bin/env python
"""
verify_refs.py -- check every entry in paper/refs.bib against the registration agencies.

A reference list is the one part of a manuscript a reader cannot check by reading. This resolves
each entry against the authorities that actually hold the record -- Crossref for published work,
DataCite for arXiv and dataset DOIs -- and reports whether the bibliography says what those
registries say.

For each entry:
  * with a DOI    -- fetch the record and compare title, first author, year, journal, volume, pages
  * without a DOI -- search Crossref by title and author and propose a DOI, but only when the title
                     matches closely enough to be the same work; anything weaker is reported as a
                     candidate for a human to confirm, never applied

Verdicts:
  OK          the registry record agrees with the bib entry
  MISMATCH    the DOI resolves, but a field disagrees (each disagreement is listed)
  FOUND       no DOI in the bib; a high-confidence record was located
  UNCERTAIN   a candidate exists but is below the match threshold -- check it yourself
  NO RECORD   nothing found; expected for books, technical reports and software

Two deviations are known and accepted, so a clean run still reports them:

  fischler1981  Crossref stores a shortened title ("Random sample consensus"); the entry
                carries the full published title. Venue, volume and pages all agree, so
                the bib is the more complete record and is left alone.
  ct3           Ultralytics YOLOv5 is cited as a repository, not by a Zenodo DOI: the
                concept DOI always resolves to the newest release, which is not the
                version the study ran.

Nothing is written unless you pass --write, and --write only applies FOUND entries.

Run:  python verify_refs.py            # report only
      python verify_refs.py --write    # additionally insert the confirmed DOIs
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

BIB = Path(__file__).resolve().parent / "paper" / "refs.bib"
MAILTO = "manhhv87@vnu.edu.vn"          # Crossref "polite pool" -- better rate limits
UA = f"verify_refs/1.0 (mailto:{MAILTO})"
TITLE_THRESHOLD = 0.90                   # below this, a search hit is only a candidate


# ----------------------------------------------------------------- bib parsing
def parse_bib(text):
    """Yield (key, type, {field: value}, raw) for each entry. Adequate for a hand-kept .bib."""
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.S):
        etype, key, body = m.group(1).lower(), m.group(2), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,\s*)?(?=\n\s*\w+\s*=|\s*$)", body, re.S):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split())
        yield key, etype, fields, m.group(0)


def norm(s):
    """Fold a title to something comparable: no LaTeX, no accents, no punctuation."""
    if not s:
        return ""
    s = s.replace("\\&", "and")
    # accent commands: \'e, \"o, \^a, \~n, \c{c} ... keep the letter, drop the accent
    s = re.sub(r"\\[`'^\"~=.uvHtcdb]\s*\{?([a-zA-Z])\}?", r"\1", s)
    for cmd, rep in (("\\AA", "A"), ("\\aa", "a"), ("\\O", "O"), ("\\o", "o"),
                     ("\\ss", "ss"), ("\\AE", "AE"), ("\\ae", "ae"),
                     ("\\OE", "OE"), ("\\oe", "oe"), ("\\L", "L"), ("\\l", "l")):
        s = s.replace(cmd + "{}", rep).replace(cmd + " ", rep).replace(cmd, rep)
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = s.replace("{", "").replace("}", "")
    s = s.translate(str.maketrans({
        "\u00f8": "o", "\u00d8": "O", "\u00e6": "ae", "\u00c6": "AE",
        "\u00df": "ss", "\u0111": "d", "\u0110": "D",
        "\u0142": "l", "\u0141": "L", "\u0153": "oe", "\u0152": "OE"}))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return " ".join(s.split())


def similar(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def first_surname(author_field):
    if not author_field:
        return ""
    first = author_field.split(" and ")[0]
    surname = first.split(",")[0] if "," in first else first.split()[-1]
    return norm(surname)


# ------------------------------------------------------------------ registries
def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception:
        return None


def crossref_by_doi(doi):
    d = get_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    return d["message"] if d else None


def datacite_by_doi(doi):
    d = get_json(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
    if not d:
        return None
    a = d["data"]["attributes"]
    return {"title": [a["titles"][0]["title"]] if a.get("titles") else [],
            "author": [{"family": c.get("familyName") or c.get("name", "")}
                       for c in a.get("creators", [])],
            "issued": {"date-parts": [[a.get("publicationYear")]]},
            "container-title": [a.get("publisher", "")], "_source": "DataCite"}


def crossref_search(title, author, year):
    q = urllib.parse.urlencode({"query.bibliographic": f"{title} {author}".strip(),
                                "rows": "5", "select": "DOI,title,author,issued,container-title,"
                                                       "volume,page,published-print,published-online"})
    d = get_json(f"https://api.crossref.org/works?{q}")
    return d["message"]["items"] if d and d.get("message", {}).get("items") else []


def rec_year(rec):
    """Registries carry several dates; the printed issue year is what a bibliography cites."""
    years = []
    for k in ("published-print", "issued", "published-online"):
        try:
            y = rec[k]["date-parts"][0][0]
            if y:
                years.append(int(y))
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    return years


# ------------------------------------------------------------------- comparing
def compare(fields, rec):
    """Return the list of disagreements between the bib entry and the registry record."""
    bad = []
    rt = (rec.get("title") or [""])[0]
    if rt and similar(fields.get("title", ""), rt) < 0.85:
        bad.append(f"title: bib '{fields.get('title','')[:55]}' vs registry '{rt[:55]}'")

    ra = rec.get("author") or []
    if ra and fields.get("author"):
        bib_first = first_surname(fields["author"])
        reg_names = [norm(a.get("family", "")) for a in ra if a.get("family")]
        same = bib_first and any(
            bib_first == r or bib_first in r.split() or r in bib_first.split()
            for r in reg_names)
        if reg_names and bib_first and not same:
            bad.append(f"author {bib_first!r} not among registry authors {reg_names[:4]}")

    ys = rec_year(rec)
    if ys and fields.get("year", "").isdigit() and int(fields["year"]) not in ys:
        bad.append(f"year: bib {fields['year']} vs registry {'/'.join(map(str, ys))}")

    rc = (rec.get("container-title") or [""])[0]
    if "arxiv" in norm(rc) and "arxiv" in norm(fields.get("journal", "")):
        rc = ""                      # both say arXiv; the exact wording does not matter
    if rc and fields.get("journal") and similar(fields["journal"], rc) < 0.80:
        bad.append(f"journal: bib '{fields['journal']}' vs registry '{rc}'")

    if rec.get("volume") and fields.get("volume") and str(rec["volume"]) != fields["volume"]:
        bad.append(f"volume: bib {fields['volume']} vs registry {rec['volume']}")

    if rec.get("page") and fields.get("pages"):
        bp = re.split(r"-+", fields["pages"])[0].strip()
        rp = re.split(r"-+", str(rec["page"]))[0].strip()
        if bp and rp and bp != rp:
            bad.append(f"first page: bib {bp} vs registry {rp}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="insert the DOIs of FOUND entries into refs.bib")
    ap.add_argument("--bib", default=str(BIB))
    args = ap.parse_args()

    text = Path(args.bib).read_text()
    entries = list(parse_bib(text))
    print(f"# {len(entries)} entries in {args.bib}\n")

    counts = {k: 0 for k in ("OK", "MISMATCH", "FOUND", "UNCERTAIN", "NO RECORD")}
    to_write = {}

    for key, etype, f, _ in entries:
        doi = f.get("doi", "").strip()
        title = f.get("title", "")
        label = f"{key} [{etype}]"

        if doi:
            rec = crossref_by_doi(doi) or datacite_by_doi(doi)
            if not rec:
                print(f"  MISMATCH   {label}\n             DOI {doi} does not resolve at Crossref or DataCite")
                counts["MISMATCH"] += 1
            else:
                bad = compare(f, rec)
                if bad:
                    print(f"  MISMATCH   {label}   doi:{doi}")
                    for b in bad:
                        print(f"             - {b}")
                    counts["MISMATCH"] += 1
                else:
                    print(f"  OK         {label}   doi:{doi}")
                    counts["OK"] += 1
        else:
            hits = crossref_search(title, first_surname(f.get("author", "")), f.get("year", ""))
            best, score = None, 0.0
            for h in hits:
                s = similar(title, (h.get("title") or [""])[0])
                if s > score:
                    best, score = h, s
            if best and score >= TITLE_THRESHOLD:
                # The same title can exist as a journal article, a book chapter and a
                # preprint with three different DOIs. Only accept a candidate whose
                # venue, volume, pages and year all agree with what the entry cites.
                bad = compare(f, best)
                if not bad:
                    print(f"  FOUND      {label}   doi:{best['DOI']}  (title match {score:.2f})")
                    to_write[key] = best["DOI"]
                    counts["FOUND"] += 1
                else:
                    print(f"  UNCERTAIN  {label}   candidate doi:{best['DOI']} (title {score:.2f}) "
                          f"but the record is a different publication:")
                    for bmsg in bad:
                        print(f"             - {bmsg}")
                    counts["UNCERTAIN"] += 1
            elif best:
                print(f"  UNCERTAIN  {label}   best candidate doi:{best['DOI']} (title match only {score:.2f})")
                print(f"             registry title: {(best.get('title') or [''])[0][:75]}")
                counts["UNCERTAIN"] += 1
            else:
                print(f"  NO RECORD  {label}   '{title[:60]}'")
                counts["NO RECORD"] += 1
        time.sleep(0.4)          # stay well inside Crossref's polite rate

    print("\n" + "=" * 74)
    print("  " + "   ".join(f"{k}: {v}" for k, v in counts.items()))

    if to_write:
        if args.write:
            out = text
            for key, doi in to_write.items():
                out = re.sub(r"(@\w+\s*\{\s*" + re.escape(key) + r"\s*,)",
                             r"\1\n  doi       = {" + doi + "},", out, count=1)
            Path(args.bib).write_text(out)
            print(f"\n  wrote {len(to_write)} DOIs into {args.bib}")
        else:
            print(f"\n  {len(to_write)} DOIs ready to insert; re-run with --write to apply")

    return 1 if counts["MISMATCH"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
