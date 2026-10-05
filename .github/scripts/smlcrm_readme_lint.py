#!/usr/bin/env python3
"""Check a repository's README.md and CITATION.cff against the Simulacrum
README standard (the simulacrum-readme skill).

Usage: lint_readme.py [REPO_DIR]

Standard library only, so it runs in CI with no install step. PyYAML is used
for CITATION.cff when present; otherwise a line-based check runs. Full CFF
schema validation is `cffconvert --validate`, which the workflow runs too.

Exit 0 when every check passes, 1 with one line per failure otherwise.
"""
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ORDER = ["header", "badges", "overview", "quickstart", "links", "citation", "license"]
# The template token pattern is case-sensitive even under re.I: BibTeX protects a
# one-word title or an entity author with double braces ({{Chronax}}), which is not
# a placeholder.
PLACEHOLDERS = [r"(?-i:\{\{\s*[A-Z][A-Z0-9_]*\s*\}\})", r"\bTODO\b", r"\bTBD\b", r"\bFIXME\b", r"lorem ipsum",
                r"\[path to", r"<placeholder", r"XXXX", r"\bREPLACE_ME\b", r"example\.com"]

errors = []


def fail(msg):
    errors.append(msg)


def section_blocks(text):
    found = {}
    for m in re.finditer(r"<!--\s*smlcrm:begin\s+([\w-]+)\s*-->", text):
        name = m.group(1)
        end = re.search(rf"<!--\s*smlcrm:end\s+{re.escape(name)}\s*-->", text[m.end():])
        if not end:
            fail(f"section '{name}': begin marker without a matching end marker")
            continue
        if name in found:
            fail(f"section '{name}' appears more than once")
        found[name] = (m.start(), text[m.end():m.end() + end.start()])
    return found


def check_header(body):
    if "<picture>" not in body:
        fail("header: no <picture> element for the light and dark logo")
    if not re.search(r'<source[^>]+media="\(prefers-color-scheme:\s*dark\)"[^>]+srcset="[^"]+"', body):
        fail('header: no <source media="(prefers-color-scheme: dark)" srcset="..."> for the dark logo')
    img = re.search(r"<img\b[^>]*>", body)
    if not img:
        fail("header: no <img> fallback inside <picture>")
    else:
        if not re.search(r'alt="[^"]+"', img.group(0)):
            fail("header: logo <img> has no alt text")
        if not re.search(r'src="[^"]+"', img.group(0)):
            fail("header: logo <img> has no src")
    if not re.search(r"<h1\b[^>]*>.+?</h1>|^# .+", body, re.M | re.S):
        fail("header: no repository name heading")
    if "https://smlcrm.com" not in body:
        fail("header: no link to https://smlcrm.com")


def check_badges(body):
    # Badges may be Markdown images or <img> tags; the alt text names the badge.
    imgs = re.findall(r"!\[([^\]]*)\]\(([^)\s]+)", body)
    imgs += [(a, s) for a, s in re.findall(r'<img\b[^>]*?alt="([^"]*)"[^>]*?src="([^"]+)"', body)]
    for label in ("License", "Release", "Language"):
        if not any(a.startswith(label) and urlparse(s).hostname == "img.shields.io" for a, s in imgs):
            fail(f"badges: no shields.io badge with alt text starting '{label}'")
    if not (any(a.startswith("Build") for a, _ in imgs) or "smlcrm:no-ci" in body):
        fail("badges: no 'Build' status badge and no <!-- smlcrm:no-ci --> note saying why")


def check_overview(body):
    if not re.search(r"^## Overview\s*$", body, re.M):
        fail("overview: no '## Overview' heading")
    if not re.search(r"does not|out of scope|not in scope", body, re.I):
        fail("overview: does not say what the repository does not do")


def check_quickstart(body):
    if not re.search(r"^## Quickstart\s*$", body, re.M):
        fail("quickstart: no '## Quickstart' heading")
    if body.count("```") < 4:
        fail("quickstart: needs a fenced install/run block and a fenced expected-output block")
    if not re.search(r"expected output", body, re.I):
        fail("quickstart: no 'Expected output'")
    tested = re.search(r"<!--\s*smlcrm:tested\s+\d{4}-\d{2}-\d{2}[^>]*-->", body)
    untested = re.search(r"\buntested\b", body, re.I)
    if not (tested or untested):
        fail("quickstart: neither a <!-- smlcrm:tested YYYY-MM-DD ... --> marker nor an 'Untested' label")


def check_links(body):
    if not re.search(r"^## Links\s*$", body, re.M):
        fail("links: no '## Links' heading")
    for needle, what in (("https://smlcrm.com", "the website"), ("/issues", "issues"), ("/releases", "releases")):
        if needle not in body:
            fail(f"links: no link to {what}")


def bib_fields(entry):
    return {k.lower(): v.strip() for k, v in re.findall(r"^\s*([A-Za-z]+)\s*=\s*\{(.*)\},?\s*$", entry, re.M)}


def check_citation(body):
    if not re.search(r"^## Citation\s*$", body, re.M):
        fail("citation: no '## Citation' heading")
    m = re.search(r"```bibtex\n(.*?)```", body, re.S)
    if not m:
        fail("citation: no ```bibtex block")
        return None
    entry = m.group(1)
    if not re.match(r"\s*@software\{[\w:\-]+,", entry):
        fail("citation: BibTeX entry is not '@software{key,'")
    if entry.count("{") != entry.count("}"):
        fail("citation: BibTeX braces are unbalanced")
    fields = bib_fields(entry)
    for f in ("title", "author", "year", "url"):
        if not fields.get(f):
            fail(f"citation: BibTeX field '{f}' missing or empty")
    if "version" not in fields and "smlcrm:no-version" not in body:
        fail("citation: no BibTeX 'version' and no <!-- smlcrm:no-version --> note")
    if fields.get("year") and not re.fullmatch(r"\d{4}", fields["year"]):
        fail("citation: BibTeX year is not four digits")
    if "doi" in fields and not re.fullmatch(r"10\.\d{4,9}/\S+", fields["doi"]):
        fail("citation: BibTeX doi is not a DOI; omit the field when there is none")
    return fields


def check_license(body):
    if not re.search(r"^## License and contact\s*$", body, re.M):
        fail("license: no '## License and contact' heading")
    if not re.search(r"mailto:[^)\s\"]+@|[\w.+-]+@[\w-]+\.[\w.]+", body):
        fail("license: no contact address")


_UNPARSED = object()


def load_cff(path):
    text = path.read_text()
    try:
        import yaml  # noqa: PLC0415
        return yaml.safe_load(text), text
    except ImportError:
        data = dict(re.findall(r"^([\w-]+):\s*(.*)$", text, re.M))
        data = {k: v.strip().strip("'\"") for k, v in data.items()}
        data["authors"] = ["?"] if re.search(r"^authors:\s*\n\s*-", text, re.M) else []
        return data, text
    except Exception as e:  # yaml.YAMLError
        fail(f"CITATION.cff: does not parse as YAML: {e}")
        return _UNPARSED, text


def check_cff(root, bib):
    path = root / "CITATION.cff"
    if not path.is_file():
        fail("CITATION.cff: missing")
        return
    data, text = load_cff(path)
    if data is _UNPARSED:
        return
    if not isinstance(data, dict):
        fail("CITATION.cff: empty or not a YAML mapping")
        return
    if str(data.get("cff-version")) != "1.2.0":
        fail("CITATION.cff: cff-version is not 1.2.0")
    for key in ("message", "title", "authors", "repository-code"):
        if not data.get(key):
            fail(f"CITATION.cff: '{key}' missing or empty")
    if "doi" in data and not re.fullmatch(r"10\.\d{4,9}/\S+", str(data["doi"])):
        fail("CITATION.cff: doi is not a DOI; omit the key when there is none")
    for pat in PLACEHOLDERS:
        if re.search(pat, text, re.I):
            fail(f"CITATION.cff: placeholder text matches /{pat}/")
    if bib:
        if bib.get("title") and str(data.get("title", "")).strip() != bib["title"].strip("{}"):
            fail("citation: BibTeX title and CITATION.cff title differ")
        if bib.get("version") and str(data.get("version", "")) != bib["version"]:
            fail("citation: BibTeX version and CITATION.cff version differ")
        if bib.get("url") and str(data.get("repository-code", "")).rstrip("/") != bib["url"].rstrip("/"):
            fail("citation: BibTeX url and CITATION.cff repository-code differ")


CHECKS = {"header": check_header, "badges": check_badges, "overview": check_overview,
          "quickstart": check_quickstart, "links": check_links, "license": check_license}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    readme = root / "README.md"
    if not readme.is_file():
        print("README.md: missing")
        sys.exit(1)
    text = readme.read_text()
    if re.search(r"<style\b", text, re.I) or re.search(r'\sstyle="', text, re.I):
        fail("README.md: inline CSS or <style>; GitHub strips it")
    found = section_blocks(text)
    present = [n for n in ORDER if n in found]
    for n in ORDER:
        if n not in found:
            fail(f"section '{n}': missing (<!-- smlcrm:begin {n} --> ... <!-- smlcrm:end {n} -->)")
    starts = [found[n][0] for n in present]
    if starts != sorted(starts):
        actual = sorted(present, key=lambda n: found[n][0])
        fail(f"sections out of order: {' > '.join(actual)}; required {' > '.join(ORDER)}")
    for name, (_, body) in found.items():
        if name not in ORDER:
            fail(f"section '{name}': not a managed section name")
            continue
        for pat in PLACEHOLDERS:
            if re.search(pat, body, re.I):
                fail(f"section '{name}': placeholder text matches /{pat}/")
    bib = None
    for name in present:
        body = found[name][1]
        if name == "citation":
            bib = check_citation(body)
        else:
            CHECKS[name](body)
    check_cff(root, bib)
    if errors:
        for e in errors:
            print(e)
        print(f"\n{len(errors)} problem(s). Fix them with the simulacrum-readme skill.")
        sys.exit(1)
    print("README.md and CITATION.cff meet the Simulacrum README standard.")


if __name__ == "__main__":
    main()
