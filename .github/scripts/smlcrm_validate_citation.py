#!/usr/bin/env python3
"""Prove the citation is valid: the README's BibTeX entry parses and formats
with pybtex (what "compiles" means without a TeX install), and CITATION.cff
passes `cffconvert --validate` against the CFF 1.2.0 schema.

Usage: validate_citation.py [REPO_DIR]
Needs: pip install pybtex cffconvert   (the CI workflow installs both)

Exit 0 when both pass, 1 otherwise, 3 when a tool is not installed.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    ok = True
    try:
        from pybtex.database import parse_string  # noqa: PLC0415
        from pybtex.plugin import find_plugin  # noqa: PLC0415
    except ImportError:
        print("pybtex is not installed: python3 -m pip install pybtex")
        sys.exit(3)
    text = (root / "README.md").read_text()
    # Validate the managed citation, not an older hand-written block elsewhere.
    sec = re.search(r"<!--\s*smlcrm:begin citation\s*-->(.*?)<!--\s*smlcrm:end citation\s*-->", text, re.S)
    m = re.search(r"```bibtex\n(.*?)```", sec.group(1) if sec else text, re.S)
    if not m:
        print("BibTeX: no ```bibtex block in the README citation section")
        sys.exit(1)
    try:
        db = parse_string(m.group(1), "bibtex")
        (key, entry), = db.entries.items()
        if entry.type != "software":
            raise ValueError(f"entry type is @{entry.type}, not @software")
        style = find_plugin("pybtex.style.formatting", "plain")()
        # plain has no @software formatter; format as @misc to prove the fields render
        entry.type = "misc"
        rendered = next(iter(style.format_entries(db.entries.values()))).text.render_as("text")
        print(f"BibTeX: @software{{{key}}} parses and renders: {rendered}")
    except Exception as e:
        print(f"BibTeX: does not compile: {e}")
        ok = False
    # Prefer the cffconvert next to this interpreter, so a venv works unactivated.
    local = Path(sys.executable).parent / "cffconvert"
    cff = str(local) if local.exists() else shutil.which("cffconvert")
    if not cff:
        print("cffconvert is not installed: python3 -m pip install cffconvert")
        sys.exit(3)
    r = subprocess.run([cff, "--validate", "-i", str(root / "CITATION.cff")], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print(f"CITATION.cff: {out.splitlines()[-1] if out else 'no output'}")
    if r.returncode != 0:
        print(out)
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
