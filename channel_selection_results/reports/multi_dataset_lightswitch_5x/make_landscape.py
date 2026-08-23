#!/usr/bin/env python3
"""Set every DOCX section to landscape with compact margins."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(name: str) -> str:
    return f"{{{W}}}{name}"


def main() -> None:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(members["word/document.xml"])
    for section in root.iter(q("sectPr")):
        page = section.find(q("pgSz"))
        if page is None:
            page = ET.SubElement(section, q("pgSz"))
        page.set(q("w"), "16838")
        page.set(q("h"), "11906")
        page.set(q("orient"), "landscape")
        margins = section.find(q("pgMar"))
        if margins is None:
            margins = ET.SubElement(section, q("pgMar"))
        margins.set(q("top"), "720")
        margins.set(q("bottom"), "720")
        margins.set(q("left"), "620")
        margins.set(q("right"), "620")
    ET.register_namespace("w", W)
    members["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


if __name__ == "__main__":
    main()
