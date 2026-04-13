from __future__ import annotations

import re
import zipfile
from pathlib import Path


def _strip_xml(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    pptx_path = Path(__file__).resolve().parents[2] / "SAMPLE _REVIEW2_PPT FORMAT.pptx"
    if not pptx_path.exists():
        raise SystemExit(f"pptx not found: {pptx_path}")

    with zipfile.ZipFile(pptx_path) as z:
        slide_files = sorted(
            [n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")],
            key=lambda n: int(re.search(r"slide(\d+)", n).group(1)),
        )
        print(f"slides={len(slide_files)}")

        for name in slide_files:
            slide_num = int(re.search(r"slide(\d+)", name).group(1))
            xml = z.read(name).decode("utf-8", errors="ignore")
            # Extract text runs <a:t>...</a:t>
            texts = re.findall(r"<a:t[^>]*>(.*?)</a:t>", xml)
            # Decode common xml entities
            cleaned = [t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") for t in texts]
            # Heuristic: first 10 text runs show slide intent
            preview = " | ".join([c for c in cleaned if c.strip()][:12])
            preview = _strip_xml(preview)
            print(f"\n--- slide {slide_num} ---")
            print(preview)


if __name__ == "__main__":
    main()
