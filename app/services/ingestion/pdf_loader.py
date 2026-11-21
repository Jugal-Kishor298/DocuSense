import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pymupdf  # PyMuPDF parser

# Add workspace root to sys.path if running as main script
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _column_edges_consistent(table_rows: List[Dict[str, Any]], col_idx: int, x_tolerance: float = 15.0) -> bool:
    """Checks if a specific column index across all candidate rows is internally consistent
    in either left-alignment (x0) OR right-alignment (x1).
    """
    x0s = [r["spans"][col_idx]["x0"] for r in table_rows if len(r["spans"]) > col_idx]
    x1s = [r["spans"][col_idx]["x1"] for r in table_rows if len(r["spans"]) > col_idx]

    if len(x0s) < len(table_rows):
        return False

    x0_aligned = (max(x0s) - min(x0s)) <= x_tolerance
    x1_aligned = (max(x1s) - min(x1s)) <= x_tolerance
    return x0_aligned or x1_aligned


def _rows_have_consistent_columns(table_rows: List[Dict[str, Any]], x_tolerance: float = 15.0) -> bool:
    """Validates that a candidate table's rows share a consistent column count and
    that column edges (x0 left-aligned or x1 right-aligned) align across rows for all data columns beyond the first.
    """
    span_counts = [len(r["spans"]) for r in table_rows]
    # Reject wildly inconsistent column counts (bullet outlines have variable span counts per line)
    if max(span_counts) - min(span_counts) > 1:
        return False

    # Reject if any cell looks like prose rather than a table cell
    # (long text, or text ending in ':' — a bullet header, not tabular content)
    for r in table_rows:
        for s in r["spans"]:
            txt = s["text"].strip()
            if len(txt) > 40 or txt.endswith(":"):
                return False

    # Determine minimum column count across candidate rows
    min_cols = min(span_counts)
    if min_cols < 2:
        return False

    # Require all data columns beyond the first (col_idx >= 1) to be internally consistent (x0 or x1 alignment)
    for col_idx in range(1, min_cols):
        if not _column_edges_consistent(table_rows, col_idx, x_tolerance=x_tolerance):
            return False

    return True


def _detect_fallback_borderless_tables(page: pymupdf.Page) -> Tuple[List[str], List[pymupdf.Rect]]:
    """Fallback pass: detects borderless tables using page dict span layout, orphan single-span row merging,
    and multi-column alignment heuristics without incremental row-by-row fragmentation.
    """
    try:
        page_dict = page.get_text("dict")
        raw_spans = []

        for b in page_dict.get("blocks", []):
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                for s in line.get("spans", []):
                    txt = s.get("text", "").strip()
                    if txt:
                        raw_spans.append({
                            "text": txt,
                            "x0": round(s["bbox"][0], 1),
                            "x1": round(s["bbox"][2], 1),
                            "y0": round(s["bbox"][1], 1),
                            "y1": round(s["bbox"][3], 1),
                        })

        if not raw_spans:
            return [], []

        raw_spans.sort(key=lambda s: (s["y0"], s["x0"]))

        # Step 1: Group spans into visual rows by matching y0 (within 3.5 points)
        visual_rows = []
        for s in raw_spans:
            matched = False
            for row in visual_rows:
                if abs(row["y0"] - s["y0"]) <= 3.5:
                    row["spans"].append(s)
                    row["spans"].sort(key=lambda x: x["x0"])
                    row["y0"] = min(row["y0"], s["y0"])
                    matched = True
                    break
            if not matched:
                visual_rows.append({"y0": s["y0"], "spans": [s]})

        # Step 2: Merge single-span orphan rows into an adjacent visual row if y0 is within 7.0 points (2x tolerance)
        merged_rows = []
        for r in visual_rows:
            if len(r["spans"]) == 1 and merged_rows:
                last_row = merged_rows[-1]
                if abs(r["y0"] - last_row["y0"]) <= 7.0:
                    last_row["spans"].extend(r["spans"])
                    last_row["spans"].sort(key=lambda x: x["x0"])
                    continue
            merged_rows.append(r)

        merged_rows.sort(key=lambda r: r["y0"])

        # Step 3: Collect all table candidate row groups on the page by y-proximity (gap <= 60.0 points)
        # single-span rows are ignored rather than acting as hard table-breaking boundaries
        table_candidates = []
        current_candidate = []

        for r in merged_rows:
            if len(r["spans"]) < 2:
                continue

            if not current_candidate:
                current_candidate.append(r)
            else:
                prev_row = current_candidate[-1]
                if (r["y0"] - prev_row["y0"]) <= 60.0:
                    current_candidate.append(r)
                else:
                    table_candidates.append(current_candidate)
                    current_candidate = [r]

        if current_candidate:
            table_candidates.append(current_candidate)

        # Step 4: Validate each candidate row group (>= 3 rows and column consistency)
        tables_md = []
        table_rects = []

        for cand in table_candidates:
            if len(cand) >= 3 and _rows_have_consistent_columns(cand, x_tolerance=20.0):
                md, rect = _build_markdown_table_from_rows(cand)
                tables_md.append(md)
                table_rects.append(rect)

        return tables_md, table_rects
    except Exception:
        return [], []


def _build_markdown_table_from_rows(table_rows: List[Dict[str, Any]]) -> Tuple[str, pymupdf.Rect]:
    rows_text = []
    x0_min = min(min(s["x0"] for s in r["spans"]) for r in table_rows)
    y0_min = min(r["y0"] for r in table_rows)
    x1_max = max(max(s["x1"] for s in r["spans"]) for r in table_rows)
    y1_max = max(max(s["y1"] for s in r["spans"]) for r in table_rows)

    header_spans = [s["text"] for s in table_rows[0]["spans"]]
    header_str = "| " + " | ".join(header_spans) + " |"
    sep_str = "| " + " | ".join(["---"] * len(header_spans)) + " |"

    rows_text.append(header_str)
    rows_text.append(sep_str)

    for r in table_rows[1:]:
        cell_values = [s["text"] for s in r["spans"]]
        row_str = "| " + " | ".join(cell_values) + " |"
        rows_text.append(row_str)

    table_md = "\n".join(rows_text)
    rect = pymupdf.Rect(x0_min, y0_min, x1_max, y1_max)
    return table_md, rect


def _extract_text_and_tables_from_page(page: pymupdf.Page, dpi: int = 300) -> Tuple[str, List[str]]:
    """Extracts normal text and Markdown tables from a single PDF page.
    Table bounding box (bbox) regions are excluded from normal text extraction to avoid content duplication.
    Applies vector find_tables() first and layout span alignment fallback for borderless tables.

    Returns:
        Tuple[str, List[str]]: (normal_text, list_of_markdown_tables)
    """
    table_markdowns = []
    table_rects = []

    try:
        tabs = page.find_tables()
        if tabs and tabs.tables:
            for t in tabs.tables:
                table_rects.append(pymupdf.Rect(t.bbox))
                table_markdowns.append(t.to_markdown())
    except Exception:
        table_markdowns = []
        table_rects = []

    # Fallback pass for borderless tables if no vector tables detected
    if not table_markdowns:
        fb_mds, fb_rects = _detect_fallback_borderless_tables(page)
        if fb_mds:
            table_markdowns.extend(fb_mds)
            table_rects.extend(fb_rects)

    blocks = page.get_text("blocks", sort=True)
    text_blocks = []

    for b in blocks:
        if len(b) < 5 or not b[4].strip():
            continue

        b_rect = pymupdf.Rect(b[:4])
        b_area = max(0.0, b_rect.width * b_rect.height)

        # Check if block overlaps significantly (>40%) with any table bbox
        is_in_table = False
        for t_rect in table_rects:
            intersection = b_rect & t_rect
            if not intersection.is_empty:
                inter_area = max(0.0, intersection.width * intersection.height)
                if b_area > 0 and (inter_area / b_area) > 0.4:
                    is_in_table = True
                    break

        if not is_in_table:
            text_blocks.append(b[4].strip())

    text = "\n\n".join(text_blocks)

    if (len(text.strip()) < 30 and not table_markdowns) or (not text and not table_markdowns):
        # OCR Fallback for image-heavy slides / scanned pages
        ocr_text = ""
        # 1. Try PyMuPDF native OCR engine first
        try:
            textpage_ocr = page.get_textpage_ocr(full=True, dpi=dpi)
            ocr_text = textpage_ocr.extractText().strip()
        except Exception:
            ocr_text = ""

        # 2. Try RapidOCR fallback if PyMuPDF native OCR returned no text
        if not ocr_text:
            try:
                from rapidocr_onnxruntime import RapidOCR
                import numpy as np
                import cv2

                pix = page.get_pixmap(dpi=dpi)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                elif pix.n == 1:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                engine = RapidOCR()
                res, _ = engine(img_np)
                if res:
                    ocr_lines = [r[1] for r in res]
                    ocr_text = "\n".join(ocr_lines).strip()
            except Exception:
                ocr_text = ""

        if ocr_text:
            text = f"{text}\n\n{ocr_text}".strip() if text else ocr_text

    return text, table_markdowns


def _extract_section_title(page: pymupdf.Page) -> str:
    """Extracts the topmost or largest font title span from a PDF page to serve as a section title anchor."""
    try:
        page_dict = page.get_text("dict")
        spans = []

        for b in page_dict.get("blocks", []):
            if b.get("type") != 0:  # text block only
                continue
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt and len(txt) > 2:
                        if txt.isdigit() and len(txt) <= 3:
                            continue
                        spans.append({
                            "text": txt,
                            "size": round(span.get("size", 0.0), 1),
                            "bbox": span.get("bbox", (0, 0, 0, 0))
                        })

        if not spans:
            return ""

        max_size = max(s["size"] for s in spans)
        candidate_spans = [s for s in spans if s["size"] >= max_size - 1.0]
        candidate_spans.sort(key=lambda s: (s["bbox"][1], -s["size"]))

        top_y = candidate_spans[0]["bbox"][1]
        same_line_spans = [s["text"] for s in candidate_spans if abs(s["bbox"][1] - top_y) < 5.0]

        return " ".join(same_line_spans).strip()
    except Exception:
        return ""


def load_pdf(filepath: str, dpi: int = 300) -> List[Dict[str, Any]]:
    """Loads a PDF document using PyMuPDF and extracts text, Markdown tables & section titles per page.
    Preserves multi-column reading order, excludes table bbox text from normal text, and applies 300 DPI OCR fallback.

    Args:
        filepath (str): Path to the PDF file to ingest.
        dpi (int): DPI resolution for OCR rendering (default: 300 DPI for high accuracy).

    Returns:
        List[Dict[str, Any]]: List of page dictionaries with extracted text, tables, section_title & source metadata.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {filepath}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File at {filepath} is not a PDF file.")

    filename = path.name
    doc_id = path.stem

    pages_data = []
    doc = pymupdf.open(str(path))

    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text, tables = _extract_text_and_tables_from_page(page, dpi=dpi)
            section_title = _extract_section_title(page)

            if text or tables:
                pages_data.append({
                    "text": text,
                    "tables": tables,
                    "section_title": section_title,
                    "page_number": page_num + 1,  # 1-indexed page number for human citations
                    "filename": filename,
                    "doc_id": doc_id,
                })
    finally:
        doc.close()

    return pages_data


if __name__ == "__main__":
    raw_data_dir = ROOT_DIR / "data" / "raw"
    pdf_files = list(raw_data_dir.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF documents in {raw_data_dir}:")
    for pdf_path in pdf_files:
        print("\n" + "=" * 60)
        print(f"Testing Ingestion Service on: {pdf_path.name}")
        pages = load_pdf(str(pdf_path), dpi=300)
        print(f"  • Total Extracted Pages: {len(pages)}")
        if pages:
            total_chars = sum(len(p["text"]) for p in pages)
            print(f"  • Total Extracted Characters: {total_chars}")
            print(f"  • Page 1 Snippet: {pages[0]['text'][:180].replace('\n', ' ')}...")
