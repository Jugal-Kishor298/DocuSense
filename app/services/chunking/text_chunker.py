import sys
from pathlib import Path
from typing import Dict, List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add workspace root to sys.path if running as main script
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import config


def chunk_table_markdown(table_md: str, max_rows: int = 15) -> List[str]:
    """Splits a Markdown table string into smaller context-preserving sub-tables,
    repeating column headers on every chunk and grouping by logical section headers
    or a maximum row threshold (max_rows).
    """
    lines = [line.strip() for line in table_md.strip().splitlines() if line.strip()]
    if not lines:
        return []

    # 1. Identify header block (up to and including separator line '|---|')
    header_lines = []
    body_lines = []
    found_separator = False

    for idx, line in enumerate(lines):
        if not found_separator:
            header_lines.append(line)
            cleaned = line.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
            if cleaned == "" and idx > 0:
                found_separator = True
        else:
            body_lines.append(line)

    if not body_lines:
        return [table_md]

    header_str = "\n".join(header_lines)

    # 2. Identify if a row is a section header (e.g. '| **Americas** |||||')
    def is_section_header(row: str) -> bool:
        cells = [c.strip() for c in row.split("|")[1:-1]]
        if not cells:
            return False
        first_cell = cells[0]
        other_cells = cells[1:]
        if first_cell and all(not c for c in other_cells):
            return True
        return False

    # 3. Split body rows into sub-group chunks
    chunks = []
    current_group = []

    for row in body_lines:
        if is_section_header(row) and current_group:
            chunk_text = header_str + "\n" + "\n".join(current_group)
            chunks.append(chunk_text)
            current_group = []
        elif len(current_group) >= max_rows:
            chunk_text = header_str + "\n" + "\n".join(current_group)
            chunks.append(chunk_text)
            current_group = []

        current_group.append(row)

    if current_group:
        chunk_text = header_str + "\n" + "\n".join(current_group)
        chunks.append(chunk_text)

    return chunks


import re

def split_text_structurally(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """Splits page text along strict structural boundaries (bullet items, paragraph double newlines, headings)
    INDEPENDENT of chunk_size. Ensures structural boundaries take absolute precedence over chunk_size,
    preventing distinct topics (e.g. Demand vs Revenue vs Profit) from merging into a single monolithic chunk.

    If a structural unit exceeds chunk_size, it is sub-split using RecursiveCharacterTextSplitter.
    """
    text = text.strip()
    if not text:
        return []

    # 1. Primary Structural Pre-Split: Split along double newlines or bullet markers
    pattern = r'\n\s*\n+|\n(?=[➢•\-])|\n(?=\d+\.\s+)'
    raw_units = [u.strip() for u in re.split(pattern, text) if u.strip()]

    if not raw_units or len(raw_units) == 1:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    # 2. Attach section headers to child bullet items for context grounding
    units = []
    current_header = ""

    for u in raw_units:
        cleaned_u = u.replace("\n", " ").strip()
        if cleaned_u.startswith("➢") or cleaned_u.startswith("•") or (len(cleaned_u) < 30 and cleaned_u.endswith(":")):
            current_header = cleaned_u
            units.append(cleaned_u)
        else:
            if current_header and not cleaned_u.startswith(current_header):
                units.append(f"{current_header}\n{cleaned_u}")
            else:
                units.append(cleaned_u)

    filtered_units = []
    for idx, u in enumerate(units):
        if (u.startswith("➢") or u.startswith("•") or u.endswith(":")) and len(u) < 30:
            if idx + 1 < len(units) and units[idx + 1].startswith(u):
                continue
        filtered_units.append(u)

    # 3. Pack tiny consecutive units up to chunk_size so multi-block PDFs
    # (e.g. academic papers) don't shatter into hundreds of 100-char fragments,
    # while still respecting structural boundaries once a chunk is substantial (>= 300 chars).
    packed_units: List[str] = []
    buffer = ""
    min_meaningful_len = min(300, chunk_size // 3)

    for u in filtered_units:
        if not buffer:
            buffer = u
        elif len(buffer) < min_meaningful_len and (len(buffer) + 2 + len(u)) <= chunk_size:
            buffer = f"{buffer}\n\n{u}"
        else:
            packed_units.append(buffer)
            buffer = u
    if buffer:
        packed_units.append(buffer)

    # 4. Sub-split any oversized structural unit using RecursiveCharacterTextSplitter
    chunks = []
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for u in packed_units:
        if len(u) > chunk_size:
            sub_chunks = sub_splitter.split_text(u)
            chunks.extend(sub_chunks)
        elif u.strip():
            chunks.append(u)

    return chunks if chunks else [text]


def chunk_document(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = None,
    chunk_overlap: int = None,
    max_table_rows: int = 15,
) -> List[Dict[str, Any]]:
    """Splits extracted page text dictionaries into structural chunks and Markdown table chunks.
    Preserves 1-indexed page citations and document metadata on every individual chunk payload.

    Args:
        pages_data (List[Dict[str, Any]]): List of page dicts from load_pdf() containing "text", "tables", "page_number", etc.
        chunk_size (int, optional): Character size per chunk. Defaults to config.chunk_size (1000).
        chunk_overlap (int, optional): Character overlap between adjacent chunks. Defaults to config.chunk_overlap (200).
        max_table_rows (int, optional): Maximum row limit per table chunk. Defaults to 15.

    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries ready for embedding and vector database insertion.
    """
    size = chunk_size if chunk_size is not None else config.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    chunks = []
    global_chunk_idx = 0

    for page in pages_data:
        page_text = page.get("text", "").strip()
        page_tables = page.get("tables", [])
        page_number = page.get("page_number", 1)
        filename = page.get("filename", "unknown.pdf")
        doc_id = page.get("doc_id", "doc")

        section_title = page.get("section_title", "").strip()
        title_prefix = f"[Section: {section_title}]\n" if section_title else ""

        # 1. Chunk normal page text using structural boundary pre-splitter
        if page_text:
            raw_chunks = split_text_structurally(page_text, chunk_size=size, chunk_overlap=overlap)
            for c_idx, chunk_str in enumerate(raw_chunks):
                cleaned_chunk = chunk_str.strip()
                if cleaned_chunk:
                    if title_prefix and not cleaned_chunk.startswith("[Section:"):
                        final_text = f"{title_prefix}{cleaned_chunk}"
                    else:
                        final_text = cleaned_chunk
                    chunk_id = f"{doc_id}_p{page_number}_c{c_idx}"
                    chunks.append({
                        "text": final_text,
                        "chunk_id": chunk_id,
                        "page_number": page_number,
                        "filename": filename,
                        "doc_id": doc_id,
                        "chunk_index": global_chunk_idx,
                        "is_table": False,
                    })
                    global_chunk_idx += 1

        # 2. Chunk Markdown tables using chunk_table_markdown
        for t_idx, table_md in enumerate(page_tables):
            table_chunks = chunk_table_markdown(table_md, max_rows=max_table_rows)
            for tbl_c_idx, tbl_chunk_str in enumerate(table_chunks):
                cleaned_tbl = tbl_chunk_str.strip()
                if cleaned_tbl:
                    if title_prefix and not cleaned_tbl.startswith("[Section:"):
                        final_tbl_text = f"{title_prefix}{cleaned_tbl}"
                    else:
                        final_tbl_text = cleaned_tbl
                    chunk_id = f"{doc_id}_p{page_number}_t{t_idx}_c{tbl_c_idx}"
                    chunks.append({
                        "text": final_tbl_text,
                        "chunk_id": chunk_id,
                        "page_number": page_number,
                        "filename": filename,
                        "doc_id": doc_id,
                        "chunk_index": global_chunk_idx,
                        "is_table": True,
                    })
                    global_chunk_idx += 1

    return chunks


if __name__ == "__main__":
    from app.services.ingestion import load_pdf

    raw_data_dir = ROOT_DIR / "data" / "raw"
    pdf_files = list(raw_data_dir.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF documents in {raw_data_dir}:")
    for pdf_path in pdf_files:
        print("\n" + "=" * 60)
        print(f"Testing Chunking Service on: {pdf_path.name}")
        pages = load_pdf(str(pdf_path), dpi=300)
        chunks = chunk_document(pages)
        print(f"  • Total Extracted Pages : {len(pages)}")
        print(f"  • Total Chunks Generated: {len(chunks)}")
        if chunks:
            avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
            print(f"  • Average Chunk Length  : {avg_len:.1f} characters")
            print(f"  • First Chunk ID       : {chunks[0]['chunk_id']}")
            print(f"  • First Chunk Snippet  :\n{chunks[0]['text'][:180].replace('\n', ' ')}...")
