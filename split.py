from logging import config
import os
import re
from pypdf import PdfReader, PdfWriter
from schemas import DocumentClassificationSplit
from clients import llm_client, MODEL_DEPLOYMENT
from phoenix_service import get_prompt_service
from env_config import get_config

env_config = get_config()
# ==========================================
# DEFAULT PROMPTS (fallback if Phoenix unavailable)
# ==========================================

DEFAULT_PROMPT_SPLIT = """You are a document boundary detection agent. Your task is to determine if this page starts a NEW document or CONTINUES the previous one.

CRITICAL RULES - CHECK IN THIS ORDER:

1. **PAGE NUMBER CHECK (HIGHEST PRIORITY)**:
   - Look for page numbers at the TOP of the page (header area - first few lines).
   - Common patterns: "<!-- PageNumber="X" -->", "Seite X", "Page X", "page X", standalone numbers like "2", "3", "10".
   - If page number is > 1 (e.g., "2", "3", "10", "page 2"), this is ALWAYS a CONTINUATION.
   - Page number "1" or absence of page number may indicate a NEW document.

2. **PAGE FOOTER FROM PREVIOUS PAGE**:
   - Look for "<!-- PageFooter=..." --> patterns - these belong to the previous page and should be ignored for classification.

3. **DOCUMENT HEADER CHECK**:
   - Only if page number is "1" or absent, check for new document indicators:
     - New letterhead, logo, or company header
     - New document title (e.g., "Lohnausweis", "Kaufvertrag", "Versicherungspolice")
     - Form identifiers or document reference numbers that differ from previous

4. **CONTENT CONTINUITY**:
   - If the content logically continues from the previous summary, it's a CONTINUATION.
   - Mid-sentence starts, continued lists, or ongoing sections = CONTINUATION.

RESPOND WITH:
- decision: "NEW" only if this is clearly a new document (page 1 or new header). "CONT" if any page number > 1 or content continues.
- doc_type: The document type (e.g., "Lohnausweis", "Kaufvertrag", "Vorsorgeausweis") or "N/A" if continuation.
- summary: One sentence describing this page's content.
"""

# ==========================================
# FUNCTIONS
# ==========================================

def get_split_prompt() -> str:
    """Get split prompt from Phoenix or use default."""
    prompt_service = get_prompt_service()
    phoenix_prompt = prompt_service.get_prompt(env_config["prompt_split"])  # Use your Phoenix prompt name
    return phoenix_prompt if phoenix_prompt else DEFAULT_PROMPT_SPLIT


def parse_azure_markdown(md_path: str) -> list:
    """
    Splits Azure Document Intelligence markdown into pages based on <!-- PageBreak -->.
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    raw_pages = re.split(r'<!--\s*PageBreak\s*-->', full_text)

    if not raw_pages:
        print("WARNUNG: Keine Seiten im Markdown gefunden!")
        return [full_text.strip()]    
    
    pages = [p.strip() for p in raw_pages if p.strip()]
    return pages


def is_new_document(current_page_md: str, previous_page_summary: str, previous_doc_type: str):
    """
    Asks LLM if the current markdown chunk looks like the start of a new document.
    """
    base_prompt = get_split_prompt()
    
    prompt = f"""{base_prompt}

PREVIOUS DOCUMENT CONTEXT:
- Type: {previous_doc_type}
- Summary: {previous_page_summary}

CURRENT PAGE CONTENT (analyze carefully, especially the FIRST 10 LINES for page numbers):
```
{current_page_md[:4000]}
```
"""

    completion = llm_client.beta.chat.completions.parse(
        model=MODEL_DEPLOYMENT,
        messages=[
            {"role": "user", "content": prompt},
        ],
        reasoning_effort="high",
        response_format=DocumentClassificationSplit,    
    )
    return completion.choices[0].message.parsed


# ==========================================
# STANDALONE EXECUTION (for testing)
# ==========================================

INPUT_PDF = "data/dok_set.pdf"
INPUT_MARKDOWN_FILE = "cache_markdown/dok_set.md"
OUTPUT_FOLDER = "split_documents"

def main():
    if not os.path.exists(INPUT_PDF) or not os.path.exists(INPUT_MARKDOWN_FILE):
        print("Error: Input files not found.")
        return

    reader = PdfReader(INPUT_PDF)
    md_pages = parse_azure_markdown(INPUT_MARKDOWN_FILE)
    
    pdf_count = len(reader.pages)
    md_count = len(md_pages)
    
    print(f"Loaded: {pdf_count} PDF pages | {md_count} Markdown pages")

    if pdf_count != md_count:
        print("⚠️ WARNING: Page count mismatch. Splitting might be inaccurate.")
        process_limit = min(pdf_count, md_count)
    else:
        process_limit = pdf_count

    documents = []
    current_start_idx = 0
    current_doc_type = "Unknown"
    prev_summary = "Start of file"

    for i in range(process_limit):
        md_text = md_pages[i]
        result = is_new_document(md_text, prev_summary, current_doc_type)

        decision = result.decision.upper()
        doc_type_val = result.doc_type
        summary_val = result.summary
        
        print(f"Page {i+1}: {decision} | {doc_type_val}")

        if decision == "NEW" and i > 0:
            documents.append((current_start_idx, i - 1, current_doc_type))
            current_start_idx = i
            current_doc_type = doc_type_val
        elif i == 0:
            current_doc_type = doc_type_val
        
        prev_summary = summary_val

    documents.append((current_start_idx, process_limit - 1, current_doc_type))

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("\n--- Saving Files ---")
    for idx, (start, end, dtype) in enumerate(documents):
        writer = PdfWriter()
        for p in range(start, end + 1):
            writer.add_page(reader.pages[p])
        
        safe_type = "".join([c for c in dtype if c.isalnum() or c in (' ', '_', '-')]).strip()
        filename = f"{OUTPUT_FOLDER}/Doc_{idx+1}_{safe_type}.pdf"
        
        with open(filename, "wb") as f:
            writer.write(f)
        print(f"Saved: {filename} (Pages {start+1} to {end+1})")

if __name__ == "__main__":
    main()