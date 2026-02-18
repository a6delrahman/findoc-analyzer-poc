import os
from typing import List, Dict, Any
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentAnalysisFeature, AnalyzeDocumentRequest
import fitz  # PyMuPDF

from env_config import get_config
from clients import llm_client, doc_intel_client, MODEL_DEPLOYMENT
from phoenix_service import get_prompt_service
from schemas import (
    FuturePotential, 
    DocumentSetAnalysis, 
    DocumentType,
    ExtractedDocument,
    get_schema_for_doc_type
)
from split import parse_azure_markdown, is_new_document
from docx_generator_service import generate_steuererklaerung_report, save_report_locally
from sharepoint_service import get_sharepoint_service
from scoring_engine import (
    calculate_opportunity_score, 
    determine_rating, 
    StrategicFocus,
    ScoringThresholds,
    ScoringWeights
)


config = get_config()


# ==========================================
# DEFAULT PROMPTS
# ==========================================

DEFAULT_PROMPT_STEUERERKLAERUNG = """
Du bist ein spezialisierter Analyst für die Hypothekarbank Lenzburg (HBL). Du analysierst Steuererklärungen, um Geschäftspotenziale in den Bereichen Anlagen (AUM), Hypotheken und Vorsorge zu identifizieren.

## KONTEXT: HBL Strategie
- **Wachstumsfokus:** Anlagen (AUM), Hypotheken, Altersvorsorge
- **"Money in Motion":** Fokus auf Liquiditätsereignisse (Pensionierung, Unternehmer 50+, Erbschaften)
- **Vorsorgelücke:** Oft ist die genaue Vorsorgesituation eine "Black Box" - identifiziere Indikatoren

## AUFGABE 1: Persönliche Stammdaten extrahieren
Extrahiere die Stammdaten für beide Personen (falls verheiratet):
- Name (falls erkennbar)
- Geburtsdatum / Alter
- Erwerbsstatus: "angestellt" oder "selbstständig" (aus Einkünften ableitbar)
- Wohnsituation: "Eigenheim" (wenn Liegenschaften > 0) oder "Miete"

## AUFGABE 2: Finanzielle Datenextraktion
Extrahiere gemäss Pydantic Schema mit **Schweizer Zahlenformat** (Hochkomma als Tausendertrenner):

**Einkommen:**
- Bruttoeinkommen (Ziff. 1.1, Code 010/020)
- Nettoeinkommen (Ziff. 20, Code 401)
- Steuerbares Einkommen

**Vermögen:**
- Bankguthaben (Ziff. 30.2)
- Wertschriften (Ziff. 30.1)
- Liegenschaften (Ziff. 31)
- Lebensversicherungen (Ziff. 30.3)

**Verbindlichkeiten:**
- Hypothekarschulden (Teil von Ziff. 34)
- Übrige Schulden (Ziff. 34)

**Vorsorge-Indikatoren:**
- Säule 3a Einzahlung (Ziff. 21)
- PK-Einkäufe (falls erkennbar)

## AUFGABE 3: Vorsorge-Indikatoren analysieren
Analysiere die Vorsorgesituation:
- Wird Säule 3a genutzt? Ist sie voll ausgeschöpft?
- Sind PK-Einkäufe erkennbar?
- **Wichtig:** Hohes Einkommen + tiefe sichtbare Ersparnisse = Indikator für Vorsorgelücke
- Schätze die Vorsorgelücke narrativ ein

## AUFGABE 4: Business Opportunities für HBL identifizieren
Identifiziere konkrete Geschäftsmöglichkeiten für die Bank:

**Anlagen (AUM):**
- Hohe Bankguthaben, die angelegt werden könnten
- Liquiditätsereignisse (Pensionierung, Verkauf)

**Vorsorge:**
- Nicht ausgeschöpfte Säule 3a
- PK-Einkaufspotenzial
- Selbstständige mit komplexen Vorsorgebedürfnissen

**Hypotheken:**
- Refinanzierungspotenzial bei bestehenden Hypotheken
- Wohneigentumskauf bei Mietern mit Kapazität

Für jede Opportunity:
- Typ (Anlage/Vorsorge/Hypothek)
- Titel und Beschreibung
- Geschätztes Potenzial
- Dringlichkeit (hoch/mittel/niedrig)
- Konkrete nächste Schritte für den Berater

## AUFGABE 5: Executive Summary
Verfasse eine prägnante Zusammenfassung (ca. 150 Wörter) der Kundenanalyse für den Berater:
- Kundenprofil in einem Satz
- Wichtigste Potenziale
- Kritische Trigger oder Zeitfenster

## AUFGABE 6: Strategische Empfehlungen für Berater
Verfasse Handlungsempfehlungen (ca. 200 Wörter) für den Kundenberater:
- Gesprächsaufhänger
- Priorisierte Themen
- Konkrete Produktvorschläge
- Timing-Empfehlungen

### Wichtige Regeln
- Output als valides JSON gemäss Pydantic Schema
- Schweizer Hochdeutsch auf professionellem Niveau
- Wenn Werte nicht vorhanden: null setzen, nicht halluzinieren
- Fokus auf actionable Insights für die Bank"""

DEFAULT_PROMPT_EXTRACTOR = """
Du bist eine spezialisierte KI für Finanzdaten-Extraktion.
Du erhältst EIN spezifisches Dokument.

Deine Aufgabe:
Extrahiere die Daten strikt nach dem vorgegebenen Schema für diesen Dokumententyp.
Wenn Felder nicht vorhanden sind, lasse sie leer (null). Halluziniere keine Werte.
"""


# ==========================================
# AZURE DOC INTELLIGENCE
# ==========================================

def extract_markdown_from_pdf(pdf_path: str, cache_dir: str = "cache_markdown") -> str:
    """Wandelt PDF in Markdown via Azure Doc Intelligence mit Caching."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    filename = os.path.basename(pdf_path)
    cache_path = os.path.join(cache_dir, f"{filename}.md")
    
    if os.path.exists(cache_path):
        print(f"CACHE HIT: Lade Markdown aus '{cache_path}'")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
            
    print(f"CACHE MISS: Sende '{pdf_path}' an Azure Document Intelligence...")
    
    try:
        with open(pdf_path, "rb") as f:
            poller = doc_intel_client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=f.read()),
                output_content_format="markdown",
                features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION]
            )
        result: AnalyzeResult = poller.result()
        markdown_content = result.content
        
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"  -> Markdown gespeichert in '{cache_path}'")
        
        return markdown_content
        
    except Exception as e:
        print(f"FEHLER bei Azure Doc Intelligence: {e}")
        raise e


# ==========================================
# DOCUMENT TYPE MAPPING
# ==========================================

def _matches_keywords(text: str, keywords: List[str]) -> bool:
    """Helper function to check if any keyword is present in text."""
    return any(keyword in text for keyword in keywords)


def map_doc_type_string_to_enum(doc_type_str: str) -> DocumentType:
    """Maps the string doc_type from split.py to the DocumentType enum."""
    doc_type_lower = doc_type_str.lower()
    
    # K105 Bonitäts-Unterlagen
    if _matches_keywords(doc_type_lower, ["lohnausweis", "salary", "certificat de salaire"]):
        return DocumentType.LOHNAUSWEIS
    if _matches_keywords(doc_type_lower, ["pensionskasse", "vorsorge", "bvg", "versichertenausweis", "certificate of insurance"]):
        return DocumentType.PENSIONSKASSENAUSWEIS
    if _matches_keywords(doc_type_lower, ["iko", "kredit"]):
        return DocumentType.IKO_AUSKUNFT
    if _matches_keywords(doc_type_lower, ["steuererkl", "tax"]):
        return DocumentType.STEUERERKLAERUNG
    
    # K106 Grundpfand-Unterlagen
    if _matches_keywords(doc_type_lower, ["bauplan", "grundriss", "floor"]):
        return DocumentType.BAUPLAN
    if _matches_keywords(doc_type_lower, ["foto", "photo"]):
        return DocumentType.FOTOS_LIEGENSCHAFT
    if "grundbuch" in doc_type_lower:
        return DocumentType.GRUNDBUCHAUSZUG
    if _matches_keywords(doc_type_lower, ["kubisch", "gebäudeversicherung", "agv", "police"]):
        return DocumentType.KUBISCHE_BERECHNUNG
    
    # K103 Verträge
    if _matches_keywords(doc_type_lower, ["darlehen", "hypothek", "kaufvertrag", "vertrag", "urkunde"]):
        return DocumentType.VERTRAEGE
    
    return DocumentType.ANDERES


# ==========================================
# STRUCTURED DATA EXTRACTION
# ==========================================

def extract_structured_data(doc_chunk: str, doc_type: DocumentType) -> ExtractedDocument:
    """Extrahiert strukturierte Daten aus dem Markdown basierend auf Dokumenttyp."""
    
    # Get the appropriate schema for this document type
    schema_class = get_schema_for_doc_type(doc_type)
    
    print(f"  > Extrahiere Daten für '{doc_type.value}' mit Schema '{schema_class.__name__}' ({len(doc_chunk)} Zeichen)")

    prompt = get_doc_extraction_prompt()
    prompt = f"{prompt}\nFokus: Extrahiere Daten für Typ: {doc_type.value}"
    
    completion = llm_client.beta.chat.completions.parse(
        model=MODEL_DEPLOYMENT,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": doc_chunk}
        ],
        response_format=schema_class,  # Use the specific schema
        reasoning_effort="high",
    )
    
    return completion.choices[0].message.parsed


# ==========================================
# USE CASE 1: Steuererklärung Potenzial
# ==========================================

def get_steuererklaerung_prompt() -> str:
    """Get Steuererklärung prompt from Phoenix or use default."""
    prompt_service = get_prompt_service()
    phoenix_prompt = prompt_service.get_prompt(config["prompt_steuererklaerung"])
    return phoenix_prompt if phoenix_prompt else DEFAULT_PROMPT_STEUERERKLAERUNG

def get_doc_extraction_prompt() -> str:
    """Get document extraction prompt from Phoenix or use default."""
    prompt_service = get_prompt_service()
    phoenix_prompt = prompt_service.get_prompt(config["prompt_doc_extraction"])
    return phoenix_prompt if phoenix_prompt else DEFAULT_PROMPT_EXTRACTOR

def analyze_tax_potential(markdown_chunk: str, strategic_focus: StrategicFocus = StrategicFocus.BALANCED) -> FuturePotential:
    """
    Analyze tax declaration and extract potential with HBL-specific scoring.
    
    Args:
        markdown_chunk: The markdown content of the tax declaration
        strategic_focus: HBL's current strategic focus for weight adjustment
    
    Returns:
        FuturePotential with complete analysis including scoring
    """
    print(f"  > Starte Steuererklärung Analyse ({len(markdown_chunk)} Zeichen)...")
    
    prompt = get_steuererklaerung_prompt()
    
    # First, get the LLM extraction (without scoring - that's calculated separately)
    from schemas import FuturePotentialLLMExtraction  # Intermediate schema without scoring
    
    completion = llm_client.beta.chat.completions.parse(
        model=MODEL_DEPLOYMENT,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Steuererklärung:\n\n{markdown_chunk}"}
        ],
        response_format=FuturePotentialLLMExtraction,
        reasoning_effort="high",
    )
    
    llm_result = completion.choices[0].message.parsed
    
    # Calculate scoring
    score = calculate_opportunity_score(
        data=llm_result.steuererklaerung_data,
        strategic_focus=strategic_focus
    )
    
    # Determine rating
    rating_result = determine_rating(
        score=score,
        data=llm_result.steuererklaerung_data
    )
    
    # Combine into final result
    return FuturePotential(
        rating_result=rating_result,
        steuererklaerung_data=llm_result.steuererklaerung_data,
        detected_assets=llm_result.detected_assets,
        pension_indicators=llm_result.pension_indicators,
        business_opportunities=llm_result.business_opportunities,
        summary=llm_result.summary,
        strategic_recommendations=llm_result.strategic_recommendations
    )


def run_steuererklaerung_pipeline(
    pdf_path: str,
    upload_to_sharepoint: bool = True,
    sharepoint_folder: str = "Output/Steuererklaerung"
) -> Dict[str, Any]:
    """
    Complete pipeline for Steuererklärung analysis:
    1. Extract markdown from PDF
    2. Analyze with LLM
    3. Generate DOCX report
    4. Upload to SharePoint (optional)
    
    Returns:
        Dict with analysis results and file paths
    """
    print("=" * 60)
    print("STEUERERKLÄRUNG PIPELINE")
    print("=" * 60)
    
    # Step 1: Extract markdown
    print("\n--- Schritt 1: Markdown Extraktion ---")
    markdown_content = extract_markdown_from_pdf(pdf_path)
    print(f"  -> {len(markdown_content)} Zeichen extrahiert")
    
    # Step 2: Analyze with LLM
    print("\n--- Schritt 2: LLM Analyse ---")
    analysis_result = analyze_tax_potential(markdown_content)
    print("  -> Analyse abgeschlossen")
    
    # Step 3: Generate DOCX report
    print("\n--- Schritt 3: DOCX Report generieren ---")
    try:
        docx_bytes, filename = generate_steuererklaerung_report(analysis_result)
        print(f"  -> Report generiert: {filename}")
        
        # Save locally
        local_path = save_report_locally(docx_bytes, filename)
        print(f"  -> Lokal gespeichert: {local_path}")
    except FileNotFoundError as e:
        print("  -> WARNUNG: Template nicht gefunden. Bitte 'python create_template.py' ausführen.")
        print(f"     {e}")
        docx_bytes, filename, local_path = None, None, None
    
    # Step 4: Upload to SharePoint (optional)
    sharepoint_result = None
    if upload_to_sharepoint and docx_bytes:
        print("\n--- Schritt 4: SharePoint Upload ---")
        try:
            sp_service = get_sharepoint_service()
            sharepoint_result = sp_service.upload_file(
                filename=filename,
                content=docx_bytes,
                folder_path=sharepoint_folder
            )
            if sharepoint_result.get("success"):
                print(f"  -> Hochgeladen: {sharepoint_result.get('web_url')}")
            else:
                print(f"  -> Upload fehlgeschlagen: {sharepoint_result.get('error')}")
        except Exception as e:
            print(f"  -> SharePoint Upload Fehler: {e}")
            sharepoint_result = {"success": False, "error": str(e)}
    
    return {
        "analysis": analysis_result,
        "local_path": local_path,
        "filename": filename,
        "sharepoint": sharepoint_result
    }


# ==========================================
# USE CASE 2: DOCUMENT SET PIPELINE
# ==========================================

def detect_document_boundaries(md_pages: List[str]) -> List[Dict[str, Any]]:
    """Analyze pages and detect document boundaries using LLM."""
    if not md_pages:
        print("Keine Seiten im Markdown gefunden!")
        return []
    
    print(f"Analysiere {len(md_pages)} Seiten für Dokumentgrenzen...")
    
    documents = []
    current_start_idx = 0
    current_doc_type = "Unknown"
    prev_summary = "Start of file"
    
    for i, md_text in enumerate(md_pages):
        result = is_new_document(md_text, prev_summary, current_doc_type)
        
        decision = result.decision.upper()
        doc_type_val = result.doc_type
        summary_val = result.summary
        
        print(f"  Seite {i+1}: {decision} | {doc_type_val}")
        
        if decision == "NEW" and i > 0:
            documents.append({
                "start_page": current_start_idx,
                "end_page": i - 1,
                "doc_type": current_doc_type
            })
            current_start_idx = i
            current_doc_type = doc_type_val
        elif i == 0:
            current_doc_type = doc_type_val
        
        prev_summary = summary_val
    
    documents.append({
        "start_page": current_start_idx,
        "end_page": len(md_pages) - 1,
        "doc_type": current_doc_type
    })
    
    return documents


def split_and_save(
    pdf_path: str, 
    md_pages: List[str], 
    boundaries: List[Dict[str, Any]], 
    output_dir: str = "output_splits"
) -> List[Dict[str, Any]]:
    """Split PDF and save corresponding markdown for each document."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Ordner erstellt: {output_dir}")
    
    generated_files = []
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Fehler beim Öffnen des PDFs '{pdf_path}': {e}")
        return []
    
    print(f"\nSpeichere {len(boundaries)} Dokumente (PDF + Markdown)...")
    
    for i, boundary in enumerate(boundaries):
        start_idx = boundary["start_page"]
        end_idx = boundary["end_page"]
        doc_type = boundary["doc_type"]
        
        if start_idx > end_idx or start_idx < 0 or end_idx >= len(md_pages):
            print(f"  Warnung: Ungültiger Bereich ({start_idx}-{end_idx}), überspringe.")
            continue
        
        safe_type = "".join([c for c in doc_type if c.isalnum() or c in (' ', '_', '-')]).strip()
        safe_type = safe_type.replace(" ", "_").lower()[:50]
        base_filename = f"{i+1:02d}_{safe_type}"
        
        # Save PDF split
        pdf_filepath = os.path.join(output_dir, f"{base_filename}.pdf")
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx)
        new_doc.save(pdf_filepath)
        new_doc.close()
        
        # Save corresponding Markdown
        md_filepath = os.path.join(output_dir, f"{base_filename}.md")
        combined_markdown = "\n\n<!-- PageBreak -->\n\n".join(md_pages[start_idx:end_idx + 1])
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(combined_markdown)
        
        print(f"  -> {base_filename}: PDF + MD (Seiten {start_idx+1}-{end_idx+1})")
        
        generated_files.append({
            "pdf_path": pdf_filepath,
            "md_path": md_filepath,
            "markdown_content": combined_markdown,
            "doc_type": doc_type,
            "start_page": start_idx + 1,
            "end_page": end_idx + 1
        })
    
    doc.close()
    return generated_files


def extract_data_from_splits(split_files: List[Dict[str, Any]]) -> List[ExtractedDocument]:
    """Extract structured data from each split using the already-available markdown."""
    extracted_docs = []
    
    print(f"\nExtrahiere strukturierte Daten aus {len(split_files)} Dokumenten...")
    
    for item in split_files:
        doc_type_str = item["doc_type"]
        markdown_content = item["markdown_content"]
        
        print(f"\n  Verarbeite: {os.path.basename(item['pdf_path'])} ({doc_type_str})")
        
        if not markdown_content or len(markdown_content) < 10:
            print("    WARNUNG: Markdown ist leer, überspringe.")
            continue
        
        doc_type_enum = map_doc_type_string_to_enum(doc_type_str)
        
        try:
            extracted = extract_structured_data(markdown_content, doc_type_enum)
            extracted_docs.append(extracted)
            print(f"    -> Erfolgreich: {doc_type_enum.value}")
        except Exception as e:
            print(f"    FEHLER bei Extraktion: {e}")
            continue
    
    return extracted_docs


def _upload_split_to_sharepoint(item: Dict[str, Any], sharepoint_folder: str, sp_service) -> Dict[str, Any]:
    """Helper function to upload a single split PDF to SharePoint with category metadata."""
    pdf_path_split = item["pdf_path"]
    doc_type = item["doc_type"]
    filename = os.path.basename(pdf_path_split)
    
    doc_type_enum = map_doc_type_string_to_enum(doc_type)
    category = doc_type_enum.value
    
    print(f"\n  Lade hoch: {filename} (Kategorie: {category})")
    
    with open(pdf_path_split, "rb") as f:
        pdf_content = f.read()
    
    result = sp_service.upload_file_with_category(
        filename=filename,
        content=pdf_content,
        folder_path=sharepoint_folder,
        category=category
    )
    
    if result.get("success"):
        print(f"    -> Hochgeladen: {result.get('web_url')}")
        metadata_status = result.get("metadata_update", {})
        if metadata_status.get("success"):
            print(f"    -> Kategorie gesetzt: {category}")
        else:
            print(f"    -> Kategorie-Update fehlgeschlagen: {metadata_status.get('error')}")
    else:
        print(f"    -> Upload fehlgeschlagen: {result.get('error')}")
    
    return {
        "filename": filename,
        "category": category,
        "result": result
    }


def _upload_splits_to_sharepoint(split_files: List[Dict[str, Any]], sharepoint_folder: str) -> List[Dict[str, Any]]:
    """Helper function to upload all split PDFs to SharePoint."""
    upload_results = []
    
    try:
        sp_service = get_sharepoint_service()
        
        for item in split_files:
            # upload pdf split
            upload_result = _upload_split_to_sharepoint(item, sharepoint_folder, sp_service)
            upload_results.append(upload_result)
            
    except Exception as e:
        print(f"  -> SharePoint Upload Fehler: {e}")
    
    return upload_results

def _process_markdown_and_boundaries(pdf_path: str, markdown_cache_path: str = None):
    """Step 1: Parse markdown into pages and detect document boundaries."""
    print("\n--- Schritt 1: Dokumentgrenzen erkennen ---")
    
    cache_path = markdown_cache_path or f"cache_markdown/{os.path.basename(pdf_path)}.md"
    
    if not os.path.exists(cache_path):
        print("\n--- Markdown nicht gecached, extrahiere... ---")
        extract_markdown_from_pdf(pdf_path)
    
    md_pages = parse_azure_markdown(cache_path)
    print(f"  -> {len(md_pages)} Seiten gefunden")
    
    boundaries = detect_document_boundaries(md_pages)
    
    if boundaries:
        print("\n  Erkannte Dokumente:")
        for i, b in enumerate(boundaries):
            print(f"    {i+1}. {b['doc_type']} (Seiten {b['start_page']+1}-{b['end_page']+1})")
    
    return md_pages, boundaries


def _split_pdf_and_extract(pdf_path: str, md_pages: List[str], boundaries: List[Dict[str, Any]], output_dir: str):
    """Steps 2-3: Split PDF and extract structured data."""
    print("\n--- Schritt 2: PDF & Markdown splitten ---")
    split_files = split_and_save(pdf_path, md_pages, boundaries, output_dir)
    
    if not split_files:
        return None, None
    
    print("\n--- Schritt 3: Daten extrahieren ---")
    extracted_docs = extract_data_from_splits(split_files)
    
    return split_files, extracted_docs


def _upload_extracted_data_item(sp_service, item, split_file, i: int, sharepoint_folder: str):
    """Upload JSON and Excel for a single extracted document."""
    doc_type_str = item.doc_type.value
    json_data = item.model_dump()
    
    if split_file:
        base_filename = os.path.basename(split_file['pdf_path']).rsplit('.', 1)[0]
    else:
        base_filename = f"doc_{i+1}"
    
    print(f"\n  Verarbeite: {base_filename} ({doc_type_str})")
    
    # Upload JSON
    print("    -> Lade JSON hoch...")
    json_result = sp_service.upload_extracted_data_as_json(
        extracted_data=json_data,
        base_filename=base_filename,
        document_type=doc_type_str,
        sharepoint_folder=sharepoint_folder
    )
    
    if json_result.get("success"):
        print(f"       ✓ JSON hochgeladen: {json_result.get('name')}")
    else:
        print(f"       ✗ JSON Upload fehlgeschlagen: {json_result.get('error')}")
    
    # Upload Excel
    print("    -> Lade Excel hoch...")
    excel_result = sp_service.upload_extracted_data_as_excel(
        extracted_data=json_data,
        base_filename=base_filename,
        document_type=doc_type_str,
        sharepoint_folder=sharepoint_folder
    )
    
    if excel_result.get("success"):
        print(f"       ✓ Excel hochgeladen: {excel_result.get('name')}")
    else:
        print(f"       ✗ Excel Upload fehlgeschlagen: {excel_result.get('error')}")
    
    return json_result, excel_result


def _upload_extracted_data_to_sharepoint(extracted_docs: List[ExtractedDocument], split_files: List[Dict[str, Any]], sharepoint_folder: str):
    """Step 5: Upload extracted content as JSON and Excel files."""
    print("\n--- Schritt 5: Extrahierte Daten nach SharePoint hochladen ---")
    
    json_upload_results = []
    excel_upload_results = []
    
    try:
        sp_service = get_sharepoint_service()
        
        for i, item in enumerate(extracted_docs):
            split_file = split_files[i] if i < len(split_files) else None
            json_result, excel_result = _upload_extracted_data_item(
                sp_service, item, split_file, i, sharepoint_folder
            )
            json_upload_results.append(json_result)
            excel_upload_results.append(excel_result)
            
    except Exception as e:
        print(f"  -> Fehler beim Hochladen der extrahierten Daten: {e}")
    
    return json_upload_results, excel_upload_results


def run_document_set_pipeline(
    pdf_path: str, 
    markdown_cache_path: str = None,
    output_dir: str = "output_splits",
    upload_to_sharepoint: bool = True,
    sharepoint_folder: str = "UseCase-2"
) -> Dict[str, Any]:
    """
    WORKFLOW:
    1. Parse markdown into pages and detect document boundaries (LLM)
    2. Split PDF and store corresponding markdown per split
    3. Extract structured data using stored markdown
    4. Upload split PDFs to SharePoint with Category metadata
    
    Returns:
        Dict with analysis results and upload details
    """
    print("=" * 60)
    print("DOCUMENT SET PIPELINE")
    print("=" * 60)
    
    # Step 1: Parse and detect boundaries
    md_pages, boundaries = _process_markdown_and_boundaries(pdf_path, markdown_cache_path)
    
    if not boundaries:
        print("Keine Dokumente erkannt!")
        return {"analysis": DocumentSetAnalysis(documents=[]), "uploads": []}
    
    # Steps 2-3: Split and extract
    split_files, extracted_docs = _split_pdf_and_extract(pdf_path, md_pages, boundaries, output_dir)
    
    if not split_files:
        print("Keine Split-Dateien erstellt!")
        return {"analysis": DocumentSetAnalysis(documents=[]), "uploads": []}
    
    # Step 4: Upload split PDFs to SharePoint (optional)
    upload_results = []
    if upload_to_sharepoint:
        print("\n--- Schritt 4: PDFs nach SharePoint hochladen ---")
        upload_results = _upload_splits_to_sharepoint(split_files, sharepoint_folder)

    # Step 5: Upload extracted content as JSON and Excel files
    json_upload_results = []
    excel_upload_results = []
    
    if upload_to_sharepoint:
        json_upload_results, excel_upload_results = _upload_extracted_data_to_sharepoint(
            extracted_docs, split_files, sharepoint_folder
        )

    return {
        "analysis": DocumentSetAnalysis(documents=extracted_docs),
        "pdf_uploads": upload_results,
        "json_uploads": json_upload_results,
        "excel_uploads": excel_upload_results
    }


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("--- Start Processing ---\n")

    # Paths
    steuererklaerung_path = "data/steuererklaerung.pdf"
    dok_set_path = "data/dok_set.pdf"
    dok_set_markdown_path = "cache_markdown/dok_set.md"
    
    # ========================================
    # USE CASE 1: Steuererklärung Analysis
    # ========================================
    print("\n" + "=" * 60)
    print("USE CASE 1: Steuererklärung Potenzialanalyse")
    print("=" * 60)
    
    if os.path.exists(steuererklaerung_path):
        result_1 = run_steuererklaerung_pipeline(
            pdf_path=steuererklaerung_path,
            upload_to_sharepoint=True,
            sharepoint_folder="UseCase-1"  # Changed folder
        )
        
        print("\n--- ERGEBNIS Use Case 1 ---")
        print(f"Lokaler Report: {result_1.get('local_path')}")
        if result_1.get('sharepoint', {}).get('success'):
            print(f"SharePoint URL: {result_1['sharepoint'].get('web_url')}")
        print("\nAnalyse-Daten:")
        print(result_1['analysis'].model_dump_json(indent=2))
    else:
        print(f"Datei nicht gefunden: {steuererklaerung_path}")
    
    # ========================================
    # USE CASE 2: Document Set Pipeline
    # ========================================
    print("\n" + "=" * 60)
    print("USE CASE 2: Document Set Pipeline")
    print("=" * 60)
    
    if os.path.exists(dok_set_path):
        # Ensure markdown is extracted
        if not os.path.exists(dok_set_markdown_path):
            extract_markdown_from_pdf(dok_set_path)
        
        result_2 = run_document_set_pipeline(
            pdf_path=dok_set_path,
            markdown_cache_path=dok_set_markdown_path,
            upload_to_sharepoint=True,
            sharepoint_folder="UseCase-2"  # Upload PDFs here
        )
        
        print("\n--- ERGEBNIS Use Case 2 ---")
        print(f"PDF Uploads: {len(result_2.get('pdf_uploads', []))} Dateien")
        for upload in result_2.get('pdf_uploads', []):
            status = "✓" if upload['result'].get('success') else "✗"
            print(f"  {status} {upload['filename']} -> {upload['category']}")

        print(f"\nJSON Uploads: {len(result_2.get('json_uploads', []))} Dateien")
        for upload in result_2.get('json_uploads', []):
            status = "✓" if upload.get('success') else "✗"
            print(f"  {status} {upload.get('name', 'Unknown')}")

        print(f"\nExcel Uploads: {len(result_2.get('excel_uploads', []))} Dateien")
        for upload in result_2.get('excel_uploads', []):
            status = "✓" if upload.get('success') else "✗"
            print(f"  {status} {upload.get('name', 'Unknown')}")

        print("\nAnalyse-Daten:")
        print(result_2['analysis'].model_dump_json(indent=2))
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)