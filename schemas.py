from typing import List, Optional, Literal, Union, Annotated
from enum import Enum
from pydantic import BaseModel, Field, Discriminator

# ==========================================
# CASE 1: Steuererklärung (HBL Sales Enablement)
# ==========================================

# --- Personal Master Data (Stammdaten) ---

class EmploymentStatus(str, Enum):
    ANGESTELLT = "angestellt"
    SELBSTSTAENDIG = "selbstständig"
    UNBEKANNT = "unbekannt"

class HousingSituation(str, Enum):
    EIGENHEIM = "Eigenheim"
    MIETE = "Miete"
    UNBEKANNT = "unbekannt"

class PersonMasterData(BaseModel):
    """Personal master data extracted from tax declaration."""
    name: Optional[str] = Field(None, description="Name der Person (falls erkennbar).")
    geburtsdatum: Optional[str] = Field(None, description="Geburtsdatum (falls erkennbar).")
    alter: Optional[int] = Field(None, description="Berechnetes Alter basierend auf Geburtsdatum.")
    employment_status: EmploymentStatus = Field(
        EmploymentStatus.UNBEKANNT, 
        description="Erwerbsstatus: angestellt oder selbstständig."
    )

class BeiblattSeite14Data(BaseModel):
    freiwillige_zuwendungen_aufstellung: Optional[bool] = Field(None)
    lohnausweise_vorhanden: Optional[bool] = Field(None)
    bescheinigung_selbstvorsorge: Optional[bool] = Field(None)
    wertschriftenverzeichnis_vorhanden: Optional[bool] = Field(None)
    zusaetzliche_beilagen: Optional[List[str]] = Field(None)

class PersonData(BaseModel):
    """Financial data for a person from tax declaration."""
    # Master data
    master_data: Optional[PersonMasterData] = Field(None, description="Stammdaten der Person.")
    
    # Income data
    haupterwerb: Optional[str] = Field(None, description="Brutto-Einkommen Ziff. 1.1 (Code 010/020).")
    nettoeinkommen: Optional[str] = Field(None, description="Nettoeinkommen Ziff. 20 (Code 401).")
    steuerbares_einkommen: Optional[str] = Field(None, description="Steuerbares Einkommen.")
    
    # Deductions and indicators
    unterhaltsbeitraege: Optional[str] = Field(None, description="Ziff. 5.1 / 5.2.")
    saeule_3a_einzahlung: Optional[str] = Field(None, description="Säule 3a Einzahlung (Ziff. 21).")
    pk_einkauf: Optional[str] = Field(None, description="Pensionskassen-Einkauf (falls erkennbar).")
    
    # Assets
    wertschriften: Optional[str] = Field(None, description="Total Steuerwert (Ziff. 30.1).")
    bankguthaben: Optional[str] = Field(None, description="Bankguthaben (Ziff. 30.2).")
    lebens_und_versicherungspolicen: Optional[str] = Field(None, description="Ziff. 30.3.")
    liegenschaften: Optional[str] = Field(None, description="Steuerwert Liegenschaften (Ziff. 31).")
    
    # Liabilities
    schulden: Optional[str] = Field(None, description="Total der Schulden (Ziff. 34).")
    hypothekarschulden: Optional[str] = Field(None, description="Hypothekarschulden (Teil von Ziff. 34).")
    
    # Attachments
    beiblatt_seite_14: Optional[BeiblattSeite14Data] = Field(None)

class SteuererklaerungData(BaseModel):
    """Complete tax declaration data."""
    person_1: PersonData = Field(..., description="Daten zur ersten Person.")
    person_2: Optional[PersonData] = Field(None, description="Daten zur zweiten Person.")
    verheiratet: bool = Field(..., description="Verheiratet oder nicht.")
    housing_situation: HousingSituation = Field(
        HousingSituation.UNBEKANNT,
        description="Wohnsituation: Eigenheim oder Miete (abgeleitet aus Liegenschaften)."
    )
    tax_year: int = Field(..., description="Steuerjahr.")

# --- Scoring & Rating ---

class ScoreComponent(BaseModel):
    """Individual scoring component with points and reasoning."""
    category: str = Field(..., description="Kategorie (z.B. 'Freie Vermögenswerte', 'Alter').")
    raw_value: Optional[str] = Field(None, description="Extrahierter Rohwert.")
    points: int = Field(..., ge=0, le=10, description="Punktzahl 0-10.")
    max_points: int = Field(10, description="Maximale Punktzahl.")
    weight_percent: int = Field(..., description="Gewichtung in Prozent.")
    weighted_score: float = Field(..., description="Gewichteter Score.")
    reasoning: str = Field(..., description="Begründung für die Punktvergabe.")

class OpportunityScore(BaseModel):
    """Client Opportunity Score with two dimensions."""
    # Potential dimension (volume focus)
    potential_score: float = Field(..., ge=0, le=100, description="Potenzial-Score (0-100).")
    potential_components: List[ScoreComponent] = Field(..., description="Komponenten des Potenzial-Scores.")
    
    # Relevance/Trigger dimension (timing focus)
    trigger_score: float = Field(..., ge=0, le=100, description="Trigger-Score (0-100).")
    trigger_components: List[ScoreComponent] = Field(..., description="Komponenten des Trigger-Scores.")
    
    # Combined
    total_score: float = Field(..., ge=0, le=100, description="Gesamtscore (0-100).")

class ClientRating(str, Enum):
    A = "A"  # Top Opportunity (85-100)
    B = "B"  # Development (60-84)
    C = "C"  # Standard (<60)

class RatingResult(BaseModel):
    """Final client rating with explanation."""
    rating: ClientRating = Field(..., description="Rating A, B oder C.")
    score: OpportunityScore = Field(..., description="Detaillierter Score.")
    rating_label: str = Field(..., description="Label z.B. 'Top Opportunity'.")
    
    # Narrative explanation for advisors
    rating_rationale: str = Field(
        ..., 
        description="Narrative Begründung des Ratings für den Kundenberater."
    )
    primary_triggers: List[str] = Field(
        ..., 
        description="Wichtigste identifizierte Trigger/Ereignisse."
    )
    recommended_action: str = Field(
        ..., 
        description="Empfohlene Handlung für den Berater."
    )
    focus_areas: List[str] = Field(
        ..., 
        description="Fokusgebiete: Anlage, Vorsorge, Hypothek."
    )

# --- Business Opportunities ---

class BusinessOpportunity(BaseModel):
    """Identified business opportunity for HBL."""
    opportunity_type: str = Field(..., description="Art der Opportunity (Anlage, Vorsorge, Hypothek).")
    title: str = Field(..., description="Kurztitel der Opportunity.")
    description: str = Field(..., description="Beschreibung der Opportunity.")
    estimated_potential: Optional[str] = Field(None, description="Geschätztes Volumen/Potenzial.")
    urgency: Literal["hoch", "mittel", "niedrig"] = Field(..., description="Dringlichkeit.")
    next_steps: List[str] = Field(..., description="Konkrete nächste Schritte.")

class PotentialAsset(BaseModel):
    asset_type: str = Field(..., description="Typ des Vermögenswerts.")
    estimated_value: Optional[str] = Field(None, description="Geschätzter Wert des Vermögenswerts.")

# --- Pension Gap Analysis ---

class PensionGapIndicators(BaseModel):
    """Indicators for pension gap analysis."""
    saeule_3a_genutzt: bool = Field(..., description="Wird Säule 3a genutzt?")
    saeule_3a_voll_ausgeschoepft: Optional[bool] = Field(None, description="Ist 3a voll ausgeschöpft?")
    pk_einkauf_erkennbar: bool = Field(..., description="Sind PK-Einkäufe erkennbar?")
    estimated_pension_gap: Optional[str] = Field(None, description="Geschätzte Vorsorgelücke (narrativ).")
    high_income_low_savings_indicator: bool = Field(
        ..., 
        description="Hohes Einkommen bei tiefen sichtbaren Ersparnissen = Potenzial."
    )

# --- Main Output Schema for Use Case 1 ---

class FuturePotential(BaseModel):
    """Complete analysis output for HBL sales enablement."""
    # Part 1: Strategic Analysis & Rating
    rating_result: RatingResult = Field(..., description="Kunden-Rating und Score.")
    
    # Part 2: Personal Data (extracted)
    steuererklaerung_data: SteuererklaerungData = Field(..., description="Extrahierte Steuerdaten.")
    
    # Part 3: Financial Analysis
    detected_assets: List[PotentialAsset] = Field(..., description="Erkannte Vermögenswerte.")
    pension_indicators: PensionGapIndicators = Field(..., description="Vorsorge-Indikatoren.")
    
    # Part 4: Business Opportunities (HBL Focus)
    business_opportunities: List[BusinessOpportunity] = Field(
        ..., 
        description="Identifizierte Geschäftsmöglichkeiten für HBL."
    )
    
    # Narrative sections for document
    summary: str = Field(..., description="Executive Summary der Analyse.")
    strategic_recommendations: str = Field(..., description="Strategische Empfehlungen für Berater.")


class FuturePotentialLLMExtraction(BaseModel):
    """
    Intermediate schema for LLM extraction.
    Does NOT include rating_result - that's calculated by the scoring engine.
    """
    steuererklaerung_data: SteuererklaerungData = Field(..., description="Extrahierte Steuerdaten.")
    detected_assets: List[PotentialAsset] = Field(..., description="Erkannte Vermögenswerte.")
    pension_indicators: PensionGapIndicators = Field(..., description="Vorsorge-Indikatoren.")
    business_opportunities: List[BusinessOpportunity] = Field(
        ..., 
        description="Identifizierte Geschäftsmöglichkeiten für HBL."
    )
    summary: str = Field(..., description="Executive Summary der Analyse.")
    strategic_recommendations: str = Field(..., description="Strategische Empfehlungen für Berater.")

# ==========================================
# CASE 2: Dokumenten-Set (Klassifizierung & Extraktion)
# ==========================================

# --- Document Type Enum ---

class DocumentType(str, Enum):
    # K105 Bonitäts-Unterlagen
    LOHNAUSWEIS = "K105 Bonitäts-Unterlagen / Lohnausweis"
    PENSIONSKASSENAUSWEIS = "K105 Bonitäts-Unterlagen / Pensionskassenausweis"
    IKO_AUSKUNFT = "K105 Bonitäts-Unterlagen / IKO-Auskunft"
    STEUERERKLAERUNG = "K105 Bonitäts-Unterlagen / Steuererklärung"
    
    # K106 Grundpfand-Unterlagen
    BAUPLAN = "K106 Grundpfand-Unterlagen / Bauplan"
    FOTOS_LIEGENSCHAFT = "K106 Grundpfand-Unterlagen / Fotos Liegenschaft"
    GRUNDBUCHAUSZUG = "K106 Grundpfand-Unterlagen / Grundbuchauszug"
    KUBISCHE_BERECHNUNG = "K106 Grundpfand-Unterlagen / Kubische Berechnung"
    
    # K103 Verträge
    VERTRAEGE = "K103 Verträge / Kaufvertrag"
    
    # Sonstige
    ANDERES = "Anderes"


# --- Base class for all extracted documents ---

class ExtractedDocumentBase(BaseModel):
    """Base class with common fields for all document types."""
    confidence_score: float = Field(..., description="Konfidenz der Klassifizierung (0-1).")


# --- K105 Bonitäts-Unterlagen ---

class ExtractedLohnausweis(ExtractedDocumentBase):
    """Aktueller Lohnausweis oder Lohnabrechnungen - K105 Bonitäts-Unterlagen"""
    doc_type: Literal[DocumentType.LOHNAUSWEIS] = DocumentType.LOHNAUSWEIS
    jahreslohn_netto: Optional[float] = Field(None, description="Jahreslohn Netto.")
    jahreslohn_brutto: Optional[float] = Field(None, description="Jahreslohn Brutto (für Berechnung: Netto zu Brutto = 86.28% von Brutto).")
    bonus: Optional[float] = Field(None, description="Bonus, falls vorhanden (Durchschnitt der letzten 2 Jahre, 50% vom Durchschnitt).")
    year: Optional[int] = Field(None, description="Jahr des Lohnausweises.")


class ExtractedPensionskassenausweis(ExtractedDocumentBase):
    """Pensionskassenausweis - K105 Bonitäts-Unterlagen"""
    doc_type: Literal[DocumentType.PENSIONSKASSENAUSWEIS] = DocumentType.PENSIONSKASSENAUSWEIS
    hoehe_pensionskassenbezug_jahr: Optional[float] = Field(None, description="Höhe des Pensionskassenbezugs pro Jahr.")
    pension_fund_name: Optional[str] = Field(None, description="Name der Pensionskasse.")


class ExtractedIKOAuskunft(ExtractedDocumentBase):
    """IKO-Auskunft - K105 Bonitäts-Unterlagen"""
    doc_type: Literal[DocumentType.IKO_AUSKUNFT] = DocumentType.IKO_AUSKUNFT
    leasing_verpflichtungen: Optional[float] = Field(None, description="Leasing-Verpflichtungen.")
    kleinkredite: Optional[float] = Field(None, description="Kleinkredite.")
    kreditkartenengagements: Optional[float] = Field(None, description="Ausstehende Kreditkartenengagements.")


class ExtractedSteuererklaerungK105(ExtractedDocumentBase):
    """Steuererklärung - K105 Bonitäts-Unterlagen"""
    doc_type: Literal[DocumentType.STEUERERKLAERUNG] = DocumentType.STEUERERKLAERUNG
    person_1: PersonData = Field(..., description="Daten zur ersten Person.")
    person_2: Optional[PersonData] = Field(None, description="Daten zur zweiten Person.")
    verheiratet: bool = Field(..., description="Verheiratet oder nicht.")
    tax_year: int = Field(..., description="Steuerjahr.")


# --- K106 Grundpfand-Unterlagen ---

class ExtractedBauplan(ExtractedDocumentBase):
    """Baupläne mit m2-Angaben - K106 Grundpfand-Unterlagen"""
    doc_type: Literal[DocumentType.BAUPLAN] = DocumentType.BAUPLAN
    nettowohnflaeche: Optional[float] = Field(None, description="Nettowohnfläche in m2.")
    terrasse_flaeche: Optional[float] = Field(None, description="Terrassenfläche in m2.")
    garten_flaeche: Optional[float] = Field(None, description="Gartenfläche in m2 (gedeckt).")


class ExtractedFotosLiegenschaft(ExtractedDocumentBase):
    """Fotos der Liegenschaft - K106 Grundpfand-Unterlagen"""
    doc_type: Literal[DocumentType.FOTOS_LIEGENSCHAFT] = DocumentType.FOTOS_LIEGENSCHAFT
    aussenfotos_vorhanden: Optional[bool] = Field(None, description="Aussenfotos vorhanden (Objekt als Ganzes sichtbar mit Parkplatz).")
    innenfotos_vorhanden: Optional[bool] = Field(None, description="Innenfotos vorhanden (Küche, Nasszellen, mind. zwei Zimmer, Elektro- und Heizungsinstallation).")
    zustand_beschreibung: Optional[str] = Field(None, description="Beschreibung des Zustands sowie Bauqualität (für IAZI-Bewertung).")


class ExtractedGrundbuchauszug(ExtractedDocumentBase):
    """Grundbuchauszug (Haupt- und Stammobjekt) - K106 Grundpfand-Unterlagen"""
    doc_type: Literal[DocumentType.GRUNDBUCHAUSZUG] = DocumentType.GRUNDBUCHAUSZUG
    grundstuecksnummer: Optional[str] = Field(None, description="Grundstücksnummer (Grundbuchblattnummer) für Finstar + FBS.")
    egrid_nummer: Optional[str] = Field(None, description="E-Grid-Nummer für Finstar.")
    grundstuecksflaeche: Optional[float] = Field(None, description="Grundstücksfläche für Finstar + IAZI.")
    eigentum_definition: Optional[str] = Field(None, description="Eigentum: Definition der Eigentumsverhältnisse für Finstar + FBS.")
    anmerkungen: Optional[str] = Field(None, description="Anmerkungen (z.B. Veräusserungsbeschränkung, BVG-Gelder).")
    dienstbarkeiten: Optional[str] = Field(None, description="Dienstbarkeiten - Überprüfung ob wertrelevant.")
    grundpfandrechte_art: Optional[str] = Field(None, description="Grundpfandrechte: Art (Register oder Physischer SB).")
    grundpfandrechte_datum: Optional[str] = Field(None, description="Datum Ausstellung SB.")
    grundpfandrechte_nominal: Optional[float] = Field(None, description="Höhe des Nominals.")
    grundpfandrechte_rangfolge: Optional[str] = Field(None, description="Rangfolgen für Finstar + FBS.")


class ExtractedKubischeBerechnung(ExtractedDocumentBase):
    """Kubische Berechnung / Gebäudeversicherungsausweis - K106 Grundpfand-Unterlagen"""
    doc_type: Literal[DocumentType.KUBISCHE_BERECHNUNG] = DocumentType.KUBISCHE_BERECHNUNG
    adresse: Optional[str] = Field(None, description="Adresse des Gebäudes.")
    baujahr: Optional[int] = Field(None, description="Baujahr.")
    schaetzdatum: Optional[str] = Field(None, description="Schätzdatum.")
    entwertung: Optional[float] = Field(None, description="Entwertung in Prozent.")
    volumen_m3: Optional[float] = Field(None, description="Volumen in m3.")


# --- K103 Verträge ---

class ExtractedVertrag(ExtractedDocumentBase):
    """Kaufvertrag / Darlehensvertrag - K103 Verträge"""
    doc_type: Literal[DocumentType.VERTRAEGE] = DocumentType.VERTRAEGE
    hypothekarhoehe: Optional[float] = Field(None, description="Höhe der Hypothek, welche abgelöst wird.")
    faelligkeit: Optional[str] = Field(None, description="Fälligkeit der Hypothek.")
    bank_name: Optional[str] = Field(None, description="Name der Bank/Gläubiger.")


# --- Sonstige ---

class ExtractedAnderes(ExtractedDocumentBase):
    """Unbekannter oder anderer Dokumententyp"""
    doc_type: Literal[DocumentType.ANDERES] = DocumentType.ANDERES
    raw_content_summary: Optional[str] = Field(None, description="Kurze Zusammenfassung des Inhalts.")


# --- Discriminated Union Type ---

ExtractedDocument = Annotated[
    Union[
        ExtractedLohnausweis,
        ExtractedPensionskassenausweis,
        ExtractedIKOAuskunft,
        ExtractedSteuererklaerungK105,
        ExtractedBauplan,
        ExtractedFotosLiegenschaft,
        ExtractedGrundbuchauszug,
        ExtractedKubischeBerechnung,
        ExtractedVertrag,
        ExtractedAnderes,
    ],
    Discriminator("doc_type")
]


# --- Mapping from DocumentType to Schema class ---

DOCUMENT_TYPE_TO_SCHEMA: dict[DocumentType, type[ExtractedDocumentBase]] = {
    DocumentType.LOHNAUSWEIS: ExtractedLohnausweis,
    DocumentType.PENSIONSKASSENAUSWEIS: ExtractedPensionskassenausweis,
    DocumentType.IKO_AUSKUNFT: ExtractedIKOAuskunft,
    DocumentType.STEUERERKLAERUNG: ExtractedSteuererklaerungK105,
    DocumentType.BAUPLAN: ExtractedBauplan,
    DocumentType.FOTOS_LIEGENSCHAFT: ExtractedFotosLiegenschaft,
    DocumentType.GRUNDBUCHAUSZUG: ExtractedGrundbuchauszug,
    DocumentType.KUBISCHE_BERECHNUNG: ExtractedKubischeBerechnung,
    DocumentType.VERTRAEGE: ExtractedVertrag,
    DocumentType.ANDERES: ExtractedAnderes,
}


def get_schema_for_doc_type(doc_type: DocumentType) -> type[ExtractedDocumentBase]:
    """Get the appropriate Pydantic schema class for a document type."""
    return DOCUMENT_TYPE_TO_SCHEMA.get(doc_type, ExtractedAnderes)


# --- Document Set Analysis ---

class DocumentSetAnalysis(BaseModel):
    documents: List[ExtractedDocument]


# --- SCHEMA FÜR SPLITTING LOGIK ---

class DocumentBoundary(BaseModel):
    doc_type: DocumentType = Field(..., description="Klassifizierter Dokumententyp.")
    start_page: int = Field(..., description="Erste Seite des Dokuments.")
    end_page: int = Field(..., description="Letzte Seite des Dokuments.")
    reasoning: str = Field(..., description="Kurze Begründung für die Einteilung.")

class SplittingResult(BaseModel):
    boundaries: List[DocumentBoundary]


# ==========================================
# CASE 3: Dokumentenrücklauf (Rahmenvertrag Validierung)
# ==========================================

class ContractValidation(BaseModel):
    is_content_modified: bool = Field(..., description="Wurde der Vertragstext verändert (z.B. Text durchgestrichen)?")
    modification_details: Optional[str] = Field(None, description="Beschreibung der gefundenen Änderungen (z.B. 'Paragraph 3 durchgestrichen').")
    signature_present: bool = Field(..., description="Ist eine Unterschrift vorhanden?")
    signature_matches_sample: bool = Field(..., description="Scheint die Unterschrift visuell zum hinterlegten Muster zu passen?")
    risk_assessment: Literal["OK", "RISK", "REJECT"]


# ==========================================
# SPLITTING CLASSIFICATION (für split.py)
# ==========================================

class DocumentClassificationSplit(BaseModel):
    decision: str  # "NEW" or "CONT"
    doc_type: str  # Document type
    summary: str   # One sentence summary