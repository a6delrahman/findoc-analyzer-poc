"""
DOCX Generator Service - Generates Word documents from Pydantic models using templates.
Enhanced for HBL Sales Enablement with Scoring Matrix.
"""
import io
import os
import logging
from datetime import datetime
from typing import Optional, List
from docxtpl import DocxTemplate
from schemas import (
    FuturePotential, 
    PersonData, 
    ScoreComponent,
    BusinessOpportunity,
    ClientRating
)

logger = logging.getLogger(__name__)

# Template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def ensure_template_dir():
    """Ensure the templates directory exists."""
    if not os.path.exists(TEMPLATE_DIR):
        os.makedirs(TEMPLATE_DIR)
        logger.info(f"Created templates directory: {TEMPLATE_DIR}")


def get_template_path(template_name: str) -> str:
    """Get the full path to a template file."""
    return os.path.join(TEMPLATE_DIR, template_name)


def _format_currency(value) -> str:
    """Format a value as Swiss currency. Handles str, float, int, or None."""
    if value is None:
        return "-"
    
    if isinstance(value, str):
        try:
            numeric_value = float(value.replace("'", "").replace(",", "").replace("CHF", "").strip())
            return f"CHF {numeric_value:,.2f}".replace(",", "'")
        except ValueError:
            return value
    
    try:
        return f"CHF {float(value):,.2f}".replace(",", "'")
    except (ValueError, TypeError):
        return str(value) if value else "-"


def _format_score(score: float) -> str:
    """Format score as X/10 or X/100."""
    return f"{score:.0f}"


# ==========================================
# PART 1: Strategic Analysis & Rating Context
# ==========================================

def _build_rating_context(data: FuturePotential) -> dict:
    """Build context for the strategic rating section."""
    rating = data.rating_result
    score = rating.score
    
    # Rating color mapping for visual distinction
    rating_colors = {
        ClientRating.A: "#28a745",  # Green
        ClientRating.B: "#ffc107",  # Yellow/Orange
        ClientRating.C: "#6c757d",  # Gray
    }
    
    return {
        # Rating overview
        "client_rating": rating.rating.value,
        "rating_label": rating.rating_label,
        "total_score": _format_score(score.total_score),
        "potential_score": _format_score(score.potential_score),
        "trigger_score": _format_score(score.trigger_score),
        
        # Rating narrative
        "rating_rationale": rating.rating_rationale,
        "recommended_action": rating.recommended_action,
        "focus_areas": rating.focus_areas,
        "primary_triggers": rating.primary_triggers,
        
        # Score components for matrix display
        "potential_components": [
            {
                "category": c.category,
                "raw_value": c.raw_value or "-",
                "points": c.points,
                "max_points": c.max_points,
                "weight": c.weight_percent,
                "weighted_score": f"{c.weighted_score:.1f}",
                "reasoning": c.reasoning
            }
            for c in score.potential_components
        ],
        "trigger_components": [
            {
                "category": c.category,
                "raw_value": c.raw_value or "-",
                "points": c.points,
                "max_points": c.max_points,
                "weight": c.weight_percent,
                "weighted_score": f"{c.weighted_score:.1f}",
                "reasoning": c.reasoning
            }
            for c in score.trigger_components
        ],
    }


# ==========================================
# PART 2: Personal Master Data Context
# ==========================================

def _build_master_data_context(data: FuturePotential) -> dict:
    """Build context for personal master data section."""
    steuer_data = data.steuererklaerung_data
    person_1 = steuer_data.person_1
    person_2 = steuer_data.person_2
    
    def get_person_master(person: Optional[PersonData], prefix: str) -> dict:
        if not person or not person.master_data:
            return {
                f"{prefix}_name": "-",
                f"{prefix}_alter": "-",
                f"{prefix}_geburtsdatum": "-",
                f"{prefix}_erwerbsstatus": "-",
            }
        
        md = person.master_data
        return {
            f"{prefix}_name": md.name or "-",
            f"{prefix}_alter": str(md.alter) if md.alter else "-",
            f"{prefix}_geburtsdatum": md.geburtsdatum or "-",
            f"{prefix}_erwerbsstatus": md.employment_status.value if md.employment_status else "-",
        }
    
    context = {
        "tax_year": steuer_data.tax_year,
        "verheiratet": "Ja" if steuer_data.verheiratet else "Nein",
        "housing_situation": steuer_data.housing_situation.value,
        "has_person_2": person_2 is not None,
    }
    
    context.update(get_person_master(person_1, "p1"))
    context.update(get_person_master(person_2, "p2"))
    
    return context


# ==========================================
# PART 3: Financial Data Context
# ==========================================

def _build_income_analysis_context(data: FuturePotential) -> dict:
    """Build context for income analysis section."""
    steuer_data = data.steuererklaerung_data
    person_1 = steuer_data.person_1
    person_2 = steuer_data.person_2
    
    def get_person_income(person: Optional[PersonData], prefix: str) -> dict:
        if not person:
            return {
                f"{prefix}_bruttoeinkommen": "-",
                f"{prefix}_nettoeinkommen": "-",
                f"{prefix}_steuerbares_einkommen": "-",
            }
        
        return {
            f"{prefix}_bruttoeinkommen": _format_currency(person.haupterwerb),
            f"{prefix}_nettoeinkommen": _format_currency(person.nettoeinkommen),
            f"{prefix}_steuerbares_einkommen": _format_currency(person.steuerbares_einkommen),
        }
    
    context = {}
    context.update(get_person_income(person_1, "p1"))
    context.update(get_person_income(person_2, "p2"))
    
    return context


def _build_asset_balance_context(data: FuturePotential) -> dict:
    """Build context for asset balance sheet."""
    steuer_data = data.steuererklaerung_data
    person_1 = steuer_data.person_1
    person_2 = steuer_data.person_2
    
    def get_person_assets(person: Optional[PersonData], prefix: str) -> dict:
        if not person:
            return {
                f"{prefix}_bankguthaben": "-",
                f"{prefix}_wertschriften": "-",
                f"{prefix}_liegenschaften": "-",
                f"{prefix}_versicherungen": "-",
                f"{prefix}_schulden": "-",
                f"{prefix}_hypotheken": "-",
            }
        
        return {
            f"{prefix}_bankguthaben": _format_currency(person.bankguthaben),
            f"{prefix}_wertschriften": _format_currency(person.wertschriften),
            f"{prefix}_liegenschaften": _format_currency(person.liegenschaften),
            f"{prefix}_versicherungen": _format_currency(person.lebens_und_versicherungspolicen),
            f"{prefix}_schulden": _format_currency(person.schulden),
            f"{prefix}_hypotheken": _format_currency(person.hypothekarschulden),
        }
    
    context = {}
    context.update(get_person_assets(person_1, "p1"))
    context.update(get_person_assets(person_2, "p2"))
    
    return context


def _build_pension_indicators_context(data: FuturePotential) -> dict:
    """Build context for pension indicators."""
    pension = data.pension_indicators
    steuer_data = data.steuererklaerung_data
    
    # Get 3a deductions
    p1_3a = _format_currency(steuer_data.person_1.saeule_3a_einzahlung) if steuer_data.person_1 else "-"
    p2_3a = _format_currency(steuer_data.person_2.saeule_3a_einzahlung) if steuer_data.person_2 else "-"
    
    p1_pk = _format_currency(steuer_data.person_1.pk_einkauf) if steuer_data.person_1 else "-"
    p2_pk = _format_currency(steuer_data.person_2.pk_einkauf) if steuer_data.person_2 else "-"
    
    return {
        "p1_saeule_3a": p1_3a,
        "p2_saeule_3a": p2_3a,
        "p1_pk_einkauf": p1_pk,
        "p2_pk_einkauf": p2_pk,
        "saeule_3a_genutzt": "Ja" if pension.saeule_3a_genutzt else "Nein",
        "saeule_3a_voll_ausgeschoepft": "Ja" if pension.saeule_3a_voll_ausgeschoepft else "Nein" if pension.saeule_3a_voll_ausgeschoepft is not None else "Unbekannt",
        "pk_einkauf_erkennbar": "Ja" if pension.pk_einkauf_erkennbar else "Nein",
        "estimated_pension_gap": pension.estimated_pension_gap or "-",
        "high_income_low_savings": "Ja" if pension.high_income_low_savings_indicator else "Nein",
    }


# ==========================================
# PART 4: Business Opportunities Context
# ==========================================

def _build_opportunities_context(data: FuturePotential) -> dict:
    """Build context for business opportunities section."""
    opportunities = data.business_opportunities
    
    return {
        "has_opportunities": len(opportunities) > 0,
        "opportunities": [
            {
                "type": opp.opportunity_type,
                "title": opp.title,
                "description": opp.description,
                "potential": opp.estimated_potential or "-",
                "urgency": opp.urgency,
                "urgency_label": {
                    "hoch": "🔴 Hoch",
                    "mittel": "🟡 Mittel", 
                    "niedrig": "🟢 Niedrig"
                }.get(opp.urgency, opp.urgency),
                "next_steps": opp.next_steps,
            }
            for opp in opportunities
        ],
    }


def _build_assets_context(detected_assets) -> dict:
    """Build assets context from detected assets."""
    if not detected_assets:
        return {
            "detected_assets": [],
            "has_assets": False,
        }
    
    return {
        "detected_assets": [
            {
                "asset_type": asset.asset_type,
                "estimated_value": _format_currency(asset.estimated_value)
            }
            for asset in detected_assets
        ],
        "has_assets": len(detected_assets) > 0,
    }


# ==========================================
# MAIN CONTEXT BUILDER
# ==========================================

def _build_report_context(data: FuturePotential) -> dict:
    """Build complete context for HBL sales enablement report."""
    context = {
        "report_date": datetime.now().strftime("%d.%m.%Y"),
    }
    
    # Part 1: Strategic Analysis & Rating
    context.update(_build_rating_context(data))
    
    # Part 2: Personal Master Data
    context.update(_build_master_data_context(data))
    
    # Part 3: Financial Data
    context.update(_build_income_analysis_context(data))
    context.update(_build_asset_balance_context(data))
    context.update(_build_pension_indicators_context(data))
    context.update(_build_assets_context(data.detected_assets))
    
    # Part 4: Business Opportunities
    context.update(_build_opportunities_context(data))
    
    # Narrative sections
    context["summary"] = data.summary
    context["strategic_recommendations"] = data.strategic_recommendations
    
    # === ADD THESE LINES TO EXPOSE NESTED PYDANTIC MODELS ===
    # Include the original Pydantic models for template access with nested attributes
    context["steuererklaerung_data"] = data.steuererklaerung_data
    context["rating_result"] = data.rating_result
    context["pension_indicators"] = data.pension_indicators
    context["business_opportunities"] = data.business_opportunities
    context["detected_assets"] = data.detected_assets
    
    return context


def _generate_output_filename(steuer_data, output_filename: Optional[str]) -> str:
    """Generate output filename for the report."""
    if output_filename:
        return f"{output_filename}.docx"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tax_year = steuer_data.tax_year if steuer_data and steuer_data.tax_year else "unknown"
    return f"HBL_Kundenanalyse_{tax_year}_{timestamp}.docx"


def generate_steuererklaerung_report(
    data: FuturePotential,
    template_name: str = "hbl_sales_enablement_template.docx",
    output_filename: Optional[str] = None
) -> tuple[bytes, str]:
    """
    Generate a DOCX report from FuturePotential data.
    
    Args:
        data: FuturePotential Pydantic model with extracted data
        template_name: Name of the template file in templates/
        output_filename: Optional custom filename (without extension)
        
    Returns:
        Tuple of (docx_bytes, filename)
    """
    template_path = get_template_path(template_name)
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    doc = DocxTemplate(template_path)
    context = _build_report_context(data)
    doc.render(context)
    
    docx_buffer = io.BytesIO()
    doc.save(docx_buffer)
    docx_bytes = docx_buffer.getvalue()
    
    filename = _generate_output_filename(data.steuererklaerung_data, output_filename)
    
    logger.info(f"Generated DOCX report: {filename} ({len(docx_bytes)} bytes)")
    
    return docx_bytes, filename


def save_report_locally(docx_bytes: bytes, filename: str, output_dir: str = "output_reports") -> str:
    """
    Save a generated report to a local directory.
    
    Args:
        docx_bytes: The DOCX file content
        filename: The filename to save as
        output_dir: Directory to save to
        
    Returns:
        Full path to the saved file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as f:
        f.write(docx_bytes)
    
    logger.info(f"Report saved locally: {filepath}")
    return filepath