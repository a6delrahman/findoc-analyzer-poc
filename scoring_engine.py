"""
HBL Client Opportunity Scoring Engine

Configurable scoring system for tax return analysis.
Two dimensions: Potential (volume) and Triggers (timing/events).
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from schemas import (
    ScoreComponent, 
    OpportunityScore, 
    ClientRating, 
    RatingResult,
    SteuererklaerungData,
    PersonData
)


# ==========================================
# CONFIGURATION: Scoring Parameters
# ==========================================

class StrategicFocus(str, Enum):
    """Bank's current strategic focus - affects weightings."""
    INVESTMENT_GROWTH = "investment_growth"  # Focus on AUM
    PENSION_FOCUS = "pension_focus"  # Focus on retirement planning
    MORTGAGE_GROWTH = "mortgage_growth"  # Focus on mortgage volume
    BALANCED = "balanced"  # Equal weighting


@dataclass
class ScoringThresholds:
    """Configurable thresholds for scoring."""
    # Free assets (Wertschriften + Bankguthaben)
    free_assets_high: float = 500_000  # 10 points
    free_assets_medium: float = 100_000  # 5 points
    
    # Gross income
    income_high: float = 200_000  # 10 points
    income_medium: float = 120_000  # 6 points
    
    # Age brackets for trigger scoring
    age_prime: tuple = (55, 63)  # 10 points - critical pension planning window
    age_development: tuple = (45, 54)  # 6 points - wealth building phase
    
    # Rating thresholds
    rating_a_threshold: int = 85
    rating_b_threshold: int = 60


@dataclass
class ScoringWeights:
    """Configurable weights for score components (must sum to 100)."""
    # Potential dimension weights
    free_assets_weight: int = 40
    income_weight: int = 20
    real_estate_weight: int = 15
    pension_gap_weight: int = 25
    
    # Trigger dimension weights
    age_weight: int = 35
    employment_weight: int = 25
    mortgage_situation_weight: int = 20
    money_in_motion_weight: int = 20


# Default configuration
DEFAULT_THRESHOLDS = ScoringThresholds()
DEFAULT_WEIGHTS = ScoringWeights()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _parse_swiss_number(value: Optional[str]) -> float:
    """Parse Swiss formatted number (e.g., '128'836' or 'CHF 500'000')."""
    if not value:
        return 0.0
    
    try:
        # Remove CHF, spaces, and convert Swiss apostrophe to nothing
        cleaned = str(value).replace("CHF", "").replace("'", "").replace(" ", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return 0.0


def _calculate_total_free_assets(person_1: PersonData, person_2: Optional[PersonData]) -> float:
    """Calculate total free assets (Wertschriften + Bankguthaben)."""
    total = 0.0
    
    for person in [person_1, person_2]:
        if person:
            total += _parse_swiss_number(person.wertschriften)
            total += _parse_swiss_number(person.bankguthaben)
    
    return total


def _calculate_total_income(person_1: PersonData, person_2: Optional[PersonData]) -> float:
    """Calculate total gross income."""
    total = 0.0
    
    for person in [person_1, person_2]:
        if person and person.haupterwerb:
            total += _parse_swiss_number(person.haupterwerb)
    
    return total


def _get_primary_age(person_1: PersonData, person_2: Optional[PersonData]) -> Optional[int]:
    """Get the primary person's age (older person if married)."""
    ages = []
    
    for person in [person_1, person_2]:
        if person and person.master_data and person.master_data.alter:
            ages.append(person.master_data.alter)
    
    return max(ages) if ages else None


def _is_self_employed(person_1: PersonData, person_2: Optional[PersonData]) -> bool:
    """Check if any person is self-employed."""
    from schemas import EmploymentStatus
    
    for person in [person_1, person_2]:
        if person and person.master_data:
            if person.master_data.employment_status == EmploymentStatus.SELBSTSTAENDIG:
                return True
    return False


def _has_high_mortgage(person_1: PersonData, person_2: Optional[PersonData]) -> bool:
    """Check if there's a significant mortgage (refinancing potential)."""
    total_mortgage = 0.0
    total_property_value = 0.0
    
    for person in [person_1, person_2]:
        if person:
            total_mortgage += _parse_swiss_number(person.hypothekarschulden)
            total_property_value += _parse_swiss_number(person.liegenschaften)
    
    # High mortgage = mortgage exists and is > 50% of property value
    if total_property_value > 0 and total_mortgage > 0:
        ltv = total_mortgage / total_property_value
        return ltv > 0.5
    
    return total_mortgage > 200_000  # Or absolute threshold


# ==========================================
# SCORING FUNCTIONS
# ==========================================

def _score_free_assets(
    data: SteuererklaerungData, 
    thresholds: ScoringThresholds,
    weight: int
) -> ScoreComponent:
    """Score free assets (Wertschriften + Bankguthaben)."""
    total = _calculate_total_free_assets(data.person_1, data.person_2)
    
    if total >= thresholds.free_assets_high:
        points = 10
        reasoning = f"Hohe freie Vermögenswerte (>{thresholds.free_assets_high:,.0f} CHF) - erhebliches Anlagepotenzial."
    elif total >= thresholds.free_assets_medium:
        points = 5
        reasoning = f"Mittlere freie Vermögenswerte ({thresholds.free_assets_medium:,.0f}-{thresholds.free_assets_high:,.0f} CHF) - Entwicklungspotenzial."
    elif total > 0:
        points = 2
        reasoning = "Geringe freie Vermögenswerte - limitiertes Anlagepotenzial."
    else:
        points = 0
        reasoning = "Keine freien Vermögenswerte erkennbar."
    
    return ScoreComponent(
        category="Freie Vermögenswerte",
        raw_value=f"CHF {total:,.0f}".replace(",", "'"),
        points=points,
        max_points=10,
        weight_percent=weight,
        weighted_score=(points / 10) * weight,
        reasoning=reasoning
    )


def _score_income(
    data: SteuererklaerungData,
    thresholds: ScoringThresholds,
    weight: int
) -> ScoreComponent:
    """Score gross income."""
    total = _calculate_total_income(data.person_1, data.person_2)
    
    if total >= thresholds.income_high:
        points = 10
        reasoning = f"Hohes Einkommen (>{thresholds.income_high:,.0f} CHF) - hohe Spar- und Anlagekapazität."
    elif total >= thresholds.income_medium:
        points = 6
        reasoning = f"Gutes Einkommen ({thresholds.income_medium:,.0f}-{thresholds.income_high:,.0f} CHF)."
    elif total > 50_000:
        points = 3
        reasoning = "Durchschnittliches Einkommen."
    else:
        points = 0
        reasoning = "Niedriges oder nicht erkennbares Einkommen."
    
    return ScoreComponent(
        category="Bruttoeinkommen",
        raw_value=f"CHF {total:,.0f}".replace(",", "'"),
        points=points,
        max_points=10,
        weight_percent=weight,
        weighted_score=(points / 10) * weight,
        reasoning=reasoning
    )


def _score_age(
    data: SteuererklaerungData,
    thresholds: ScoringThresholds,
    weight: int
) -> ScoreComponent:
    """Score age as trigger indicator."""
    age = _get_primary_age(data.person_1, data.person_2)
    
    if age is None:
        points = 0
        reasoning = "Alter nicht erkennbar."
    elif thresholds.age_prime[0] <= age <= thresholds.age_prime[1]:
        points = 10
        reasoning = f"Kritisches Alter ({age} Jahre) - optimales Fenster für Vorsorgeplanung und Pensionsvorbereitung."
    elif thresholds.age_development[0] <= age <= thresholds.age_development[1]:
        points = 6
        reasoning = f"Entwicklungsphase ({age} Jahre) - Vermögensaufbau und langfristige Planung."
    elif age > thresholds.age_prime[1]:
        points = 8
        reasoning = f"Kurz vor/nach Pensionierung ({age} Jahre) - Auszahlungs- und Anlageentscheidungen."
    else:
        points = 3
        reasoning = f"Jüngeres Alter ({age} Jahre) - langfristiger Aufbau."
    
    return ScoreComponent(
        category="Alter (Trigger)",
        raw_value=str(age) if age else "N/A",
        points=points,
        max_points=10,
        weight_percent=weight,
        weighted_score=(points / 10) * weight,
        reasoning=reasoning
    )


def _score_employment(
    data: SteuererklaerungData,
    weight: int
) -> ScoreComponent:
    """Score employment status as trigger."""
    is_self_emp = _is_self_employed(data.person_1, data.person_2)
    
    if is_self_emp:
        points = 10
        reasoning = "Selbstständig - komplexere Vorsorgebedürfnisse, höherer Beratungsbedarf, potenzielles Firmenkunden-Potenzial."
    else:
        points = 5
        reasoning = "Angestellt - standardisierte Vorsorgesituation."
    
    from schemas import EmploymentStatus
    status = "Selbstständig" if is_self_emp else "Angestellt"
    
    return ScoreComponent(
        category="Erwerbsstatus",
        raw_value=status,
        points=points,
        max_points=10,
        weight_percent=weight,
        weighted_score=(points / 10) * weight,
        reasoning=reasoning
    )


def _score_mortgage_situation(
    data: SteuererklaerungData,
    weight: int
) -> ScoreComponent:
    """Score mortgage situation for refinancing potential."""
    has_mortgage = _has_high_mortgage(data.person_1, data.person_2)
    
    total_mortgage = 0.0
    for person in [data.person_1, data.person_2]:
        if person:
            total_mortgage += _parse_swiss_number(person.hypothekarschulden)
    
    if has_mortgage:
        points = 8
        reasoning = "Bestehende Hypothek mit Refinanzierungspotenzial."
    elif total_mortgage > 0:
        points = 4
        reasoning = "Hypothek vorhanden, limitiertes Refinanzierungspotenzial."
    else:
        points = 2
        reasoning = "Keine Hypothek erkennbar - potenzieller Neukunde für Immobilienfinanzierung."
    
    return ScoreComponent(
        category="Hypothekarsituation",
        raw_value=f"CHF {total_mortgage:,.0f}".replace(",", "'") if total_mortgage > 0 else "Keine",
        points=points,
        max_points=10,
        weight_percent=weight,
        weighted_score=(points / 10) * weight,
        reasoning=reasoning
    )


def _has_pillar_3a(person_1: PersonData, person_2: Optional[PersonData]) -> bool:
    """Check if any person has Säule 3a contributions."""
    for person in [person_1, person_2]:
        if person and person.saeule_3a_einzahlung:
            val = _parse_swiss_number(person.saeule_3a_einzahlung)
            if val > 0:
                return True
    return False


def _determine_pension_gap_score(income: float, free_assets: float, has_3a: bool) -> tuple[int, str]:
    """Determine pension gap points and reasoning based on financial indicators."""
    if income > 150_000 and free_assets < 100_000:
        return 10, "Hohes Einkommen bei tiefen sichtbaren Ersparnissen - signifikante Vorsorgelücke wahrscheinlich."
    
    if income > 100_000 and not has_3a:
        return 8, "Gutes Einkommen ohne erkennbare Säule 3a - Optimierungspotenzial."
    
    if not has_3a and income > 50_000:
        return 6, "Keine Säule 3a erkennbar - Beratungsbedarf für Vorsorgeoptimierung."
    
    if has_3a:
        return 4, "Säule 3a wird genutzt - PK-Einkaufspotenzial prüfen."
    
    return 2, "Limitierte Informationen zur Vorsorgesituation."


def _score_pension_gap(
    data: SteuererklaerungData,
    weight: int
) -> ScoreComponent:
    """Score pension gap indicators."""
    income = _calculate_total_income(data.person_1, data.person_2)
    free_assets = _calculate_total_free_assets(data.person_1, data.person_2)
    has_3a = _has_pillar_3a(data.person_1, data.person_2)
    
    points, reasoning = _determine_pension_gap_score(income, free_assets, has_3a)
    
    # Determine raw value based on points
    raw_value = "Indikator" if points > 5 else "Gering"
    
    return ScoreComponent(
        category="Vorsorgelücke",
        raw_value=raw_value,
        points=points,
        max_points=10,
        weight_percent=weight,
        weighted_score=(points / 10) * weight,
        reasoning=reasoning
    )


def _score_real_estate(
    data: SteuererklaerungData,
    weight: int
) -> ScoreComponent:
    """Score real estate situation."""
    total_property = 0.0
    for person in [data.person_1, data.person_2]:
        if person:
            total_property += _parse_swiss_number(person.liegenschaften)
    
    if total_property > 1_000_000:
        points = 8
        reasoning = "Hochwertige Immobilie - Vermögensverwaltung und Nachfolgeplanung relevant."
    elif total_property > 500_000:
        points = 6
        reasoning = "Immobilie vorhanden - Cross-Selling für Gebäudeversicherung und Renovation."
    elif total_property > 0:
        points = 4
        reasoning = "Immobilie vorhanden."
    else:
        points = 2
        reasoning = "Keine Immobilie - potenzieller Hypothekarkunde bei Kaufabsicht."
    
    return ScoreComponent(
        category="Immobiliensituation",
        raw_value=f"CHF {total_property:,.0f}".replace(",", "'") if total_property > 0 else "Keine",
        points=points,
        max_points=10,
        weight_percent=weight,
        weighted_score=(points / 10) * weight,
        reasoning=reasoning
    )


def _score_money_in_motion(
    data: SteuererklaerungData,
    weight: int
) -> ScoreComponent:
    """Score money in motion indicators (inheritance, sale, cash events)."""
    age = _get_primary_age(data.person_1, data.person_2)
    is_self_emp = _is_self_employed(data.person_1, data.person_2)
    
    # Self-employed 50+ = potential business succession
    if is_self_emp and age and age >= 50:
        points = 10
        reasoning = "Selbstständig 50+ - potenzielle Geschäftsnachfolge und Liquiditätsereignis in Sicht."
    elif age and age >= 58:
        points = 8
        reasoning = "Nahe Pensionierung - Kapitalbezugsentscheidungen und Anlage von PK-Geldern."
    elif age and age >= 50:
        points = 5
        reasoning = "Mögliche Erbschaften oder Vermögensübertragungen in diesem Lebensabschnitt."
    else:
        points = 2
        reasoning = "Keine unmittelbaren Money-in-Motion-Indikatoren erkennbar."
    
    # Determine raw value based on points
    if points >= 8:
        raw_value = "Hoch"
    elif points >= 5:
        raw_value = "Mittel"
    else:
        raw_value = "Niedrig"
    
    return ScoreComponent(
        category="Money in Motion",
        raw_value=raw_value,
        points=points,
        max_points=10,
        weight_percent=weight,
        weighted_score=(points / 10) * weight,
        reasoning=reasoning
    )


# ==========================================
# MAIN SCORING FUNCTION
# ==========================================

def calculate_opportunity_score(
    data: SteuererklaerungData,
    strategic_focus: StrategicFocus = StrategicFocus.BALANCED,
    thresholds: ScoringThresholds = DEFAULT_THRESHOLDS,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
    segment_bonus: int = 0  # e.g., +15 for entrepreneurs
) -> OpportunityScore:
    """
    Calculate the complete Client Opportunity Score.
    
    Returns:
        OpportunityScore with potential and trigger dimensions.
    """
    # Adjust weights based on strategic focus
    adjusted_weights = _adjust_weights_for_focus(weights, strategic_focus)
    
    # Calculate Potential dimension components
    potential_components = [
        _score_free_assets(data, thresholds, adjusted_weights.free_assets_weight),
        _score_income(data, thresholds, adjusted_weights.income_weight),
        _score_real_estate(data, adjusted_weights.real_estate_weight),
        _score_pension_gap(data, adjusted_weights.pension_gap_weight),
    ]
    
    # Calculate Trigger dimension components
    trigger_components = [
        _score_age(data, thresholds, adjusted_weights.age_weight),
        _score_employment(data, adjusted_weights.employment_weight),
        _score_mortgage_situation(data, adjusted_weights.mortgage_situation_weight),
        _score_money_in_motion(data, adjusted_weights.money_in_motion_weight),
    ]
    
    # Sum weighted scores
    potential_score = sum(c.weighted_score for c in potential_components)
    trigger_score = sum(c.weighted_score for c in trigger_components)
    
    # Total score (average of both dimensions + segment bonus)
    total_score = min(100, ((potential_score + trigger_score) / 2) + segment_bonus)
    
    return OpportunityScore(
        potential_score=round(potential_score, 1),
        potential_components=potential_components,
        trigger_score=round(trigger_score, 1),
        trigger_components=trigger_components,
        total_score=round(total_score, 1)
    )


def _adjust_weights_for_focus(
    weights: ScoringWeights, 
    focus: StrategicFocus
) -> ScoringWeights:
    """Adjust weights based on strategic focus."""
    if focus == StrategicFocus.INVESTMENT_GROWTH:
        return ScoringWeights(
            free_assets_weight=50,  # Increased
            income_weight=20,
            real_estate_weight=10,
            pension_gap_weight=20,
            age_weight=30,
            employment_weight=20,
            mortgage_situation_weight=15,
            money_in_motion_weight=35  # Increased
        )
    elif focus == StrategicFocus.PENSION_FOCUS:
        return ScoringWeights(
            free_assets_weight=30,
            income_weight=25,  # Increased (income vs savings gap)
            real_estate_weight=10,
            pension_gap_weight=35,  # Increased
            age_weight=40,  # Increased
            employment_weight=25,
            mortgage_situation_weight=10,
            money_in_motion_weight=25
        )
    elif focus == StrategicFocus.MORTGAGE_GROWTH:
        return ScoringWeights(
            free_assets_weight=35,
            income_weight=25,
            real_estate_weight=20,  # Increased
            pension_gap_weight=20,
            age_weight=25,
            employment_weight=20,
            mortgage_situation_weight=35,  # Increased
            money_in_motion_weight=20
        )
    
    return weights  # BALANCED - use defaults


def _get_rating_details(total_score: float, thresholds: ScoringThresholds) -> tuple[ClientRating, str, str]:
    """Determine rating, label, and recommended action based on score."""
    if total_score >= thresholds.rating_a_threshold:
        return (
            ClientRating.A,
            "Top Opportunity",
            "Sofortige Kontaktaufnahme durch Senior Berater. Fokus auf ganzheitliche Vermögens- und Vorsorgeplanung."
        )
    elif total_score >= thresholds.rating_b_threshold:
        return (
            ClientRating.B,
            "Entwicklungspotenzial",
            "Aufnahme in Vorsorge-Optimierungskampagne. Mittelfristige Betreuung mit regelmässigem Kontakt."
        )
    else:
        return (
            ClientRating.C,
            "Standard",
            "Standard-Betreuung mit Fokus auf digitale Kanäle. Bei Lebensereignissen proaktiv ansprechen."
        )


def _extract_primary_triggers(score: OpportunityScore) -> list[str]:
    """Identify primary triggers from high-scoring components."""
    primary_triggers = []
    all_components = score.potential_components + score.trigger_components
    high_scoring = sorted(all_components, key=lambda x: x.points, reverse=True)[:3]
    
    for comp in high_scoring:
        if comp.points >= 6:
            primary_triggers.append(f"{comp.category}: {comp.reasoning}")
    
    return primary_triggers


def _determine_focus_areas(score: OpportunityScore) -> list[str]:
    """Determine focus areas based on component scores."""
    focus_areas = []
    
    for comp in score.potential_components:
        if comp.category == "Freie Vermögenswerte" and comp.points >= 6:
            focus_areas.append("Anlageberatung")
        if comp.category == "Vorsorgelücke" and comp.points >= 6:
            focus_areas.append("Vorsorgeplanung")
    
    for comp in score.trigger_components:
        if comp.category == "Hypothekarsituation" and comp.points >= 6:
            focus_areas.append("Hypothekenberatung")
    
    return focus_areas if focus_areas else ["Allgemeine Finanzberatung"]


def determine_rating(
    score: OpportunityScore,
    data: SteuererklaerungData,
    thresholds: ScoringThresholds = DEFAULT_THRESHOLDS
) -> RatingResult:
    """
    Determine client rating based on score and generate advisor narrative.
    """
    rating, rating_label, recommended_action = _get_rating_details(score.total_score, thresholds)
    primary_triggers = _extract_primary_triggers(score)
    focus_areas = _determine_focus_areas(score)
    
    all_components = score.potential_components + score.trigger_components
    high_scoring = sorted(all_components, key=lambda x: x.points, reverse=True)[:3]
    rationale = _generate_rating_rationale(rating, score, data, high_scoring)
    
    return RatingResult(
        rating=rating,
        score=score,
        rating_label=rating_label,
        rating_rationale=rationale,
        primary_triggers=primary_triggers,
        recommended_action=recommended_action,
        focus_areas=focus_areas
    )


def _generate_rating_rationale(
    rating: ClientRating,
    score: OpportunityScore,
    data: SteuererklaerungData,
    top_components: list
) -> str:
    """Generate a narrative explanation of the rating for advisors."""
    income = _calculate_total_income(data.person_1, data.person_2)
    assets = _calculate_total_free_assets(data.person_1, data.person_2)
    age = _get_primary_age(data.person_1, data.person_2)
    is_self_emp = _is_self_employed(data.person_1, data.person_2)
    
    parts = []
    
    # Opening based on rating
    if rating == ClientRating.A:
        parts.append(f"Dieser Kunde weist mit einem Score von {score.total_score:.0f}/100 ein herausragendes Potenzial auf.")
    elif rating == ClientRating.B:
        parts.append(f"Mit einem Score von {score.total_score:.0f}/100 zeigt dieser Kunde gutes Entwicklungspotenzial.")
    else:
        parts.append(f"Der Score von {score.total_score:.0f}/100 deutet auf ein standardmässiges Kundenprofil hin.")
    
    # Key findings
    if income > 150_000:
        parts.append(f"Das hohe Bruttoeinkommen von über CHF {income:,.0f} signalisiert erhebliche Spar- und Anlagekapazität.")
    
    if age and 55 <= age <= 63:
        parts.append(f"Mit {age} Jahren befindet sich der Kunde im kritischen Fenster für die Vorsorgeplanung.")
    
    if is_self_emp:
        parts.append("Als Selbstständiger bestehen komplexere Vorsorge- und potenzielle Nachfolgebedürfnisse.")
    
    if income > 100_000 and assets < 50_000:
        parts.append("Die Diskrepanz zwischen hohem Einkommen und tiefen sichtbaren Wertschriften deutet auf eine signifikante Vorsorgelücke hin.")
    
    return " ".join(parts)