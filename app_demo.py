"""
HBL FinDoc Analyzer - Streamlit Demo Application
Showcases Use Case 1 (Steuererklärung) and Use Case 2 (Document Set Pipeline)
"""
import os
import sys
import json
import logging
import base64
from io import BytesIO, StringIO
from datetime import datetime
from typing import Optional, Dict, Any, List
import plotly.graph_objects as go


# Suppress warnings before importing other modules
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress Arize Phoenix specific warnings
os.environ["ARIZE_DISABLE_WARNINGS"] = "1"
os.environ["PHOENIX_DISABLE_WARNINGS"] = "1"

# Suppress specific loggers for Phoenix/Arize
logging.getLogger("arize").setLevel(logging.ERROR)
logging.getLogger("phoenix").setLevel(logging.ERROR)
logging.getLogger("arize_phoenix").setLevel(logging.ERROR)
logging.getLogger("openinference").setLevel(logging.ERROR)



from sharepoint_service import SharePointService

import streamlit as st

try:
    import phoenix
    phoenix.logger.setLevel(logging.ERROR)
except (ImportError, AttributeError):
    pass

try:
    from phoenix.trace import suppress_tracing_warnings
    suppress_tracing_warnings()
except (ImportError, AttributeError):
    pass

# Configure page - must be first Streamlit command
st.set_page_config(
    page_title="HBL FinDoc Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide default sidebar
)

# Custom CSS for modern grid layout
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #f8f9fa;
        color: #000000;
    }
    
    /* Hide default sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Header styling */
    .app-header {
        background-color: #ffffff;
        padding: 1rem 2rem;
        border-bottom: 3px solid #007bc5;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .header-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 100%;
    }
    
    .header-logo {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .header-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #007bc5;
        margin: 0;
    }
    
    /* Parameters panel */
    .params-panel {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #007bc5;
        margin-bottom: 1.5rem;
        height: fit-content;
        position: sticky;
        top: 100px;
    }
    
    .params-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #007bc5;
    }
    
    .params-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #007bc5;
    }
    
    /* Main content card */
    .content-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Logs section */
    .logs-section {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-top: 3px solid #007bc5;
    }
    
    .log-container {
        background-color: #1b2136;
        border-radius: 8px;
        border: 2px solid #007bc5;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        max-height: 400px;
        overflow-y: auto;
        color: #000000;
    }
    
    /* Card styling */
    .metric-card {
        background-color: #007bc5;
        color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .score-card {
        background-color: #bcbcbc;
        color: #000000; 
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,123,197,0.2);
    }
    
    .score-a { border: 3px solid #282828; }
    .score-b { border: 3px solid #282828; }
    .score-c { border: 3px solid #282828; }
    
    /* Button styling */
    .stButton > button {
        background-color: #ffffff;
        color: #007bc5;
        border: 2px solid #007bc5;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #007bc5;
        color: #ffffff;
        border: 2px solid #007bc5;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,123,197,0.3);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #007bc5;
        color: #ffffff;
        border: 2px solid #007bc5;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #005a94;
        border: 2px solid #005a94;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border: 1px solid #007bc5;
        border-radius: 8px;
        color: #007bc5;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e7f3ff;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #0000000 !important;
    }
    
    .stSlider > div > div > div > div[role="slider"] {
        background-color: #ffffff !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background-color: #fffffff !important;
    }
    
    .stSlider [role="slider"] {
        background-color: #007bc5 !important;
    }
            
    .stSlider > div > div > div[data-baseweb="slider"] > div > div {
        color: #007bc5 !important;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border: 2px solid #007bc5;
        border-radius: 8px;
        padding: 10px 20px;
        color: #007bc5;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bc5;
        color: #ffffff;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border: 2px solid #007bc5;
        border-radius: 8px;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #007bc5;
        font-weight: bold;
    }
    
    /* Success/Warning/Error messages */
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Collapsible section toggle */
    .toggle-btn {
        background-color: #007bc5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .toggle-btn:hover {
        background-color: #005a94;
    }
    
    /* Info box */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #007bc5;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Mode selection
        "use_case": "Use Case 1: Steuererklärung",
        
        # File selection
        "selected_file": None,
        "pdf_bytes": None,
        
        # Processing state
        "processing": False,
        "processed": False,
        "logs": [],
        
        # Results
        "uc1_result": None,
        "uc1_docx_bytes": None,
        "uc1_docx_filename": None,
        
        "uc2_result": None,
        "uc2_split_files": [],
        "uc2_extracted_docs": [],
        
        # Scoring weights (Use Case 1)
        "free_assets_weight": 40,
        "income_weight": 20,
        "real_estate_weight": 15,
        "pension_gap_weight": 25,
        "age_weight": 35,
        "employment_weight": 25,
        "mortgage_situation_weight": 20,
        "money_in_motion_weight": 20,
        
        # Scoring thresholds
        "free_assets_high": 500000,
        "free_assets_medium": 100000,
        "income_high": 200000,
        "income_medium": 120000,
        "age_prime_start": 55,
        "age_prime_end": 63,
        "age_dev_start": 45,
        "age_dev_end": 54,
        "rating_a_threshold": 85,
        "rating_b_threshold": 60,
        
        # Strategic focus
        "strategic_focus": "balanced",
        
        # UI state
        "show_params": True,
        "show_logs": True
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==========================================
# LOGGING HANDLER
# ==========================================

class StreamlitLogHandler(logging.Handler):
    """Custom log handler that captures logs to session state."""
    
    def emit(self, record):
        _ = self.format(record)
        if "logs" in st.session_state:
            st.session_state.logs.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": record.levelname,
                "message": record.getMessage()
            })


class PrintCapture:
    """Context manager to capture print statements."""
    
    def __init__(self):
        self.logs = []
        
    def write(self, text):
        if text.strip():
            st.session_state.logs.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": "INFO",
                "message": text.strip()
            })
            # Update real-time log container if available
            if "log_container" in st.session_state and st.session_state.log_container is not None:
                _update_log_display(st.session_state.log_container)
        sys.__stdout__.write(text)
    
    def flush(self):
        sys.__stdout__.flush()


def setup_logging():
    """Setup logging to capture to session state."""
    if "logging_setup_done" in st.session_state:
        return
    
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    
    root_logger = logging.getLogger()
    
    for existing_handler in root_logger.handlers[:]:
        if isinstance(existing_handler, StreamlitLogHandler):
            root_logger.removeHandler(existing_handler)
    
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    st.session_state.logging_setup_done = True


def add_log(message: str, level: str = "INFO"):
    """Add a log entry to session state and update display."""
    st.session_state.logs.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "message": message
    })
    if "log_container" in st.session_state and st.session_state.log_container is not None:
        _update_log_display(st.session_state.log_container)


def clear_logs():
    """Clear all logs."""
    st.session_state.logs = []


def _render_realtime_logs():
    """Render logs in real-time mode."""
    log_container = st.empty()
    if st.session_state.logs:
        _update_log_display(log_container)
    return log_container


def _render_static_logs():
    """Render logs in static mode."""
    if st.session_state.logs:
        log_html = '<div class="log-container">'
        for log in st.session_state.logs:
            color = {
                "INFO": "#4dabe6",
                "WARNING": "#ffc107",
                "ERROR": "#dc3545",
                "SUCCESS": "#bcff36"
            }.get(log["level"], "#6c757d")
            
            log_html += f'<div style="color: {color}; margin: 2px 0;"><strong>[{log["time"]}]</strong> {log["message"]}</div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No logs yet. Click 'Process' to start analysis.")


# ==========================================
# FILE OPERATIONS
# ==========================================

def get_data_files() -> List[str]:
    """Get list of PDF files in the data directory."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []
    
    return [f for f in os.listdir(data_dir) if f.endswith('.pdf')]


def load_pdf_bytes(filepath: str) -> bytes:
    """Load PDF file as bytes."""
    with open(filepath, "rb") as f:
        return f.read()


def render_pdf_preview(pdf_bytes: bytes, height: int = 600):
    """Render PDF preview using base64 embedding."""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'''
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="{height}px" 
            type="application/pdf"
            style="border-radius: 8px; border: 2px solid #007bc5;">
        </iframe>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)


# ==========================================
# PLOTLY VISUALIZATION HELPERS
# ==========================================

def _build_radar_chart(score) -> go.Figure:
    """Build a radar chart summarizing all score components."""
    all_components = score.potential_components + score.trigger_components
    categories = [c.category for c in all_components]
    values = [c.points for c in all_components]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(0,123,197,0.15)',
        line=dict(color='#007bc5', width=2),
        marker=dict(size=6, color='#007bc5'),
        name='Score',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                gridcolor='rgba(0,0,0,0.08)',
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.08)',
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=40),
        height=480,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=16),
    )
    return fig


def _build_fill_bar(points: int, max_points: int, weighted_score: float, weight_percent: int) -> go.Figure:
    """Build a tiny horizontal bar showing the fill rate of a score component."""
    fill_pct = (points / max_points) * 100 if max_points > 0 else 0
    bar_color = '#28a745' if fill_pct >= 70 else '#ffc107' if fill_pct >= 40 else '#dc3545'

    fig = go.Figure()

    # Background bar (max)
    fig.add_trace(go.Bar(
        x=[100],
        y=[''],
        orientation='h',
        marker=dict(color='rgba(0,0,0,0.06)', line_width=0),
        showlegend=False,
        hoverinfo='skip',
    ))

    # Filled bar
    fig.add_trace(go.Bar(
        x=[fill_pct],
        y=[''],
        orientation='h',
        marker=dict(color=bar_color, line_width=0),
        showlegend=False,
        text=f'{points}/{max_points} pts  ({weighted_score:.1f}/{weight_percent} wt)',
        textposition='inside',
        textfont=dict(color='white', size=16),
        hovertemplate=f'Fill: {fill_pct:.0f}%<br>Points: {points}/{max_points}<br>Weighted: {weighted_score:.1f}/{weight_percent}<extra></extra>',
    ))

    fig.update_layout(
        barmode='overlay',
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=32,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _build_ltv_bar(property_value: float, mortgage_value: float) -> go.Figure:
    """Build a dedicated LTV (Loan-to-Value) bar chart for real estate."""
    ltv_pct = (mortgage_value / property_value * 100) if property_value > 0 else 0
    equity_pct = max(0, 100 - ltv_pct)

    # Swiss regulatory colour coding
    if ltv_pct <= 66.7:
        ltv_color = '#28a745'   # green – within 1st mortgage band
        label = 'Healthy (≤ 66.7%)'
    elif ltv_pct <= 80:
        ltv_color = '#ffc107'   # amber – 2nd mortgage band
        label = 'Moderate (≤ 80%)'
    else:
        ltv_color = "#d3718d"   # red – above regulatory comfort zone
        label = 'Elevated (> 80%)'

    # Dynamic x-axis range: always show at least up to 105%, but expand if LTV exceeds that
    x_max = max(105, ltv_pct + 5)

    fig = go.Figure()

    if ltv_pct <= 100:
        # Normal case: stacked mortgage + equity fills up to 100%
        fig.add_trace(go.Bar(
            x=[ltv_pct],
            y=['LTV'],
            orientation='h',
            marker=dict(color=ltv_color),
            name='Mortgage',
            text=f'Hypothek: {ltv_pct:.1f}%',
            textposition='inside' if ltv_pct > 20 else 'outside',
            textfont=dict(color='white' if ltv_pct > 20 else '#333', size=13),
            hovertemplate=f'Mortgage: CHF {mortgage_value:,.0f}<br>LTV: {ltv_pct:.1f}%<extra></extra>',
        ))
        fig.add_trace(go.Bar(
            x=[equity_pct],
            y=['LTV'],
            orientation='h',
            marker=dict(color='rgba(0,123,197,0.25)'),
            name='Equity',
            text=f'Eigenkapital: {equity_pct:.1f}%' if equity_pct > 12 else '',
            textposition='inside',
            textfont=dict(color='#333', size=13),
            hovertemplate=f'Equity: {equity_pct:.1f}%<extra></extra>',
        ))
    else:
        # Over-leveraged: single bar showing full LTV, no equity portion
        fig.add_trace(go.Bar(
            x=[100],
            y=['LTV'],
            orientation='h',
            marker=dict(color=ltv_color),
            name='Mortgage (100%)',
            text='',
            hovertemplate=f'Property value: CHF {property_value:,.0f}<extra></extra>',
        ))
        # The exceeding portion drawn on top in a darker shade
        fig.add_trace(go.Bar(
            x=[ltv_pct - 100],
            y=['LTV'],
            orientation='h',
            marker=dict(color='#a30000'),
            name='Over-leveraged',
            text=f'Hypothek: {ltv_pct:.1f}%',
            textposition='inside' if (ltv_pct - 100) > 10 else 'outside',
            textfont=dict(color='white' if (ltv_pct - 100) > 10 else '#333', size=13),
            hovertemplate=f'Mortgage: CHF {mortgage_value:,.0f}<br>LTV: {ltv_pct:.1f}%<br>Over-leveraged by {ltv_pct - 100:.1f}%<extra></extra>',
        ))

        # Swiss regulatory thresholds — drawn as shapes + free-standing annotations
    # so labels sit clearly above the bar with proper spacing
    for thresh_x, thresh_color, thresh_label, x_anchor in [
        (66.7, '#007bc5', '1. Hyp. 66.7%', 'right'),
        (80.0, "#3f3f3f", 'Max 80%', 'left'),
    ]:
        fig.add_shape(
            type='line',
            x0=thresh_x, x1=thresh_x,
            y0=-0.5, y1=0.5,
            yref='y',
            line=dict(color=thresh_color, width=1.5, dash='dash'),
        )
        fig.add_annotation(
            x=thresh_x,
            y=1,              # above the bar (y category index 0, so 1 is well above)
            yref='y',
            text=thresh_label,
            showarrow=False,
            font=dict(size=11, color=thresh_color, weight='bold'),
            xanchor=x_anchor,
            yanchor='bottom',
        )

    fig.update_layout(
        barmode='stack',
        xaxis=dict(range=[0, x_max], title='', ticksuffix='%', dtick=20),
        yaxis=dict(visible=False, range=[-0.5, 1.2]),   # extra room above for annotations
        margin=dict(l=0, r=10, t=40, b=30),
        height=130,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        title=dict(
            text=f'Belehnungsgrad (LTV): {ltv_pct:.1f}% — {label}',
            font=dict(size=14, color='#333'),
            x=0,
        ),
    )
    return fig


# ==========================================
# PROCESSING FUNCTIONS (Keep existing ones)
# ==========================================

def process_use_case_1(pdf_path: str):
    """Process Use Case 1: Steuererklärung Analysis."""
    from poc_hbl import extract_markdown_from_pdf, analyze_tax_potential
    from docx_generator_service import generate_steuererklaerung_report
    from scoring_engine import ScoringThresholds, ScoringWeights, StrategicFocus
    
    add_log("=" * 50)
    add_log("Starting Use Case 1: Steuererklärung Analysis")
    add_log("=" * 50)
    
    try:
        thresholds = ScoringThresholds(
            free_assets_high=st.session_state.free_assets_high,
            free_assets_medium=st.session_state.free_assets_medium,
            income_high=st.session_state.income_high,
            income_medium=st.session_state.income_medium,
            age_prime=(st.session_state.age_prime_start, st.session_state.age_prime_end),
            age_development=(st.session_state.age_dev_start, st.session_state.age_dev_end),
            rating_a_threshold=st.session_state.rating_a_threshold,
            rating_b_threshold=st.session_state.rating_b_threshold
        )
        
        weights = ScoringWeights(
            free_assets_weight=st.session_state.free_assets_weight,
            income_weight=st.session_state.income_weight,
            real_estate_weight=st.session_state.real_estate_weight,
            pension_gap_weight=st.session_state.pension_gap_weight,
            age_weight=st.session_state.age_weight,
            employment_weight=st.session_state.employment_weight,
            mortgage_situation_weight=st.session_state.mortgage_situation_weight,
            money_in_motion_weight=st.session_state.money_in_motion_weight
        )
        
        focus_map = {
            "balanced": StrategicFocus.BALANCED,
            "investment_growth": StrategicFocus.INVESTMENT_GROWTH,
            "pension_focus": StrategicFocus.PENSION_FOCUS,
            "mortgage_growth": StrategicFocus.MORTGAGE_GROWTH
        }
        strategic_focus = focus_map.get(st.session_state.strategic_focus, StrategicFocus.BALANCED)
        
        add_log("Step 1: Extracting markdown from PDF...")
        markdown_content = extract_markdown_from_pdf(pdf_path)
        add_log(f"  -> Extracted {len(markdown_content)} characters")
        
        add_log("Step 2: Analyzing with LLM...")
        from poc_hbl import get_steuererklaerung_prompt
        from clients import llm_client, MODEL_DEPLOYMENT
        from schemas import FuturePotentialLLMExtraction
        from scoring_engine import calculate_opportunity_score, determine_rating
        
        prompt = get_steuererklaerung_prompt()
        
        completion = llm_client.beta.chat.completions.parse(
            model=MODEL_DEPLOYMENT,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Steuererklärung:\n\n{markdown_content}"}
            ],
            response_format=FuturePotentialLLMExtraction,
            reasoning_effort="high",
        )
        
        llm_result = completion.choices[0].message.parsed
        add_log("  -> LLM analysis complete")
        
        add_log("Step 3: Calculating scores with custom weights...")
        score = calculate_opportunity_score(
            data=llm_result.steuererklaerung_data,
            strategic_focus=strategic_focus,
            thresholds=thresholds,
            weights=weights
        )
        
        rating_result = determine_rating(
            score=score,
            data=llm_result.steuererklaerung_data,
            thresholds=thresholds
        )
        
        from schemas import FuturePotential
        analysis_result = FuturePotential(
            rating_result=rating_result,
            steuererklaerung_data=llm_result.steuererklaerung_data,
            detected_assets=llm_result.detected_assets,
            pension_indicators=llm_result.pension_indicators,
            business_opportunities=llm_result.business_opportunities,
            summary=llm_result.summary,
            strategic_recommendations=llm_result.strategic_recommendations
        )
        
        add_log(f"  -> Rating: {rating_result.rating.value} ({rating_result.rating_label})")
        add_log(f"  -> Total Score: {score.total_score}")
        
        add_log("Step 4: Generating DOCX report...")
        try:
            docx_bytes, filename = generate_steuererklaerung_report(analysis_result)
            add_log(f"  -> Report generated: {filename}")
            st.session_state.uc1_docx_bytes = docx_bytes
            st.session_state.uc1_docx_filename = filename
        except FileNotFoundError as e:
            add_log("  -> WARNING: Template not found. Run 'python make_template.py' first.", "WARNING")
            add_log(f"     {e}", "WARNING")
        
        st.session_state.uc1_result = analysis_result
        add_log("=" * 50)
        add_log("Use Case 1 processing complete!", "SUCCESS")
        
        return True
        
    except Exception as e:
        add_log(f"ERROR: {str(e)}", "ERROR")
        import traceback
        add_log(traceback.format_exc(), "ERROR")
        return False


def process_use_case_2(pdf_path: str):
    """Process Use Case 2: Document Set Pipeline."""
    from poc_hbl import (
        extract_markdown_from_pdf,
        _process_markdown_and_boundaries,
        split_and_save,
        extract_data_from_splits
    )
    
    add_log("=" * 50)
    add_log("Starting Use Case 2: Document Set Pipeline")
    add_log("=" * 50)
    
    try:
        cache_path = f"cache_markdown/{os.path.basename(pdf_path)}.md"
        
        if not os.path.exists(cache_path):
            add_log("Step 1: Extracting markdown from PDF...")
            extract_markdown_from_pdf(pdf_path)
            add_log("  -> Markdown extracted")
        else:
            add_log("Step 1: Using cached markdown")
        
        add_log("Step 2: Detecting document boundaries...")
        from split import parse_azure_markdown
        
        md_pages = parse_azure_markdown(cache_path)
        add_log(f"  -> Found {len(md_pages)} pages")
        
        from poc_hbl import detect_document_boundaries
        boundaries = detect_document_boundaries(md_pages)
        
        add_log(f"  -> Detected {len(boundaries)} documents:")
        for i, b in enumerate(boundaries):
            add_log(f"     {i+1}. {b['doc_type']} (Pages {b['start_page']+1}-{b['end_page']+1})")
        
        if not boundaries:
            add_log("No documents detected!", "WARNING")
            return False
        
        add_log("Step 3: Splitting PDF and saving files...")
        output_dir = "output_splits"
        split_files = split_and_save(pdf_path, md_pages, boundaries, output_dir)
        add_log(f"  -> Created {len(split_files)} split files")
        
        add_log("Step 4: Extracting structured data from splits...")
        extracted_docs = extract_data_from_splits(split_files)
        add_log(f"  -> Extracted data from {len(extracted_docs)} documents")
        
        st.session_state.uc2_split_files = split_files
        st.session_state.uc2_extracted_docs = extracted_docs
        st.session_state.uc2_result = {
            "boundaries": boundaries,
            "split_files": split_files,
            "extracted_docs": [doc.model_dump() for doc in extracted_docs]
        }
        
        add_log("=" * 50)
        add_log("Use Case 2 processing complete!", "SUCCESS")
        
        return True
        
    except Exception as e:
        add_log(f"ERROR: {str(e)}", "ERROR")
        import traceback
        add_log(traceback.format_exc(), "ERROR")
        return False


# ==========================================
# UI COMPONENTS - HEADER
# ==========================================

def render_header():
    """Render the application header."""
    logo_path = "hbl.png"  # Ensure you have a logo image at this path or update accordingly
    
    col1, col2, col3, col4 = st.columns([1.5, 2, 2.5, 2])
    
    with col1:
        if os.path.exists(logo_path):
            st.image(logo_path, width=300)
        else:
            st.markdown("### 🏦 Demo AG")
    
    with col2:
        # Mode selector
        use_case = st.radio(
            "Mode",
            ["Use Case 1: Steuererklärung", "Use Case 2: Document Set"],
            horizontal=True,
            key="use_case_radio",
            label_visibility="collapsed"
        )
        st.session_state.use_case = use_case
        
    with col3:
        # File selector
        pdf_files = get_data_files()
        if pdf_files:
            if "Steuererklärung" in st.session_state.use_case:
                suggested = [f for f in pdf_files if "steuer" in f.lower()]
            else:
                suggested = [f for f in pdf_files if "dok" in f.lower() or "set" in f.lower()]
            
            if suggested:
                pdf_files = suggested + [f for f in pdf_files if f not in suggested]
            
            selected = st.selectbox(
                "Select Document",
                pdf_files,
                key="file_selector",
                label_visibility="collapsed"
            )
            
            if selected:
                st.session_state.selected_file = os.path.join("data", selected)
                st.session_state.pdf_bytes = load_pdf_bytes(st.session_state.selected_file)
        else:
            st.warning("No PDF files in data/ folder")
    
    with col4:
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Process", type="primary", width='content'):
                st.session_state.processing = True
                st.session_state.processed = False
                clear_logs()
                st.rerun()
        
        with col_btn2:
            if st.button("🗑️ Clear", width='content'):
                clear_results()
                st.rerun()
    
    st.markdown("---")

# ==========================================
# UI COMPONENTS - PARAMETERS PANEL
# ==========================================

def render_parameters_panel():
    """Render the scoring parameters panel (for Use Case 1)."""
    # Remove the container wrapper and just use direct Streamlit components with custom styling
    
    st.markdown("### ⚙️ Scoring Parameters")
    
    # Strategic focus
    st.selectbox(
        "Strategic Focus",
        ["balanced", "investment_growth", "pension_focus", "mortgage_growth"],
        format_func=lambda x: {
            "balanced": "🎯 Balanced",
            "investment_growth": "📈 Investment Growth",
            "pension_focus": "🏖️ Pension Focus",
            "mortgage_growth": "🏠 Mortgage Growth"
        }.get(x, x),
        key="strategic_focus"
    )
    
    st.markdown("---")
    
    # Potential Weights
    with st.expander("📊 Potential Weights", expanded=False):
        st.caption("*Must sum to 100*")
        st.slider("Free Assets", 0, 100, key="free_assets_weight")
        st.slider("Income", 0, 100, key="income_weight")
        st.slider("Real Estate", 0, 100, key="real_estate_weight")
        st.slider("Pension Gap", 0, 100, key="pension_gap_weight")
        
        total_potential = (
            st.session_state.free_assets_weight + 
            st.session_state.income_weight + 
            st.session_state.real_estate_weight + 
            st.session_state.pension_gap_weight
        )
        if total_potential != 100:
            st.warning(f"Total: {total_potential}/100")
        else:
            st.success(f"Total: {total_potential}/100 ✓")
    
    # Trigger Weights
    with st.expander("🎯 Trigger Weights", expanded=False):
        st.caption("*Must sum to 100*")
        st.slider("Age", 0, 100, key="age_weight")
        st.slider("Employment", 0, 100, key="employment_weight")
        st.slider("Mortgage Situation", 0, 100, key="mortgage_situation_weight")
        st.slider("Money in Motion", 0, 100, key="money_in_motion_weight")
        
        total_trigger = (
            st.session_state.age_weight + 
            st.session_state.employment_weight + 
            st.session_state.mortgage_situation_weight + 
            st.session_state.money_in_motion_weight
        )
        if total_trigger != 100:
            st.warning(f"Total: {total_trigger}/100")
        else:
            st.success(f"Total: {total_trigger}/100 ✓")
    
    # Thresholds
    with st.expander("📏 Thresholds", expanded=False):
        st.markdown("**Asset Thresholds (CHF)**")
        st.number_input("Free Assets High", value=500000, step=50000, key="free_assets_high")
        st.number_input("Free Assets Medium", value=100000, step=10000, key="free_assets_medium")
        
        st.markdown("**Income Thresholds (CHF)**")
        st.number_input("Income High", value=200000, step=10000, key="income_high")
        st.number_input("Income Medium", value=120000, step=10000, key="income_medium")
        
        st.markdown("**Age Brackets**")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Prime Start", value=55, key="age_prime_start")
            st.number_input("Dev Start", value=45, key="age_dev_start")
        with col2:
            st.number_input("Prime End", value=63, key="age_prime_end")
            st.number_input("Dev End", value=54, key="age_dev_end")
        
        st.markdown("**Rating Thresholds**")
        st.slider("Rating A (>=)", 0, 100, key="rating_a_threshold")
        st.slider("Rating B (>=)", 0, 100, key="rating_b_threshold")


# ==========================================
# UI COMPONENTS - RESULTS
# ==========================================

def _render_rating_overview(rating, score):
    """Render rating overview cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    rating_color = {
        "A": "#28a745",
        "B": "#ffc107",
        "C": "#282828"
    }.get(rating.rating.value, "#6c757d")
    
    with col1:
        st.markdown(f"""
            <div class="score-card score-{rating.rating.value.lower()}">
                <h1 style="color: {rating_color}; margin: 0;">{rating.rating.value}</h1>
                <p style="margin: 0; font-size: 1.1rem;">{rating.rating_label}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Total Score", f"{score.total_score:.0f}/100", help="Overall opportunity score")
    
    with col3:
        st.metric("Potential Score", f"{score.potential_score:.0f}/100", help="Financial potential")
    
    with col4:
        st.metric("Trigger Score", f"{score.trigger_score:.0f}/100", help="Life event triggers")


def _render_score_component(comp):
    """Render a single score component with an inline fill-rate sparkline."""
    with st.expander(f"{comp.category}: {comp.points}/{comp.max_points} pts"):
        st.plotly_chart(
            _build_fill_bar(comp.points, comp.max_points, comp.weighted_score, comp.weight_percent),
            width='content',
            config={'displayModeBar': False},
            key=f"fill_bar_{comp.category}_{comp.weight_percent}",
        )
        st.write(f"**Raw Value:** {comp.raw_value}")
        st.write(f"**Weight:** {comp.weight_percent}%")
        st.write(f"**Weighted Score:** {comp.weighted_score:.1f}")
        st.write(f"**Reasoning:** {comp.reasoning}")


def _parse_monetary_value(value_str: str) -> float:
    """Parse a monetary string to float, handling various formats."""
    if not value_str:
        return 0.0
    try:
        cleaned = value_str.replace("CHF", "").replace("'", "").replace(",", "").replace("\u2009", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return 0.0


def _accumulate_person_values(person) -> tuple[float, float]:
    """Accumulate property and mortgage values for a single person."""
    property_val = _parse_monetary_value(person.liegenschaften) if person and person.liegenschaften else 0.0
    mortgage_val = _parse_monetary_value(person.hypothekarschulden) if person and person.hypothekarschulden else 0.0
    return property_val, mortgage_val


def _render_ltv_section(result):
    """Render the LTV bar if real-estate and mortgage data exist."""
    data = result.steuererklaerung_data

    total_property = 0.0
    total_mortgage = 0.0
    for person in [data.person_1, data.person_2]:
        if person:
            prop_val, mort_val = _accumulate_person_values(person)
            total_property += prop_val
            total_mortgage += mort_val

    if total_property > 0 or total_mortgage > 0:
        st.markdown("#### 🏠 Loan-to-Value (Belehnungsgrad)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Liegenschaftswert", f"CHF {total_property:,.0f}".replace(",", "'"))
        with col2:
            st.metric("Hypothekarschuld", f"CHF {total_mortgage:,.0f}".replace(",", "'"))
        with col3:
            ltv = (total_mortgage / total_property * 100) if total_property > 0 else 0
            st.metric("LTV", f"{ltv:.1f}%")
        st.plotly_chart(
            _build_ltv_bar(total_property, total_mortgage),
            use_container_width=True,
            config={'displayModeBar': False},
            key="ltv_bar_chart",
        )
    else:
        st.info("Keine Immobilien- oder Hypothekardaten vorhanden.")

def _render_score_details_tab(score, result=None):
    """Render the score details tab with radar chart, sparklines, and LTV bar."""
    st.markdown("#### 🕸️ Score Profile Overview")
    st.plotly_chart(
        _build_radar_chart(score),
        width='content',
        config={'displayModeBar': False},
        key="radar_chart_overview",
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💰 Potential Components")
        for comp in score.potential_components:
            _render_score_component(comp)

    with col2:
        st.markdown("#### 🎯 Trigger Components")
        for comp in score.trigger_components:
            _render_score_component(comp)

    if result is not None:
        st.markdown("---")
        _render_ltv_section(result)


def _render_person_master_data(person):
    """Render master data section for a person."""
    if not person.master_data:
        return
    
    st.markdown("**📝 Personal Information**")
    if person.master_data.name:
        st.write(f"**Name:** {person.master_data.name}")
    if person.master_data.geburtsdatum:
        st.write(f"**Birth Date:** {person.master_data.geburtsdatum}")
    if person.master_data.alter:
        st.write(f"**Age:** {person.master_data.alter}")
    st.write(f"**Employment Status:** {person.master_data.employment_status.value}")
    st.markdown("---")


def _render_person_income(person):
    """Render income section for a person."""
    st.markdown("**💰 Income**")
    if person.haupterwerb:
        st.write(f"**Gross Income:** {person.haupterwerb}")
    if person.nettoeinkommen:
        st.write(f"**Net Income:** {person.nettoeinkommen}")
    if person.steuerbares_einkommen:
        st.write(f"**Taxable Income:** {person.steuerbares_einkommen}")
    st.markdown("---")


def _render_person_assets(person):
    """Render assets section for a person."""
    st.markdown("**🏦 Assets**")
    if person.wertschriften:
        st.write(f"**Securities:** {person.wertschriften}")
    if person.bankguthaben:
        st.write(f"**Bank Balance:** {person.bankguthaben}")
    if person.liegenschaften:
        st.write(f"**Real Estate:** {person.liegenschaften}")
    if person.lebens_und_versicherungspolicen:
        st.write(f"**Insurance Policies:** {person.lebens_und_versicherungspolicen}")
    st.markdown("---")


def _render_person_liabilities(person):
    """Render liabilities section for a person."""
    st.markdown("**💳 Liabilities**")
    if person.schulden:
        st.write(f"**Debts:** {person.schulden}")
    if person.hypothekarschulden:
        st.write(f"**Mortgage Debt:** {person.hypothekarschulden}")
    st.markdown("---")


def _render_person_pension(person):
    """Render pension section for a person."""
    st.markdown("**🏖️ Pension Contributions**")
    if person.saeule_3a_einzahlung:
        st.write(f"**Pillar 3a:** {person.saeule_3a_einzahlung}")
    if person.pk_einkauf:
        st.write(f"**Pension Fund Purchase:** {person.pk_einkauf}")
    if person.unterhaltsbeitraege:
        st.write(f"**Maintenance Payments:** {person.unterhaltsbeitraege}")


def _render_person_details(person, person_number):
    """Render all details for a person."""
    st.markdown(f"### 👤 Person {person_number}")
    if person:
        _render_person_master_data(person)
        _render_person_income(person)
        _render_person_assets(person)
        _render_person_liabilities(person)
        _render_person_pension(person)
    else:
        st.info(f"No {'data available for Person 1' if person_number == 1 else 'second person'}")


def _render_general_info(data):
    """Render general information section."""
    st.markdown("### 📋 General Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Marital Status", "Married" if data.verheiratet else "Single")
    with col2:
        st.metric("Housing", data.housing_situation.value)
    with col3:
        st.metric("Tax Year", data.tax_year)
    st.markdown("---")


def _render_pension_gap_analysis(pension):
    """Render pension gap analysis section."""
    st.markdown("---")
    st.markdown("### 🎯 Pension Gap Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pillar 3a Used", "✅ Ja" if pension.saeule_3a_genutzt else "❌ Nein")
    with col2:
        if pension.saeule_3a_voll_ausgeschoepft is not None:
            st.metric("3a Fully Utilized", "✅ Ja" if pension.saeule_3a_voll_ausgeschoepft else "❌ Nein")
    with col3:
        st.metric("Pension Fund Purchases", "✅ Ja" if pension.pk_einkauf_erkennbar else "❌ Nein")
    
    if pension.high_income_low_savings_indicator:
        st.warning("⚠️ **High Income / Low Savings Indicator:** Dieser Kunde könnte eine erhebliche Vorsorgelücke aufweisen.")
    
    if pension.estimated_pension_gap:
        st.info(f"**Estimated Pension Gap:** {pension.estimated_pension_gap}")


def _render_client_data_tab(result):
    """Render the client data tab."""
    data = result.steuererklaerung_data
    
    _render_general_info(data)
    
    col1, col2 = st.columns(2)
    with col1:
        _render_person_details(data.person_1, 1)
    with col2:
        _render_person_details(data.person_2, 2)
    
    _render_pension_gap_analysis(result.pension_indicators)


def _render_opportunity(opp, urgency_colors):
    """Render a single business opportunity."""
    urgency_icon = urgency_colors.get(opp.urgency.lower(), "⚪")
    
    with st.expander(f"{urgency_icon} {opp.title}"):
        st.write(f"**Description:** {opp.description}")
        
        if opp.estimated_potential:
            st.write(f"**Estimated Potential:** {opp.estimated_potential}")
        
        st.write(f"**Urgency:** {opp.urgency}")
        
        st.markdown("**Next Steps:**")
        for i, step in enumerate(opp.next_steps, 1):
            st.write(f"{i}. {step}")


def _render_opportunities_tab(result, rating):
    """Render the opportunities tab."""
    st.markdown("### 💼 Business Opportunities")
    
    if result.business_opportunities:
        opportunities_by_type = {}
        for opp in result.business_opportunities:
            opp_type = opp.opportunity_type
            if opp_type not in opportunities_by_type:
                opportunities_by_type[opp_type] = []
            opportunities_by_type[opp_type].append(opp)
        
        type_icons = {
            "Anlage": "📈", "Vorsorge": "🏖️", "Hypothek": "🏠",
            "Investment": "📈", "Pension": "🏖️", "Mortgage": "🏠"
        }
        
        urgency_colors = {
            "hoch": "🔴", "mittel": "🟡", "niedrig": "🟢",
            "high": "🔴", "medium": "🟡", "low": "🟢"
        }
        
        for opp_type, opps in opportunities_by_type.items():
            st.markdown(f"#### {type_icons.get(opp_type, '💡')} {opp_type}")
            for opp in opps:
                _render_opportunity(opp, urgency_colors)
            st.markdown("---")
    else:
        st.info("No specific opportunities identified.")
    
    st.markdown("### 📝 Executive Summary")
    st.write(result.summary)
    
    st.markdown("### 🎯 Strategic Recommendations")
    st.write(result.strategic_recommendations)
    
    st.markdown("### 🔍 Recommended Focus Areas")
    for area in rating.focus_areas:
        st.write(f"- {area}")


def _render_report_tab():
    """Render the report download tab."""
    if st.session_state.uc1_docx_bytes:
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download DOCX Report",
                data=st.session_state.uc1_docx_bytes,
                file_name=st.session_state.uc1_docx_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                width='content'
            )
        
        with col2:
            if st.button("☁️ Upload to SharePoint", key="sp_upload_uc1", width='content'):
                upload_to_sharepoint_uc1()
        
        st.success("✅ Report ready for download")
    else:
        st.warning("📋 DOCX report not generated. Please run 'python make_template.py' first.")


def render_uc1_results():
    """Render Use Case 1 results."""
    if not st.session_state.uc1_result:
        return
    
    result = st.session_state.uc1_result
    rating = result.rating_result
    score = rating.score
    
    st.markdown("## 📊 Analysis Results")
    
    _render_rating_overview(rating, score)
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Score Details", "👤 Client Data", "💼 Opportunities", "📄 Report"])
    
    with tab1:
        _render_score_details_tab(score, result=result)   # <-- pass result here
    
    with tab2:
        _render_client_data_tab(result)
    
    with tab3:
        _render_opportunities_tab(result, rating)
    
    with tab4:
        _render_report_tab()


def render_uc2_results():
    """Render Use Case 2 results."""
    if not st.session_state.uc2_result:
        return
    
    split_files = st.session_state.uc2_split_files
    extracted_docs = st.session_state.uc2_extracted_docs
    
    st.markdown("## 📑 Split Documents")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents Found", len(split_files))
    with col2:
        st.metric("Data Extracted", len(extracted_docs))
    with col3:
        doc_types = set(f["doc_type"] for f in split_files)
        st.metric("Document Types", len(doc_types))
    with col4:
        if st.button("☁️ Upload All to SharePoint", key="sp_upload_all_uc2", width='content'):
            upload_all_uc2_to_sharepoint()
    
    st.markdown("---")
    
    for i, split_file in enumerate(split_files):
        with st.expander(f"📄 {i+1}. {split_file['doc_type']} (Pages {split_file['start_page']}-{split_file['end_page']})"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**PDF Preview**")
                if os.path.exists(split_file['pdf_path']):
                    pdf_bytes = load_pdf_bytes(split_file['pdf_path'])
                    render_pdf_preview(pdf_bytes, height=400)
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_bytes,
                            file_name=os.path.basename(split_file['pdf_path']),
                            mime="application/pdf",
                            key=f"dl_pdf_{i}",
                            width='content'
                        )
                    with btn_col2:
                        if st.button("☁️ Upload PDF", key=f"sp_pdf_{i}", width='content'):
                            upload_to_sharepoint_uc2("pdf", i)
            
            with col2:
                st.markdown("**Extracted Data (JSON)**")
                if i < len(extracted_docs):
                    doc_data = extracted_docs[i].model_dump()
                    st.json(doc_data)
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        st.download_button(
                            label="📥 Download JSON",
                            data=json.dumps(doc_data, indent=2, ensure_ascii=False),
                            file_name=f"{os.path.basename(split_file['pdf_path']).replace('.pdf', '')}_data.json",
                            mime="application/json",
                            key=f"dl_json_{i}",
                            width='content'
                        )
                    with btn_col2:
                        if st.button("☁️ Upload JSON", key=f"sp_json_{i}", width='content'):
                            upload_to_sharepoint_uc2("json", i)
                else:
                    st.warning("No extracted data available for this document")

def _render_log_toggle_buttons():
    """Render toggle and clear buttons for logs."""
    col1, col2, col3 = st.columns([1, 5, 1])
    
    with col1:
        toggle_label = "📤 Hide Logs" if st.session_state.show_logs else "📥 Show Logs"
        if st.button(toggle_label, key="toggle_logs_btn"):
            st.session_state.show_logs = not st.session_state.show_logs
            # Don't rerun if currently processing - it would restart the process
            if not st.session_state.processing:
                st.rerun()
    
    with col3:
        if st.session_state.show_logs and st.button("🗑️ Clear", key="clear_logs_btn"):
            clear_logs()
            # Don't rerun if currently processing
            if not st.session_state.processing:
                st.rerun()


def _render_static_logs():
    """Render logs statically."""
    if not st.session_state.logs:
        st.info("ℹ️ No logs yet. Click 'Process' to start analysis.")
        return
    
    log_html = '<div class="log-container">'
    for log in st.session_state.logs:
        color = {
            "INFO": "#4dabe6",
            "WARNING": "#ffc107",
            "ERROR": "#dc3545",
            "SUCCESS": "#bcff36"
        }.get(log["level"], "#6c757d")
        
        log_html += f'<div style="color: {color}; margin: 2px 0;"><strong>[{log["time"]}]</strong> {log["message"]}</div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)


def render_log_section(realtime: bool = False):
    """Render the log section.
    
    Args:
        realtime: If True, returns an empty container for real-time updates
    """
    _render_log_toggle_buttons()
    
    if not st.session_state.show_logs:
        return None
    
    st.markdown("### 📋 Processing Logs")
    
    if realtime:
        log_container = st.empty()
        if st.session_state.logs:
            _update_log_display(log_container)
        return log_container
    
    _render_static_logs()
    return None



def clear_results():
    """Clear all results and reset state."""
    st.session_state.processed = False
    st.session_state.processing = False
    st.session_state.uc1_result = None
    st.session_state.uc1_docx_bytes = None
    st.session_state.uc1_docx_filename = None
    st.session_state.uc2_result = None
    st.session_state.uc2_split_files = []
    st.session_state.uc2_extracted_docs = []
    clear_logs()


def upload_to_sharepoint_uc1():
    """Upload Use Case 1 DOCX report to SharePoint."""
    if not st.session_state.uc1_docx_bytes:
        st.error("No report to upload. Generate a report first.")
        return
    
    try:
        sp_service = SharePointService()
        
        result = sp_service.upload_file(
            filename=st.session_state.uc1_docx_filename,
            content=st.session_state.uc1_docx_bytes,
            folder_path="UseCase-1"
        )
        
        if result["success"]:
            st.success(f"✅ Uploaded to SharePoint: {result.get('name')}")
            st.markdown(f"📎 [Open in SharePoint]({result.get('web_url')})")
            add_log(f"Uploaded to SharePoint: {result.get('web_url')}", "SUCCESS")
        else:
            st.error(f"❌ Upload failed: {result.get('error')}")
            add_log(f"SharePoint upload failed: {result.get('error')}", "ERROR")
            
    except Exception as e:
        st.error(f"❌ Upload error: {str(e)}")
        add_log(f"SharePoint upload error: {str(e)}", "ERROR")



def upload_to_sharepoint_uc2(file_type: str, index: int):
    """Upload Use Case 2 files to SharePoint.
    
    Args:
        file_type: "pdf" or "json"
        index: Index of the split file
    """
    split_files = st.session_state.uc2_split_files
    extracted_docs = st.session_state.uc2_extracted_docs
    
    if index >= len(split_files):
        st.error("Invalid file index")
        return
    
    split_file = split_files[index]
    
    try:
        sp_service = SharePointService()
        
        if file_type == "pdf":
            pdf_path = split_file['pdf_path']
            if not os.path.exists(pdf_path):
                st.error("PDF file not found")
                return
            
            with open(pdf_path, "rb") as f:
                content = f.read()
            
            filename = os.path.basename(pdf_path)
            
        elif file_type == "json":
            if index >= len(extracted_docs):
                st.error("No extracted data for this document")
                return
            
            doc_data = extracted_docs[index].model_dump()
            content = json.dumps(doc_data, indent=2, ensure_ascii=False).encode('utf-8')
            filename = f"{os.path.basename(split_file['pdf_path']).replace('.pdf', '')}_data.json"
        else:
            st.error("Invalid file type")
            return
        
        result = sp_service.upload_file(
            filename=filename,
            content=content,
            folder_path="UseCase-2"
        )
        
        if result["success"]:
            st.success(f"✅ Uploaded: {result.get('name')}")
            st.markdown(f"📎 [Open in SharePoint]({result.get('web_url')})")
            add_log(f"Uploaded to SharePoint: {result.get('web_url')}", "SUCCESS")
        else:
            st.error(f"❌ Upload failed: {result.get('error')}")
            add_log(f"SharePoint upload failed: {result.get('error')}", "ERROR")
            
    except Exception as e:
        st.error(f"❌ Upload error: {str(e)}")
        add_log(f"SharePoint upload error: {str(e)}", "ERROR")

def _upload_pdf_to_sharepoint(sp_service, pdf_path, status_text, index, total):
    """Upload a single PDF file to SharePoint."""
    if not os.path.exists(pdf_path):
        return None
    
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    
    status_text.text(f"Uploading PDF {index+1}/{total}...")
    result = sp_service.upload_file(
        filename=os.path.basename(pdf_path),
        content=pdf_content,
        folder_path="UseCase-2"
    )
    
    return result


def _upload_json_to_sharepoint(sp_service, doc_data, pdf_path, status_text, index, total):
    """Upload a single JSON file to SharePoint."""
    json_content = json.dumps(doc_data, indent=2, ensure_ascii=False).encode('utf-8')
    json_filename = f"{os.path.basename(pdf_path).replace('.pdf', '')}_data.json"
    
    status_text.text(f"Uploading JSON {index+1}/{total}...")
    result = sp_service.upload_file(
        filename=json_filename,
        content=json_content,
        folder_path="UseCase-2"
    )
    
    return result


def _show_upload_summary(success_count, error_count, uploaded_urls):
    """Display upload summary and links."""
    if error_count == 0:
        st.success(f"✅ Successfully uploaded {success_count} files to SharePoint")
        add_log(f"Uploaded {success_count} files to SharePoint", "SUCCESS")
    else:
        st.warning(f"⚠️ Uploaded {success_count} files, {error_count} failed")
        add_log(f"Partial upload: {success_count} success, {error_count} failed", "WARNING")
    
    if uploaded_urls:
        with st.expander("📎 View uploaded files in SharePoint", expanded=True):
            for name, url in uploaded_urls:
                st.markdown(f"- [{name}]({url})")


def upload_all_uc2_to_sharepoint():
    """Upload all Use Case 2 files to SharePoint."""
    split_files = st.session_state.uc2_split_files
    extracted_docs = st.session_state.uc2_extracted_docs
    
    if not split_files:
        st.error("No files to upload")
        return
    
    try:
        sp_service = SharePointService()
        success_count = 0
        error_count = 0
        uploaded_urls = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(split_files) * 2  # PDF + JSON for each
        current = 0
        
        for i, split_file in enumerate(split_files):
            # Upload PDF
            pdf_path = split_file['pdf_path']
            result = _upload_pdf_to_sharepoint(sp_service, pdf_path, status_text, i, len(split_files))
            
            if result and result["success"]:
                success_count += 1
                uploaded_urls.append((result.get('name'), result.get('web_url')))
            elif result:
                error_count += 1
            
            current += 1
            progress_bar.progress(current / total_files)
            
            # Upload JSON
            if i < len(extracted_docs):
                doc_data = extracted_docs[i].model_dump()
                result = _upload_json_to_sharepoint(sp_service, doc_data, pdf_path, status_text, i, len(split_files))
                
                if result["success"]:
                    success_count += 1
                    uploaded_urls.append((result.get('name'), result.get('web_url')))
                else:
                    error_count += 1
            
            current += 1
            progress_bar.progress(current / total_files)
        
        progress_bar.empty()
        status_text.empty()
        
        _show_upload_summary(success_count, error_count, uploaded_urls)
            
    except Exception as e:
        st.error(f"❌ Upload error: {str(e)}")
        add_log(f"SharePoint upload error: {str(e)}", "ERROR")

# ==========================================
# LOGGING HANDLER
# ==========================================

def _update_log_display(container):
    """Update the log display container with current logs."""
    if not st.session_state.logs:
        return
    
    log_html = '<div class="log-container" style="background-color: #1b2136; border-radius: 8px; border: 2px solid #007bc5; padding: 1rem; font-family: monospace; font-size: 12px; max-height: 400px; overflow-y: auto;">'
    for log in st.session_state.logs:
        color = {
            "INFO": "#4dabe6",
            "WARNING": "#ffc107",
            "ERROR": "#dc3545",
            "SUCCESS": "#bcff36"
        }.get(log["level"], "#6c757d")
        
        log_html += f'<div style="color: {color}; margin: 2px 0;"><strong>[{log["time"]}]</strong> {log["message"]}</div>'
    log_html += '</div>'
    
    container.markdown(log_html, unsafe_allow_html=True)


class StreamlitLogHandler(logging.Handler):
    """Custom log handler that captures logs to session state."""
    
    def emit(self, record):
        _ = self.format(record)
        if "logs" in st.session_state:
            st.session_state.logs.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": record.levelname,
                "message": record.getMessage()
            })


class PrintCapture:
    """Context manager to capture print statements."""
    
    def __init__(self):
        self.logs = []
        
    def write(self, text):
        if text.strip():
            st.session_state.logs.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": "INFO",
                "message": text.strip()
            })
            # Update real-time log container if available
            if "log_container" in st.session_state and st.session_state.log_container is not None:
                _update_log_display(st.session_state.log_container)
        sys.__stdout__.write(text)
    
    def flush(self):
        sys.__stdout__.flush()


def setup_logging():
    """Setup logging to capture to session state."""
    if "logging_setup_done" in st.session_state:
        return
    
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    
    root_logger = logging.getLogger()
    
    for existing_handler in root_logger.handlers[:]:
        if isinstance(existing_handler, StreamlitLogHandler):
            root_logger.removeHandler(existing_handler)
    
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    st.session_state.logging_setup_done = True


def add_log(message: str, level: str = "INFO"):
    """Add a log entry to session state and update display."""
    st.session_state.logs.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "message": message
    })
    if "log_container" in st.session_state and st.session_state.log_container is not None:
        _update_log_display(st.session_state.log_container)


def clear_logs():
    """Clear all logs."""
    st.session_state.logs = []


def _render_realtime_logs():
    """Render logs in real-time mode."""
    log_container = st.empty()
    if st.session_state.logs:
        _update_log_display(log_container)
    return log_container


def _render_static_logs():
    """Render logs in static mode."""
    if st.session_state.logs:
        log_html = '<div class="log-container">'
        for log in st.session_state.logs:
            color = {
                "INFO": "#4dabe6",
                "WARNING": "#ffc107",
                "ERROR": "#dc3545",
                "SUCCESS": "#bcff36"
            }.get(log["level"], "#6c757d")
            
            log_html += f'<div style="color: {color}; margin: 2px 0;"><strong>[{log["time"]}]</strong> {log["message"]}</div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No logs yet. Click 'Process' to start analysis.")


# ==========================================
# MAIN APPLICATION
# ==========================================

def _handle_processing(log_container, status_placeholder):
    """Handle document processing workflow.
    
    Args:
        log_container: The st.empty() container for real-time log updates
        status_placeholder: The st.empty() container for status message at the top
    """
    if not (st.session_state.processing and not st.session_state.processed):
        return False
    
    # Store log container in session state for real-time updates
    st.session_state.log_container = log_container
    
    old_stdout = sys.stdout
    sys.stdout = PrintCapture()
    
    try:
        if "Steuererklärung" in st.session_state.use_case:
            process_use_case_1(st.session_state.selected_file)
        else:
            process_use_case_2(st.session_state.selected_file)
        
        st.session_state.processed = True
        st.session_state.processing = False
        
    finally:
        sys.stdout = old_stdout
        st.session_state.log_container = None
    
    status_placeholder.empty()
    return True


def _is_use_case_1():
    """Check if current use case is Steuererklärung."""
    return "Steuererklärung" in st.session_state.use_case


def _render_uc1_with_params():
    """Render Use Case 1 with parameters panel."""
    col_params, col_results = st.columns([1, 3])
    
    with col_params:
        render_parameters_panel()
    
    with col_results:
        if st.session_state.uc1_result:
            render_uc1_results()
        else:
            st.info("ℹ️ Select a document and click 'Process' to start analysis")


def _render_uc1_full_width():
    """Render Use Case 1 full width (without parameters)."""
    if st.session_state.uc1_result:
        render_uc1_results()
    else:
        st.info("ℹ️ Select a document and click 'Process' to start analysis")


def _render_uc2_content():
    """Render Use Case 2 content."""
    if st.session_state.uc2_result:
        render_uc2_results()
    else:
        st.info("ℹ️ Select a document and click 'Process' to start analysis")


def _render_main_content():
    """Render the main content area based on use case and settings."""
    if _is_use_case_1() and st.session_state.show_params:
        _render_uc1_with_params()
    elif _is_use_case_1():
        _render_uc1_full_width()
    else:
        _render_uc2_content()


def _render_toggle_params_button():
    """Render the toggle parameters button for Use Case 1."""
    if not _is_use_case_1():
        return
    
    st.markdown("---")
    button_label = "⚙️ Hide Parameters Panel" if st.session_state.show_params else "⚙️ Toggle Parameters Panel"
    
    if st.button(button_label):
        st.session_state.show_params = not st.session_state.show_params
        st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    setup_logging()
    
    # Initialize log_container in session state if not present
    if "log_container" not in st.session_state:
        st.session_state.log_container = None
    
    render_header()
    
    # Show processing status at the TOP if processing
    status_placeholder = st.empty()
    if st.session_state.processing and not st.session_state.processed:
        status_placeholder.info("⏳ Processing document... (see logs at the bottom for progress)")
    
    # Render main content
    _render_main_content()
    _render_toggle_params_button()
    
    # Render logs section at the BOTTOM for real-time visibility during processing
    st.markdown("---")
    log_container = render_log_section(realtime=True)
    
    # Handle processing with the log container for real-time updates
    needs_rerun = _handle_processing(log_container, status_placeholder)
    
    if needs_rerun:
        st.rerun()


if __name__ == "__main__":
    main()