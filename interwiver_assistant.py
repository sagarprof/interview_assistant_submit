
import os
import time
import re
import json
import asyncio
import io
from pathlib import Path

# UI & Utils
import streamlit as st
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

# Google ADK
try:
    from google.adk.agents import Agent, SequentialAgent
    from google.adk.tools.tool_context import ToolContext
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import google_search
    from google.genai import types
except ImportError:
    pass 

# Load Environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================================
# 1. UI UTILS & STYLING
# ============================================================================

def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --primary: #6C63FF;
            --secondary: #FF6584;
            --bg-dark: #0e1117;
            --card-bg: #161b22;
            --border: #30363d; /* Explicit border color */
            --text-primary: #ffffff;
            --text-secondary: #8b949e;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
        }

        .stApp {
            background: radial-gradient(circle at top left, rgba(108, 99, 255, 0.1), transparent 40%),
                        radial-gradient(circle at bottom right, rgba(255, 101, 132, 0.05), transparent 40%);
        }

        /* Sidebar & Inputs */
        [data-testid="stSidebar"] { background-color: var(--card-bg); border-right: 1px solid var(--border); }
        
        /* BUTTON STYLING - FULL WIDTH */
        div.stButton > button {
            background: #a855f7; /* Updated to Violet */
            color: white;
            border: none;
            padding: 0.75rem 1.5rem; 
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
            width: 100%; 
            font-size: 1.1rem; 
        }
        div.stButton > button:hover {
            background: #9333ea; /* Darker Violet for hover */
            box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3); /* Violet glow */
            transform: translateY(-1px);
        }

        [data-testid="stFileUploader"] {
            border: 1px dashed var(--border);
            border-radius: 12px;
            padding: 1rem;
            background-color: rgba(22, 27, 34, 0.5);
        }

        /* METRIC CARDS - BORDER ADDED HERE */
        div[data-testid="column"] {
            background-color: var(--card-bg);
            border: 1px solid var(--border); /* Visible Border */
            border-radius: 12px;
            padding: 1.5rem 1rem; /* More padding for card look */
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            height: 100%;
        }

        /* Table Styling */
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
            border-radius: 8px;
            overflow: hidden;
        }
        .styled-table thead tr { background-color: var(--primary); color: #ffffff; text-align: left; }
        .styled-table th, .styled-table td { padding: 12px 15px; border-bottom: 1px solid var(--border); }
        .styled-table tbody tr { background-color: var(--card-bg); }
        
        .badge-yes {
            background-color: rgba(16, 185, 129, 0.2);
            color: #34d399;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.75rem;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-no {
            background-color: rgba(239, 68, 68, 0.2);
            color: #f87171;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.75rem;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .analysis-item-good {
            background: rgba(16, 185, 129, 0.1);
            border-left: 3px solid #10b981;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 4px;
        }
        .analysis-item-bad {
            background: rgba(239, 68, 68, 0.1);
            border-left: 3px solid #ef4444;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 4px;
        /* EQUAL WIDTH TABS */
        div[data-baseweb="tab-list"] {
            gap: 5px; /* Optional: Gap between tabs */
            width: 100%;
        }

        button[data-baseweb="tab"] {
            flex-grow: 1 !important; /* Forces all tabs to take equal space */
            width: auto; /* Resets rigid width */
            font-weight: 600;
        }
        
        /* Optional: Selected Tab Highlight in Violet */
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #a855f7 !important;
            border-bottom-color: #a855f7 !important;
        
        }
        }
        </style>
    """, unsafe_allow_html=True)

def render_metric_internal(label, value, suffix=""):
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 20px;
        background-color: #161b22; /* Card Background */
        border: 1px solid #a855f7; /* Violet Border */
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(168, 85, 247, 0.1); /* Subtle violet glow */
    ">
        <div style="font-size: 2.2rem; font-weight: 700; color: #fff;">{value}<span style="font-size: 1.2rem; color: #8b949e;">{suffix}</span></div>
        <div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def clean_and_parse_json(text):
    """Robust JSON extraction that handles messy LLM output and List vs Dict confusion."""
    if not text: return {}
    if not isinstance(text, str): return text
    
    # 1. Strip Markdown Code Blocks
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = text.strip()
    
    # 2. Parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: finding outermost brackets
        try:
            if '{' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                data = json.loads(text[start:end])
            elif '[' in text:
                start = text.find('[')
                end = text.rfind(']') + 1
                data = json.loads(text[start:end])
            else:
                return {}
        except:
            return {}

    return data

def sanitize_data_structure(data):
    """
    FIXES THE 'J, o, h, n' ISSUE.
    Ensures lists are actually lists, and numbers are numbers.
    """
    # 1. Sanitize Analysis / Match Score
    analysis = data.get('analysis', {})
    if isinstance(analysis, dict):
        match_score = analysis.get('match_score', {})
        
        # Ensure match_score is a dict
        if not isinstance(match_score, dict):
            # If it's a number (sometimes LLM returns just int), fix it
            if isinstance(match_score, (int, float)):
                analysis['match_score'] = {'overall_percentage': match_score, 'strengths': [], 'gaps': []}
            else:
                analysis['match_score'] = {'overall_percentage': 0, 'strengths': [], 'gaps': []}
        
        # Ensure Strengths/Gaps are LISTS, not STRINGS
        ms = analysis['match_score']
        if isinstance(ms.get('strengths'), str):
            ms['strengths'] = [ms['strengths']] # Wrap string in list
        if isinstance(ms.get('gaps'), str):
            ms['gaps'] = [ms['gaps']] # Wrap string in list
            
    # 2. Sanitize Skill Map
    skill_map = data.get('skill_map', [])
    if isinstance(skill_map, dict): # Sometimes LLM returns a dict wrapper
        skill_map = skill_map.get('skills', []) or []
    if not isinstance(skill_map, list):
        skill_map = []
    data['skill_map'] = skill_map
    
    return data


# =======================
# pydantic classes
# =======================
import json
from pydantic import BaseModel, Field
from typing import List

# --- Define the schemas to use for Prompt Injection ---
class Match(BaseModel):
    overall_percentage: int = Field(description="0 to 10 integer score")
    match: List[str] = Field(description="List of matching requirements")
    gaps: List[str] = Field(description="List of missing requirements")

class AnalysisOutput(BaseModel):
    candidate_summary: str
    match_score: Match

class SkillItem(BaseModel):
    skill: str = Field(description="Name of the skill")
    present: bool = Field(description="True if explicitly found, False otherwise")
    related_skills: str = Field(description="Evidence or context found, else '-'")

class SkillMappingOutput(BaseModel):
    skills: List[SkillItem]

class QuestionItem(BaseModel):
    question: str = Field(description="Interview question text")
    answer_hints: List[str] = Field(description="detailed points/keywords expected in answer")

class QuestionOutput(BaseModel):
    technical_questions: List[QuestionItem] = Field(description="List of ai technical interview questions")
   

class SynthesisOutput(BaseModel):
    summary:str = Field(description="Concise summary of candidate fit based on analysis")
    Tilted_experience:str = Field(description="Predicted previous experience tilt whether data scientist/devops/test engineer/data engineer from his projects and tools he used and justification in 2-3 lines")
    level_of_understanding:str = Field(description="Level of understanding of Gen AI concepts and practical experience as beginner, intermediate or expert")
    learning_interest:str = Field(description="Indicates candidate's passion and interest in learning based on certifications, courses completed, blogs written, papers, hackathons published")
    readiness_score: int = Field(description="0 to 10 integer score")
# ============================================================================
# 2. PDF GENERATION
# ============================================================================

from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

def generate_pdf(analysis, questions, summary, skill_map):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    
    story = []
    story.append(Paragraph("Candidate Interview Guide", styles['Title']))
    story.append(Spacer(1, 24))
    
    # 1. Executive Summary & Insights
    if summary:
        story.append(Paragraph("Executive Summary", styles['Heading1']))
        story.append(Paragraph(str(summary.get('summary', '')), styles['Justify']))
        story.append(Spacer(1, 12))
        
        # Insights
        data = [
            ["Metric", "Value"],
            ["Readiness Score", f"{summary.get('readiness_score', 0)}/10"],
            ["Level of Understanding", summary.get('level_of_understanding', 'N/A')],
            ["Tilted Experience", summary.get('Tilted_experience', 'N/A')],
            ["Learning Interest", summary.get('learning_interest', 'N/A')]
        ]
        t = Table(data, colWidths=[150, 300])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lavender),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 24))

    # 2. Match Analysis
    if analysis:
        match_score = analysis.get('match_score', {})
        story.append(Paragraph(f"Match Analysis (Overall Fit: {match_score.get('overall_percentage', 0)}/10)", styles['Heading1']))
        
        matches = match_score.get('match', [])
        if isinstance(matches, str): matches = [matches]
        if matches:
            story.append(Paragraph("Strengths / Matches:", styles['Heading2']))
            for m in matches:
                story.append(Paragraph(f"• {m}", styles['Normal']))
        
        gaps = match_score.get('gaps', [])
        if isinstance(gaps, str): gaps = [gaps]
        if gaps:
            story.append(Paragraph("Gaps / Missing Requirements:", styles['Heading2']))
            for g in gaps:
                story.append(Paragraph(f"• {g}", styles['Normal']))
        story.append(Spacer(1, 24))

    # 3. Skill Mapping
    if skill_map:
        story.append(Paragraph("Skill Mapping", styles['Heading1']))
        skills_list = skill_map.get('skills', []) if isinstance(skill_map, dict) else []
        
        if skills_list:
            table_data = [["Skill", "Status", "Evidence"]]
            for item in skills_list:
                status = "Verified" if item.get('present') else "Missing"
                evidence = item.get('related_skills', '-')
                # Truncate long evidence
                if len(evidence) > 50: evidence = evidence[:47] + "..."
                table_data.append([item.get('skill', ''), status, evidence])
            
            t_skills = Table(table_data, colWidths=[120, 80, 250])
            t_skills.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 9)
            ]))
            story.append(t_skills)
            story.append(Spacer(1, 24))

    # 4. Interview Questions
    if questions and isinstance(questions, dict):
        story.append(Paragraph("Interview Questions", styles['Heading1']))
        for category, q_list in questions.items():
            if isinstance(q_list, list) and q_list:
                story.append(Paragraph(category.replace('_', ' ').title(), styles['Heading2']))
                for idx, q in enumerate(q_list, 1):
                    if isinstance(q, dict):
                        story.append(Paragraph(f"Q{idx}: {q.get('question', '')}", styles['Normal']))
                        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

def setup_agents(api_key):
  
    def get_schema(model):
        return json.dumps(model.model_json_schema(), indent=2)

    # 1. Analyzer Agent
    agent_analyzer = Agent(
        name="Analyzer",
        model="gemini-2.5-flash",
        output_key="analysis_result",
        description="Analysis",
        instruction=f"""
        Analyze RESUME vs JD.Try to extract key information about the candidate from resume and how well that match with the job description.
        Dont be vague.Try to check for specific skills, experiences, and qualifications mentioned in both documents.
        Provide answer in match and gaps format.
        You act as a pure JSON API. Output strictly valid JSON. No markdown tags.
        Schema:
        {get_schema(AnalysisOutput)}
        """
    )

    # 2. Skill Mapper Agent - (UPDATED FOR BETTER MATCHING)
    agent_skill_mapper = Agent(
        name="SkillMapper",
        model="gemini-2.5-flash",
        output_key="skill_mapping_result",
        description="Mapping",
        instruction=f"""
        Analyze RESUME for specific AI Technical Skills.
        1. Search for: Prompt Engineering, RAG, Agents,RAG evaluations,Vector database,Keyword search,LLM finetuning,LLM inference,Small language model creation,Traditional Machine Learning(NLP,CV),API development,Databases(SQL,NoSQL),Cloud deployment,Frontend development.
        2. IF EXACT MATCH NOT FOUND: Look for synonyms (e.g., "Language Models" = "GenAI", "Postgres" = "SQL").
        3. If found, set 'present' to true and quote the text in 'related_skills'other wise false and '-'.
        4. Output strictly valid JSON. No markdown tags.
        Schema:
        {get_schema(SkillMappingOutput)}
        """
    )

    # 3. Question Gen Agent
    agent_questions = Agent(
        name="QuestionGen",
        model="gemini-2.5-flash",
        output_key="questions_result",
        description="Questions",
        tools=[google_search],
        instruction=f"""
        Generate 10 Technical questions related to Gen AI based on the candidate's resume and job description.
        Questions should be advanced and intermediate and relevant. 
        Expected answers should be in keywords but in details.
        Use google search to get the latest information.
        Output strictly valid JSON. No markdown tags.
        Schema:
        {get_schema(QuestionOutput)}
        """
    )

    # 4. Synthesis Agent
    agent_synthesis = Agent(
        name="Synthesis",
        model="gemini-2.5-flash",
        output_key="synthesis_result",
        description="Summary",
        instruction=f"""
        Provide final candidate suitability summary.In summary give your opinion whether the candidate is a good fit for the job role based on the resume and job description provided and also add your justification.
        Summary should be precise and concise.
        Predict candadate's level of understanding of Gen AI concepts and practical experience as beginner, intermediate or expert.
        Also predict his previous experience tilt whether data scientist/devops/test engineer/data engineer from his projects and tools he used justification for prediction should be in 2-3 lines.
        Tell me whether the candidate has passion and interest in learning based on candidate's certification,course completed,blogs written,papers,hackathons published.
        Give a readiness score from 0 to 10 integer score.
        Output strictly valid JSON. No markdown tags.

        Schema:
        {get_schema(SynthesisOutput)}
        """
    )

    return SequentialAgent(name="Pipeline", sub_agents=[agent_analyzer, agent_skill_mapper, agent_questions, agent_synthesis], description="Sequential Flow")
# ============================================================================
# 4. MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="Interview Assistant", page_icon="✨", layout="wide")
    load_css()
    api_key = os.getenv('GOOGLE_API_KEY')

    st.markdown("<h1 style='text-align: center;'>Gen AI Interview Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e;'>Recruitment & Analysis Agent Swarm</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### :violet[:material/person:] Candidate Info") 
        resume_file = st.file_uploader("Upload", type=['txt', 'pdf'], key="resume", label_visibility="collapsed")
    with col2:
        st.markdown("#### :violet[:material/event:] Interview Details")
        jd_file = st.file_uploader("Upload", type=['txt', 'pdf'], key="jd", label_visibility="collapsed")

    st.markdown("###")
    
    start_btn = st.button(":material/rocket_launch: Analyze & Generate Guide", type="primary", use_container_width=True)

    if start_btn:
        # if not resume_file or not jd_file or not api_key:
        #     st.error("Missing files or API Key.")
        #     return

        def read_file(f):
            try:
                if f.type == "application/pdf":
                    reader = PyPDF2.PdfReader(io.BytesIO(f.read()))
                    return "".join(p.extract_text() for p in reader.pages)
                return f.read().decode("utf-8")
            except: return ""

        resume_text = read_file(resume_file)
        jd_text = read_file(jd_file)

        status = st.empty()
        prog = st.progress(0)
        
        async def run_pipeline():
            root_agent = setup_agents(api_key)
            session_service = InMemorySessionService()
            await session_service.create_session(app_name="ia", user_id="u1", session_id="s1")
            runner = Runner(agent=root_agent, app_name="ia", session_service=session_service)
            content = types.Content(role='user', parts=[types.Part(text=f"RESUME:{resume_text[:3000]}\nJD:{jd_text[:2000]}")])
            
            results = {}
            step = 0
            
            async for event in runner.run_async(user_id="u1", session_id="s1", new_message=content):
                sess = await session_service.get_session(app_name="ia", user_id="u1", session_id="s1")
                if sess:
                    if sess.state.get("analysis_result") and step == 0:
                        
                        status.markdown(":material/psychology: Analyzing...")
                        prog.progress(30)
                        step = 1
                    elif sess.state.get("skill_mapping_result") and step == 1:
                        status.markdown(":material/search: Mapping Skills...")
                        prog.progress(60)
                        step = 2
                
                
            
                if event.is_final_response():
                    final = await session_service.get_session(app_name="ia", user_id="u1", session_id="s1")
                    if final:
                        # Parse
                        results['analysis'] = clean_and_parse_json(final.state.get('analysis_result'))
                        results['skill_map'] = clean_and_parse_json(final.state.get('skill_mapping_result'))
                        results['questions'] = clean_and_parse_json(final.state.get('questions_result'))
                        results['synthesis'] = clean_and_parse_json(final.state.get('synthesis_result'))
            
            prog.progress(100)
            status.empty()
            return results

        try:
            with st.spinner("Agents working...",show_time=True):
                data = asyncio.run(run_pipeline())
                
            # ==========================================
            # CRITICAL FIX: DATA NORMALIZATION
            # ==========================================
            
            # 1. Normalize Skills: Ensure it's a Dictionary with "skills" key
            raw_skills = data.get('skill_map', {})
            if isinstance(raw_skills, list):
                # LLM returned just a list -> Wrap it
                data['skill_map'] = {"skills": raw_skills}
            elif isinstance(raw_skills, dict):
                if 'skills' not in raw_skills:
                    # LLM returned a dict but with wrong key (or empty) -> Fix
                    data['skill_map'] = {"skills": []}
            else:
                data['skill_map'] = {"skills": []}

            # 2. Normalize Questions: Ensure Technical/Behavioral keys exist
            q_data = data.get('questions', {})
            if isinstance(q_data, list):
                # LLM returned a list of questions? Try to salvage
                data['questions'] = {"technical_questions": q_data,}
            
            # 3. Sanitize the rest
            data = sanitize_data_structure(data)

            
            # --- METRICS ---
            synthesis = data.get('synthesis', {})
            match_data = data.get('analysis', {}).get('match_score', {})
            skill_map_data = data.get('skill_map', [])

            # Extract integer for fit (Fixes the "Dictionary in UI" bug)
            fit_val = match_data.get('overall_percentage', 0)
            if not isinstance(fit_val, (int, float)):
                fit_val = 0 # Fallback if data is messy

            # Extract "Yes" count (Fixes the "0" bug by checking strings)
            yes_count = 0
            # Handle the structure: { "skills": [ ... ] }
            raw_skills = skill_map_data.get('skills', []) if isinstance(skill_map_data, dict) else []
            
            for item in raw_skills:
                if item.get('present') is True: # Direct boolean check
                    yes_count += 1

            m1, m2, = st.columns(2, gap="large")
            with m1: render_metric_internal("Skill Fit", synthesis.get('readiness_score', 0), "/10")
            with m2: render_metric_internal("JD Fit", fit_val, "/10")
            # with m3: render_metric_internal("Skills Verified", yes_count)

            st.markdown("###")
            tab1, tab2, tab3, tab4,tab5,tab6= st.tabs([":material/description: Questions",
                                                        ":material/bar_chart: Gen AI Skills Mapping", 
                                                        ":material/psychology: Requirement Match", 
                                                        ":material/article: Suitability Prediction",
                                                        ":material/article: Tilted Experience Analysis",
                                                        ":material/star: Learning Interest Analysis",])
            with tab1:
                q_data = data.get('questions', {})
                if isinstance(q_data, dict):
                    for cat, q_list in q_data.items():
                        if isinstance(q_list, list) and q_list:
                            st.markdown(f"#### {cat.replace('_', ' ').title()}")
                            for q in q_list:
                                if isinstance(q, dict):
                                    with st.expander(f":material/help: {q.get('question', '')}"):
                                        hints = q.get('answer_hints', [])
                                        h_str = ", ".join(hints) if isinstance(hints, list) else str(hints)
                                        st.write(f"Look for: {h_str}")

            with tab2:
                # Get the list from the dictionary wrapper
                skills_list = skill_map_data.get('skills', []) if isinstance(skill_map_data, dict) else []
                # st.write(skill_map_data)
                import pandas as pd
                df = pd.DataFrame(skill_map_data)
                st.write(df)

            with tab3:
                # Fixes the "Single Letter" Bug by ensuring lists
                match = match_data.get('match', [])
                gaps = match_data.get('gaps', [])
                
                # Double check (Sanitization should have handled it, but safety first)
                if isinstance(match, str): match = [match]
                if isinstance(gaps, str): gaps = [gaps]

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### :material/check_circle: Matches")
                    st.write(' ')
                    for s in match:
                        st.markdown(f"<div class='analysis-item-good'>{s}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("#### :material/warning: Gaps")
                    st.write(' ')
                    for g in gaps:
                        st.markdown(f"<div class='analysis-item-bad'>{g}</div>", unsafe_allow_html=True)

            with tab4:
                st.markdown(synthesis.get('summary', ''))
            
                level_of_understanding = synthesis.get('level_of_understanding', 'N/A')
                
                st.markdown(f"**Level of Understanding:** {level_of_understanding}")

            with tab5:

                st.markdown("#### Tilted Experience Analysis")
                st.markdown(f"**Tilted Experience:** {synthesis.get('Tilted_experience', 'N/A')}")

            with tab6:
                st.markdown("#### Learning Interest Analysis")
                st.markdown(f"**Learning Interest:** {synthesis.get('learning_interest', 'N/A')}")
               
            st.markdown("---")
            pdf_data = generate_pdf(data.get('analysis'), data.get('questions'), data.get('synthesis'), data.get('skill_map'))
            st.download_button(":material/download: Download Report", pdf_data, "report.pdf", "application/pdf")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
