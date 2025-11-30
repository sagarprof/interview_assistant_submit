import streamlit as st
import base64
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
