"""
Multi-Agent Travel Planner System
Assignment: Build a Multi-Agent System using LangChain + LangGraph

This system uses 4 specialized agents to create comprehensive travel plans:
1. Planner Agent - Extracts structured information from user requests
2. Research Agent - Gathers destination information and attractions
3. Itinerary Builder Agent - Creates day-by-day travel itineraries
4. Budget Estimator Agent - Provides cost breakdowns and budget analysis

The agents collaborate through a shared state (TravelState) and are orchestrated
using LangGraph's directed workflow.

Usage:
    CLI Mode: python multi_agent_system_streamlit.py
    Streamlit Mode: streamlit run multi_agent_system_streamlit.py
"""

import os
import sys
import json
from typing import TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# SHARED STATE DEFINITION
# ============================================================================

class TravelState(TypedDict):
    """
    Shared state passed between all agents in the workflow.
    Each agent reads from and writes to specific fields.
    """
    user_input: str          # Raw free-text from the user
    destination: str         # Extracted by Planner; "unknown" if not found
    travel_dates: str        # Extracted by Planner; empty string if not found
    budget: str              # Extracted by Planner; empty string if not found
    preferences: str         # Extracted by Planner; empty string if not found
    research_notes: str      # Populated by Research Agent
    itinerary: str           # Populated by Itinerary Builder Agent
    budget_estimate: str     # Populated by Budget Estimator Agent


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Get API key from environment or Streamlit secrets
def get_api_key():
    """Get Groq API key from environment variables or Streamlit secrets"""
    # Try to get from environment first (local development)
    api_key = os.getenv("GROQ_API_KEY")
    
    # If not found, try Streamlit secrets (cloud deployment)
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except:
            pass
    
    return api_key

# Shared LLM instance — API key is read from GROQ_API_KEY env var or Streamlit secrets
# Using Groq's free tier with llama-3.3-70b-versatile model
api_key = get_api_key()
if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it in:\n"
        "- Local: .env file with GROQ_API_KEY=your_key\n"
        "- Streamlit Cloud: App Settings → Secrets → Add GROQ_API_KEY = \"your_key\""
    )

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, api_key=api_key)


# ============================================================================
# AGENT 1: PLANNER AGENT
# ============================================================================

def planner_node(state: TravelState) -> dict:
    """
    Planner Agent: Extracts structured travel information from user input.
    
    Role: Parse free-text travel requests and extract:
        - Destination
        - Travel dates
        - Budget
        - Preferences (activities, interests, etc.)
    
    Input: user_input from TravelState
    Output: Updates destination, travel_dates, budget, preferences in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a travel planning assistant. Extract structured travel intent from the user's request.\n"
         "Return ONLY a valid JSON object with these exact keys: destination, travel_dates, budget, preferences.\n"
         "Do not include any other text, explanations, or markdown formatting.\n"
         "If you cannot determine a value, use an empty string. If destination is unclear, use \"unknown\".\n\n"
         "Example output:\n"
         '{{"destination": "Paris", "travel_dates": "June 15-22", "budget": "$2000", "preferences": "art museums and cafes"}}'),
        ("human", "Travel request: {user_input}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"user_input": state["user_input"]})
    
    try:
        # Clean up response - remove markdown code blocks if present
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        parsed = json.loads(content)
        return {
            "destination": parsed.get("destination", "unknown"),
            "travel_dates": parsed.get("travel_dates", ""),
            "budget": parsed.get("budget", ""),
            "preferences": parsed.get("preferences", ""),
        }
    except Exception as e:
        print(f"Warning: Failed to parse planner response: {e}")
        print(f"Raw response: {response.content}")
        return {"destination": "unknown", "travel_dates": "", "budget": "", "preferences": ""}


# ============================================================================
# AGENT 2: RESEARCH AGENT
# ============================================================================

def research_node(state: TravelState) -> dict:
    """
    Research Agent: Gathers destination information and travel tips.
    
    Role: Research the destination and provide:
        - Top attractions and highlights
        - Practical travel tips
        - Local insights based on user preferences
    
    Input: destination, preferences from TravelState
    Output: Updates research_notes in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a destination research specialist. Provide at least 3 highlights, top attractions,\n"
         "and practical travel tips for the given destination and traveller preferences.\n"
         "If destination is \"unknown\", provide general travel tips instead."),
        ("human", "Destination: {destination}\nPreferences: {preferences}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "preferences": state["preferences"],
    })
    return {"research_notes": response.content}


# ============================================================================
# AGENT 3: ITINERARY BUILDER AGENT
# ============================================================================

def itinerary_node(state: TravelState) -> dict:
    """
    Itinerary Builder Agent: Creates detailed day-by-day travel plans.
    
    Role: Build a structured itinerary that:
        - Organizes activities by day
        - Incorporates research findings
        - Aligns with user preferences and travel dates
    
    Input: destination, travel_dates, preferences, research_notes from TravelState
    Output: Updates itinerary in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert travel itinerary builder. Create a detailed day-by-day itinerary.\n"
         "Each day must include at least one activity drawn directly from the research notes provided.\n"
         "Format clearly with Day 1, Day 2, etc."),
        ("human",
         "Destination: {destination}\n"
         "Dates: {travel_dates}\n"
         "Preferences: {preferences}\n"
         "Research Notes: {research_notes}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "travel_dates": state["travel_dates"],
        "preferences": state["preferences"],
        "research_notes": state["research_notes"],
    })
    return {"itinerary": response.content}


# ============================================================================
# AGENT 4: BUDGET ESTIMATOR AGENT
# ============================================================================

def budget_node(state: TravelState) -> dict:
    """
    Budget Estimator Agent: Provides cost analysis and budget breakdown.
    
    Role: Analyze the itinerary and provide:
        - Cost breakdown (accommodation, transport, food, activities)
        - Budget feasibility assessment
        - Recommendations for staying within budget
    
    Input: destination, travel_dates, budget, itinerary from TravelState
    Output: Updates budget_estimate in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a travel budget analyst. Provide a cost breakdown covering accommodation,\n"
         "transport, food, and activities. If a numeric budget is provided, explicitly state\n"
         "whether the estimated total is within, at, or over that budget."),
        ("human",
         "Destination: {destination}\n"
         "Dates: {travel_dates}\n"
         "Budget: {budget}\n"
         "Itinerary: {itinerary}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "travel_dates": state["travel_dates"],
        "budget": state["budget"],
        "itinerary": state["itinerary"],
    })
    return {"budget_estimate": response.content}


# ============================================================================
# LANGGRAPH WORKFLOW DEFINITION
# ============================================================================

def build_graph():
    """
    Build the LangGraph workflow connecting all agents.
    
    Workflow:
        START → Planner → Research → Itinerary Builder → Budget Estimator → END
    
    Each agent processes the shared TravelState and passes it to the next agent.
    """
    graph = StateGraph(TravelState)
    
    # Add nodes (agents)
    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("itinerary", itinerary_node)
    graph.add_node("budget", budget_node)
    
    # Define workflow edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "itinerary")
    graph.add_edge("itinerary", "budget")
    graph.add_edge("budget", END)
    
    return graph.compile()


# ============================================================================
# EXECUTION FUNCTION
# ============================================================================

def execute_travel_planning(user_input: str) -> TravelState:
    """
    Execute the multi-agent travel planning workflow.
    
    Args:
        user_input: User's travel request in natural language
    
    Returns:
        TravelState: Final state with all fields populated by agents
    """
    initial_state: TravelState = {
        "user_input": user_input,
        "destination": "",
        "travel_dates": "",
        "budget": "",
        "preferences": "",
        "research_notes": "",
        "itinerary": "",
        "budget_estimate": "",
    }
    
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    return final_state


# ============================================================================
# CLI MODE
# ============================================================================

def print_travel_plan(state: TravelState) -> None:
    """Print travel plan to console (CLI mode)"""
    print("\n" + "="*80)
    print("TRAVEL PLAN GENERATED")
    print("="*80)
    
    print("\n📍 DESTINATION")
    print("-" * 80)
    print(state["destination"] if state["destination"] else "Not determined")
    
    print("\n📅 TRAVEL DATES")
    print("-" * 80)
    print(state["travel_dates"] if state["travel_dates"] else "Not specified")
    
    print("\n💰 BUDGET")
    print("-" * 80)
    print(state["budget"] if state["budget"] else "Not specified")
    
    print("\n🎯 PREFERENCES")
    print("-" * 80)
    print(state["preferences"] if state["preferences"] else "Not specified")
    
    print("\n📝 ITINERARY")
    print("-" * 80)
    print(state["itinerary"] if state["itinerary"] else "No itinerary generated")
    
    print("\n💵 BUDGET ESTIMATE")
    print("-" * 80)
    print(state["budget_estimate"] if state["budget_estimate"] else "No estimate generated")
    
    print("\n" + "="*80)


def main() -> None:
    """Main function for CLI mode"""
    print("="*80)
    print("MULTI-AGENT TRAVEL PLANNER")
    print("="*80)
    print("\nThis system uses 4 AI agents to create your perfect travel plan:")
    print("  1. Planner Agent - Extracts travel details")
    print("  2. Research Agent - Finds attractions and tips")
    print("  3. Itinerary Builder - Creates day-by-day plans")
    print("  4. Budget Estimator - Analyzes costs")
    print("\n" + "="*80 + "\n")
    
    # Get user input
    user_input = ""
    while not user_input.strip():
        user_input = input("Enter your travel request: ")
    
    print("\n🔄 Processing your request through 4 specialized agents...")
    print("   This may take 15-30 seconds...\n")
    
    # Execute workflow
    final_state = execute_travel_planning(user_input)
    
    # Display results
    print_travel_plan(final_state)


# ============================================================================
# STREAMLIT MODE
# ============================================================================

def run_streamlit_app():
    """Run Streamlit web interface"""
    import streamlit as st

    st.set_page_config(
        page_title="Travel Planner",
        page_icon="✈️",
        layout="centered"
    )

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        #MainMenu, footer, header { visibility: hidden; }
        .stApp { background-color: #0d1117; color: #c9d1d9; }

        .hero { text-align: center; padding: 2.5rem 1rem 1rem; }
        .hero-badge {
            display: inline-block;
            background: rgba(88,166,255,0.1);
            color: #58a6ff;
            font-size: 0.72rem; font-weight: 600;
            letter-spacing: 0.12em; text-transform: uppercase;
            padding: 0.3rem 0.9rem; border-radius: 999px;
            border: 1px solid rgba(88,166,255,0.25);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.4rem; font-weight: 700;
            color: #f0f6fc; letter-spacing: -0.02em; margin: 0 0 0.5rem;
        }
        .hero-sub { font-size: 0.95rem; color: #6e7681; margin: 0; }

        .stTextArea textarea {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 10px !important;
            color: #c9d1d9 !important;
            font-family: "Inter", sans-serif !important;
            font-size: 0.93rem !important;
            padding: 0.9rem !important;
        }
        .stTextArea textarea:focus {
            border-color: #58a6ff !important;
            box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
        }
        .stTextArea textarea::placeholder { color: #484f58 !important; }
        .stTextArea label { color: #6e7681 !important; font-size: 0.8rem !important; }

        .stButton > button[kind="primary"] {
            background: #238636 !important;
            border: 1px solid #2ea043 !important;
            border-radius: 8px !important;
            color: #fff !important;
            font-family: "Inter", sans-serif !important;
            font-size: 0.92rem !important; font-weight: 600 !important;
            padding: 0.6rem 1.5rem !important;
            transition: background 0.2s !important;
        }
        .stButton > button[kind="primary"]:hover { background: #2ea043 !important; }

        .rcard {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }
        .rcard-label {
            font-size: 0.65rem; font-weight: 700;
            letter-spacing: 0.12em; text-transform: uppercase;
            color: #58a6ff; margin-bottom: 0.8rem;
        }
        .pill-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
        .pill {
            background: #0d1117; border: 1px solid #21262d;
            border-radius: 8px; padding: 0.7rem 0.9rem;
        }
        .pill-lbl {
            font-size: 0.62rem; font-weight: 600;
            letter-spacing: 0.1em; text-transform: uppercase;
            color: #484f58; margin-bottom: 0.25rem;
        }
        .pill-val { color: #c9d1d9; font-size: 0.88rem; font-weight: 500; }

        .stDownloadButton > button {
            background: transparent !important;
            border: 1px solid #30363d !important;
            border-radius: 7px !important;
            color: #8b949e !important;
            font-size: 0.83rem !important;
        }
        .stDownloadButton > button:hover {
            border-color: #58a6ff !important; color: #58a6ff !important;
        }
        .streamlit-expanderHeader { color: #6e7681 !important; font-size: 0.82rem !important; }
        </style>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
        <div class="hero">
            <div class="hero-badge">✨ 4 Specialized AI Agents</div>
            <h1 class="hero-title">Travel Planner</h1>
            <p class="hero-sub">Tell us where you want to go. We'll build the plan.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    user_input = st.text_area(
        "travel_request",
        height=110,
        placeholder="e.g. 7 days in Tokyo this March, $3000 budget. Love food and temples.",
        label_visibility="collapsed"
    )

    with st.expander("See example prompts"):
        st.markdown("""
- *"7 days in Tokyo, March, $3000 — food & temples."*
- *"Romantic Paris trip, 5 days June, $2500 — art & cafes."*
- *"Family Bali, 10 days December, $5000 — beaches & wildlife."*
- *"Solo Thailand backpack, 2 weeks, $1500 — culture & nightlife."*
        """)

    go = st.button("Generate Plan →", use_container_width=True, type="primary")

    st.markdown("<br>", unsafe_allow_html=True)

    if go:
        if not user_input.strip():
            st.error("Please enter a travel request first.")
        else:
            with st.spinner("Running 4 agents — this takes 15–30 seconds…"):
                try:
                    final_state = execute_travel_planning(user_input)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            dest  = final_state["destination"]  or "—"
            dates = final_state["travel_dates"] or "—"
            budg  = final_state["budget"]        or "—"
            prefs = final_state["preferences"]   or "—"

            # Overview
            st.markdown(f"""
                <div class="rcard">
                    <div class="rcard-label">Trip Overview</div>
                    <div class="pill-grid">
                        <div class="pill"><div class="pill-lbl">Destination</div><div class="pill-val">{dest}</div></div>
                        <div class="pill"><div class="pill-lbl">Dates</div><div class="pill-val">{dates}</div></div>
                        <div class="pill"><div class="pill-lbl">Budget</div><div class="pill-val">{budg}</div></div>
                        <div class="pill"><div class="pill-lbl">Preferences</div><div class="pill-val">{prefs}</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Itinerary
            st.markdown('<div class="rcard"><div class="rcard-label">Itinerary</div>', unsafe_allow_html=True)
            if final_state["itinerary"]:
                st.markdown(final_state["itinerary"])
            else:
                st.caption("No itinerary generated.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Budget
            st.markdown('<div class="rcard"><div class="rcard-label">Budget Breakdown</div>', unsafe_allow_html=True)
            if final_state["budget_estimate"]:
                st.markdown(final_state["budget_estimate"])
            else:
                st.caption("No estimate generated.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Research
            st.markdown('<div class="rcard"><div class="rcard-label">Research Notes</div>', unsafe_allow_html=True)
            if final_state["research_notes"]:
                st.markdown(final_state["research_notes"])
            else:
                st.caption("No research notes.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Download
            plan_txt = f"""TRAVEL PLAN
===========
Destination:  {final_state["destination"]}
Dates:        {final_state["travel_dates"]}
Budget:       {final_state["budget"]}
Preferences:  {final_state["preferences"]}

ITINERARY
=========
{final_state["itinerary"]}

BUDGET ESTIMATE
===============
{final_state["budget_estimate"]}

RESEARCH NOTES
==============
{final_state["research_notes"]}
"""
            st.download_button(
                "↓ Download plan (.txt)",
                data=plan_txt,
                file_name="travel_plan.txt",
                mime="text/plain"
            )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If streamlit is imported and we're in streamlit context
        if hasattr(st, 'runtime') and st.runtime.exists():
            run_streamlit_app()
        else:
            # Streamlit installed but not running in streamlit context
            main()
    except ImportError:
        # Streamlit not installed, run CLI mode
        main()
