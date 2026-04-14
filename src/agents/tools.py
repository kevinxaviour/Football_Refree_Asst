"""
src/agents/tools.py
--------------------
Tools available to the FIFA Referee AI Agent.
This implements the "tool use" pattern for agentic AI.

Tools:
1. search_laws       — Semantic search over all 17 Laws
2. get_law_summary   — Get structured info about a specific Law
3. clarify_question  — Generate a clarification request
4. check_penalty     — Specialized tool for penalty decisions
"""

from typing import Any, Dict, List
from src.vectorstore.chroma_store import search


# Summary of all 17 Laws (static reference)
LAWS_REFERENCE = {
    1: {
        "name": "The Field of Play",
        "key_topics": ["pitch dimensions", "markings", "goal posts", "corner arcs", "penalty area"],
        "summary": "Defines the playing surface dimensions, markings, goals, and technical areas."
    },
    2: {
        "name": "The Ball",
        "key_topics": ["ball size", "pressure", "replacement ball", "defective ball"],
        "summary": "Specifies ball specifications and procedures for defective or lost balls."
    },
    3: {
        "name": "The Players",
        "key_topics": ["team size", "substitutions", "extra players", "outside agents"],
        "summary": "Covers player numbers, substitution procedures, and team officials."
    },
    4: {
        "name": "The Players' Equipment",
        "key_topics": ["kit", "shin guards", "footwear", "jewellery", "armbands"],
        "summary": "Defines mandatory and permitted equipment for all players."
    },
    5: {
        "name": "The Referee",
        "key_topics": ["authority", "powers", "decisions", "advantage", "injury time"],
        "summary": "Defines referee's authority, decision-making powers, and responsibilities."
    },
    6: {
        "name": "The Other Match Officials",
        "key_topics": ["assistant referee", "fourth official", "VAR", "offside flag"],
        "summary": "Roles of assistant referees, fourth official, and video assistant referee (VAR)."
    },
    7: {
        "name": "The Duration of the Match",
        "key_topics": ["45 minutes", "half time", "injury time", "extra time", "stoppage time"],
        "summary": "Match duration, half-time intervals, and time-keeping procedures."
    },
    8: {
        "name": "The Start and Restart of Play",
        "key_topics": ["kick-off", "dropped ball", "coin toss", "restart procedures"],
        "summary": "Procedures for kick-off and all restarts of play."
    },
    9: {
        "name": "The Ball In and Out of Play",
        "key_topics": ["ball out of play", "ball in play", "goal posts", "crossbar"],
        "summary": "Defines when the ball is in and out of play."
    },
    10: {
        "name": "Determining the Outcome of a Match",
        "key_topics": ["goal scored", "penalty shootout", "kicks from mark"],
        "summary": "How goals are awarded and how match outcomes are determined."
    },
    11: {
        "name": "Offside",
        "key_topics": ["offside position", "interfering with play", "interfering with opponent", "gaining advantage"],
        "summary": "Defines the offside rule and when being offside is penalised."
    },
    12: {
        "name": "Fouls and Misconduct",
        "key_topics": ["direct free kick", "indirect free kick", "yellow card", "red card", "handball", "DOGSO", "SPA"],
        "summary": "All offences, disciplinary measures, and the full handling/handball rules."
    },
    13: {
        "name": "Free Kicks",
        "key_topics": ["direct free kick", "indirect free kick", "wall distance", "10 yards"],
        "summary": "Procedures for direct and indirect free kicks."
    },
    14: {
        "name": "The Penalty Kick",
        "key_topics": ["penalty spot", "goalkeeper position", "run-up", "penalty area"],
        "summary": "Complete procedures for penalty kicks during and after the match."
    },
    15: {
        "name": "The Throw-In",
        "key_topics": ["throw-in procedure", "both feet", "behind head", "offside from throw-in"],
        "summary": "Correct throw-in technique and associated rules."
    },
    16: {
        "name": "The Goal Kick",
        "key_topics": ["goal kick", "penalty area", "goal area", "opponents outside area"],
        "summary": "Procedures for goal kicks."
    },
    17: {
        "name": "The Corner Kick",
        "key_topics": ["corner arc", "corner flag", "opponents distance", "corner kick procedure"],
        "summary": "Procedures for corner kicks."
    },
}


def search_laws(query: str, top_k: int = 5, law_filter: str = None) -> Dict[str, Any]:
    """
    Tool: Search the Laws of the Game database.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
        law_filter: Optional specific law to search within
        
    Returns:
        Dict with results and metadata
    """
    try:
        results = search(query=query, top_k=top_k, law_filter=law_filter)
        return {
            "tool": "search_laws",
            "query": query,
            "results": results,
            "result_count": len(results),
            "status": "success"
        }
    except Exception as e:
        return {
            "tool": "search_laws",
            "query": query,
            "results": [],
            "result_count": 0,
            "status": "error",
            "error": str(e)
        }


def get_law_summary(law_number: int) -> Dict[str, Any]:
    """
    Tool: Get a structured summary of a specific Law (1-17).
    
    Args:
        law_number: The Law number (1-17)
        
    Returns:
        Dict with law details
    """
    if law_number not in LAWS_REFERENCE:
        return {
            "tool": "get_law_summary",
            "status": "error",
            "error": f"Law {law_number} doesn't exist. Laws are numbered 1-17."
        }

    law = LAWS_REFERENCE[law_number]
    # Also do a quick search for the law's content
    results = search(
        query=f"Law {law_number} {law['name']}",
        top_k=3,
        law_filter=f"Law {law_number} - {law['name']}"
    )

    return {
        "tool": "get_law_summary",
        "law_number": law_number,
        "law_name": law["name"],
        "full_title": f"Law {law_number} - {law['name']}",
        "summary": law["summary"],
        "key_topics": law["key_topics"],
        "relevant_text": [r["text"] for r in results[:2]],
        "status": "success"
    }


def clarify_question(question: str, missing_info: str) -> Dict[str, Any]:
    """
    Tool: Signal that a question needs clarification.
    
    Args:
        question: The original question
        missing_info: What information is needed to answer
        
    Returns:
        Dict with clarification request
    """
    return {
        "tool": "clarify_question",
        "original_question": question,
        "clarification_needed": missing_info,
        "status": "needs_clarification"
    }


def identify_applicable_laws(question: str) -> List[int]:
    """
    Heuristic: Identify likely applicable Laws based on keywords.
    Used by the agent to prioritize search.
    """
    question_lower = question.lower()
    applicable = []

    keyword_map = {
        11: ["offside", "off-side", "level", "nearer to goal"],
        12: ["foul", "handball", "hand ball", "card", "red", "yellow", "tackle", "push",
             "kick", "strike", "spit", "bite", "misconduct", "DOGSO", "dangerous"],
        14: ["penalty", "penalty kick", "spot kick", "pk"],
        5: ["referee", "advantage", "whistle", "decision"],
        3: ["substitut", "extra player", "team size"],
        7: ["time", "stoppage", "injury time", "extra time", "half time"],
        10: ["goal", "scored", "disallow", "valid goal"],
        15: ["throw-in", "throw in", "touchline"],
        16: ["goal kick", "goalie kick"],
        17: ["corner", "corner kick"],
        4: ["equipment", "kit", "boots", "shin guard", "jewel"],
        1: ["pitch", "field", "goal post", "crossbar", "dimensions"],
        9: ["out of play", "in play", "boundary"],
        13: ["free kick", "wall", "10 yards", "9.15"],
        2: ["ball", "pressure", "defective"],
    }

    for law_num, keywords in keyword_map.items():
        if any(kw in question_lower for kw in keywords):
            applicable.append(law_num)

    return applicable if applicable else list(range(1, 18))  # All laws if no match


# Tool registry — maps tool names to functions
TOOL_REGISTRY = {
    "search_laws": search_laws,
    "get_law_summary": get_law_summary,
    "clarify_question": clarify_question,
    "identify_applicable_laws": identify_applicable_laws,
}
