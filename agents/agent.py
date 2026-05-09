# agents/agent.py
# AI Assistant for delivery intelligence — powered by Groq LLM

import os
import json
import streamlit as st
from groq import Groq

from .tools import (
    traffic_analysis,
    weather_analysis,
    vehicle_analysis,
    city_analysis,
    peak_hour_analysis,
    rating_analysis,
    distance_analysis,
    festival_analysis,
    general_stats,
    comprehensive_summary,
)

# ── System Prompt ────────────────────────────────────────
SYSTEM_PROMPT = """You are DeliverIQ, an AI-powered delivery intelligence analyst.

You analyze food delivery operations data and provide actionable business insights.

## Your capabilities:
- Analyze delivery time patterns across traffic, weather, vehicles, cities, and peak hours
- Identify operational bottlenecks and inefficiencies
- Recommend data-driven improvements for delivery performance
- Explain statistical findings in business-friendly language

## Response guidelines:
1. Start with a direct answer to the question
2. Support your answer with specific numbers from the data provided
3. Provide 2-3 actionable recommendations where relevant
4. Use bullet points for readability
5. Keep responses concise but insightful (150-300 words)
6. Use emojis sparingly for visual structure (📊, 🚦, ⏱️, etc.)

## Important:
- Only use the data provided in the context — never fabricate numbers
- If the data doesn't contain enough information to answer, say so clearly
- Compare metrics to provide relative context (e.g., "20% slower than average")
"""

# ── Model Config ─────────────────────────────────────────
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _get_groq_client():
    """Get Groq client with API key from Streamlit secrets or environment."""
    api_key = None

    # Try Streamlit secrets first (direct access, not .get())
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError, Exception):
        pass

    # Fallback to environment variable
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def _route_question(question):
    """Route question to the appropriate analysis tool(s)."""
    q = question.lower()

    # Map keywords to analysis functions
    routes = {
        'traffic': 'traffic',
        'congestion': 'traffic',
        'road': 'traffic',
        'weather': 'weather',
        'rain': 'weather',
        'fog': 'weather',
        'storm': 'weather',
        'sunny': 'weather',
        'vehicle': 'vehicle',
        'motorcycle': 'vehicle',
        'scooter': 'vehicle',
        'car': 'vehicle',
        'city': 'city',
        'urban': 'city',
        'metropolitan': 'city',
        'peak': 'peak',
        'rush hour': 'peak',
        'busy': 'peak',
        'rating': 'rating',
        'rated': 'rating',
        'driver': 'rating',
        'distance': 'distance',
        'far': 'distance',
        'near': 'distance',
        'km': 'distance',
        'festival': 'festival',
        'holiday': 'festival',
    }

    matched = set()
    for keyword, route in routes.items():
        if keyword in q:
            matched.add(route)

    return matched if matched else {'general'}


def _get_context(question, df):
    """Build analysis context based on the question."""
    routes = _route_question(question)
    context = {}

    route_map = {
        'traffic': ('traffic_analysis', traffic_analysis),
        'weather': ('weather_analysis', weather_analysis),
        'vehicle': ('vehicle_analysis', vehicle_analysis),
        'city': ('city_analysis', city_analysis),
        'peak': ('peak_hour_analysis', peak_hour_analysis),
        'rating': ('rating_analysis', rating_analysis),
        'distance': ('distance_analysis', distance_analysis),
        'festival': ('festival_analysis', festival_analysis),
        'general': ('comprehensive_summary', comprehensive_summary),
    }

    for route in routes:
        if route in route_map:
            name, func = route_map[route]
            context[name] = func(df)

    # Always include general stats for context
    if 'general' not in routes:
        context['general_stats'] = general_stats(df)

    return context


def _call_groq(question, context, stream=False):
    """
    Call Groq LLM with question and data context.
    Returns the response text, or a generator if stream=True.
    """
    client = _get_groq_client()
    if client is None:
        return None

    user_message = (
        f"Question: {question}\n\n"
        f"Data Context:\n{json.dumps(context, indent=2, default=str)}"
    )

    kwargs = {
        "model": PRIMARY_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": stream,
    }

    try:
        response = client.chat.completions.create(**kwargs)

        if stream:
            return response  # Return the stream generator
        return response.choices[0].message.content

    except Exception as e:
        # Try fallback model
        try:
            kwargs["model"] = FALLBACK_MODEL
            kwargs["stream"] = False
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception:
            return None


def _fallback_response(context):
    """Generate a template response when LLM is unavailable."""
    parts = []

    if 'traffic_analysis' in context:
        data = context['traffic_analysis']
        worst = max(data, key=lambda k: data[k]['avg_time'])
        best = min(data, key=lambda k: data[k]['avg_time'])
        parts.append(
            f"🚦 **Traffic Impact**: {worst} traffic causes the longest delays "
            f"(avg {data[worst]['avg_time']} min), while {best} traffic is fastest "
            f"(avg {data[best]['avg_time']} min)."
        )

    if 'weather_analysis' in context:
        data = context['weather_analysis']
        worst = max(data, key=lambda k: data[k]['avg_time'])
        parts.append(
            f"🌧️ **Weather Impact**: {worst} conditions cause the most delays "
            f"(avg {data[worst]['avg_time']} min)."
        )

    if 'vehicle_analysis' in context:
        data = context['vehicle_analysis']
        best = min(data, key=lambda k: data[k]['avg_time'])
        parts.append(
            f"🏍️ **Vehicle Performance**: {best} is the fastest vehicle type "
            f"(avg {data[best]['avg_time']} min)."
        )

    if 'city_analysis' in context:
        data = context['city_analysis']
        worst = max(data, key=lambda k: data[k]['avg_time'])
        parts.append(
            f"🏙️ **City Performance**: {worst} has the highest delivery times "
            f"(avg {data[worst]['avg_time']} min)."
        )

    if 'peak_hour_analysis' in context:
        data = context['peak_hour_analysis']
        if 'peak' in data and 'non_peak' in data:
            diff = data['peak']['avg_time'] - data['non_peak']['avg_time']
            parts.append(
                f"⏰ **Peak Hours**: Peak deliveries take {data['peak']['avg_time']} min "
                f"vs {data['non_peak']['avg_time']} min off-peak "
                f"(+{diff:.1f} min difference)."
            )

    if 'rating_analysis' in context:
        data = context['rating_analysis']
        best = min(data, key=lambda k: data[k]['avg_time'])
        parts.append(
            f"⭐ **Driver Ratings**: Drivers rated {best} deliver fastest "
            f"(avg {data[best]['avg_time']} min)."
        )

    if not parts:
        stats = context.get('general_stats') or context.get('comprehensive_summary', {}).get('general', {})
        if stats:
            parts.append(
                f"📊 **Overview**: {stats.get('total_deliveries', 'N/A'):,} deliveries, "
                f"avg time {stats.get('avg_delivery_time', 'N/A')} min, "
                f"avg distance {stats.get('avg_distance', 'N/A')} km."
            )

    return "\n\n".join(parts) if parts else "I don't have enough context to answer that question. Try asking about traffic, weather, vehicles, cities, ratings, or peak hours."


# ── Public API ───────────────────────────────────────────

def ai_agent(question, df, stream=False):
    """
    Complete AI agent — routes question, gathers context, calls LLM.

    Args:
        question: User's natural language question
        df: Filtered delivery DataFrame
        stream: If True, returns a stream generator for real-time display

    Returns:
        str or generator: AI response
    """
    context = _get_context(question, df)

    if stream:
        result = _call_groq(question, context, stream=True)
        if result is None:
            # Can't stream fallback, return as single string
            return _fallback_response(context)
        return result

    response = _call_groq(question, context, stream=False)
    if response is None:
        return _fallback_response(context)
    return response


def answer_question(question, df):
    """
    Basic rule-based agent — returns raw analysis data (no LLM).
    Used for the 'View Raw Data' expander.
    """
    context = _get_context(question, df)
    return json.dumps(context, indent=2, default=str)