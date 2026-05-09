import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime
from pathlib import Path

from agents.agent import ai_agent, answer_question
from model.predict import load_metadata, predict_delivery_time

# ====================================================
# 1. PAGE CONFIGURATION
# ====================================================

st.set_page_config(
    page_title="DeliverIQ - Food Delivery Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# 2. CUSTOM STYLING
# ====================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }

    /* KPI card styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }

    /* Chat styling */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Model badge */
    .model-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .model-badge-rf {
        background: #d4edda;
        color: #155724;
    }
    .model-badge-lr {
        background: #cce5ff;
        color: #004085;
    }

    /* Section divider */
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================
# 3. LOAD DATA & MODEL
# ====================================================

@st.cache_data
def load_data():
    """Load and cache the cleaned delivery data."""
    try:
        df = pd.read_csv('cleaned_delivery_data.csv')
        return df
    except FileNotFoundError:
        st.error("cleaned_delivery_data.csv not found. Please ensure the file is in the project directory.")
        return None

@st.cache_resource
def load_model():
    """Load and cache the trained model."""
    try:
        model = joblib.load('delivery_time_model.pkl')
        return model
    except FileNotFoundError:
        st.error("delivery_time_model.pkl not found. Run 'python -m model.train' first.")
        return None

@st.cache_data
def get_model_metadata():
    """Load model metadata for display."""
    return load_metadata()

# Load data and model
df = load_data()
model = load_model()
metadata = get_model_metadata()

if df is None or model is None:
    st.stop()

# ====================================================
# 4. HEADER
# ====================================================

st.markdown("""
<div class="main-header">
    <h1>🚚 DeliverIQ — Food Delivery Intelligence</h1>
    <p>AI-powered analytics dashboard with real-time delivery predictions and Groq LLM insights</p>
</div>
""", unsafe_allow_html=True)

# ====================================================
# 5. SIDEBAR FILTERS
# ====================================================

st.sidebar.title("🎛️ Filters")

# Model info in sidebar
if metadata:
    model_type = metadata.get('model_type', 'Unknown')
    badge_class = 'model-badge-rf' if 'Random' in model_type else 'model-badge-lr'
    r2 = metadata.get('test_metrics', {}).get('r2', 'N/A')
    st.sidebar.markdown(
        f'<span class="model-badge {badge_class}">🤖 {model_type} (R²: {r2})</span>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

# City filter
cities = sorted(df['city'].dropna().unique())
selected_cities = st.sidebar.multiselect(
    "Select Cities:",
    cities,
    default=cities,
    key="city_filter"
)

# Weather filter
weather_conditions = sorted(df['weatherconditions'].dropna().unique())
selected_weather = st.sidebar.multiselect(
    "Weather Conditions:",
    weather_conditions,
    default=weather_conditions,
    key="weather_filter"
)

# Traffic filter
traffic_levels = sorted(df['road_traffic_density'].dropna().unique())
selected_traffic = st.sidebar.multiselect(
    "Traffic Density:",
    traffic_levels,
    default=traffic_levels,
    key="traffic_filter"
)

# Vehicle filter
vehicles = sorted(df['type_of_vehicle'].dropna().unique())
selected_vehicles = st.sidebar.multiselect(
    "Vehicle Types:",
    vehicles,
    default=vehicles,
    key="vehicle_filter"
)

# Festival filter
festivals = sorted(df['festival'].dropna().unique())
selected_festivals = st.sidebar.multiselect(
    "Festival Status:",
    festivals,
    default=festivals,
    key="festival_filter"
)

# Apply filters
filtered_df = df[
    (df['city'].isin(selected_cities)) &
    (df['weatherconditions'].isin(selected_weather)) &
    (df['road_traffic_density'].isin(selected_traffic)) &
    (df['type_of_vehicle'].isin(selected_vehicles)) &
    (df['festival'].isin(selected_festivals))
].copy()

if len(filtered_df) == 0:
    st.warning("No data matches the selected filters. Please adjust your selections.")
    st.stop()

# ====================================================
# 6. KPI SECTION
# ====================================================

st.header("📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_delivery_time = filtered_df['time_takenmin'].mean()
    st.metric(
        label="Avg Delivery Time",
        value=f"{avg_delivery_time:.1f} min",
    )

with col2:
    avg_distance = filtered_df['distance_km'].mean()
    st.metric(
        label="Avg Distance",
        value=f"{avg_distance:.1f} km",
    )

with col3:
    avg_pickup_delay = filtered_df['pickup_delay_min'].mean()
    st.metric(
        label="Avg Pickup Delay",
        value=f"{avg_pickup_delay:.1f} min",
    )

with col4:
    total_deliveries = len(filtered_df)
    st.metric(
        label="Total Deliveries",
        value=f"{total_deliveries:,}",
    )

with col5:
    peak_hour_pct = (filtered_df['is_peak_hour'].sum() / len(filtered_df)) * 100
    st.metric(
        label="Peak Hour %",
        value=f"{peak_hour_pct:.1f}%",
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================
# 7. DELIVERY TIME ANALYTICS
# ====================================================

st.header("📈 Delivery Time Analytics")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚦 Traffic Impact", "🌧️ Weather Impact", "📍 Distance Analysis",
    "🏍️ Vehicle Performance", "⏰ Peak Hour Analysis"
])

with tab1:
    st.subheader("Traffic Density vs Delivery Time")
    traffic_data = filtered_df.groupby('road_traffic_density')['time_takenmin'].mean().reset_index()
    traffic_data = traffic_data.sort_values('time_takenmin', ascending=False)
    fig_traffic = px.bar(
        traffic_data,
        x='road_traffic_density',
        y='time_takenmin',
        title="Average Delivery Time by Traffic Density",
        labels={'road_traffic_density': 'Traffic Density', 'time_takenmin': 'Avg Delivery Time (min)'},
        color='time_takenmin',
        color_continuous_scale='Reds'
    )
    fig_traffic.update_layout(height=400)
    st.plotly_chart(fig_traffic, use_container_width=True)

with tab2:
    st.subheader("Weather Conditions vs Delivery Time")
    weather_data = filtered_df.groupby('weatherconditions')['time_takenmin'].mean().reset_index()
    weather_data = weather_data.sort_values('time_takenmin', ascending=False)
    fig_weather = px.bar(
        weather_data,
        x='weatherconditions',
        y='time_takenmin',
        title="Average Delivery Time by Weather Condition",
        labels={'weatherconditions': 'Weather Condition', 'time_takenmin': 'Avg Delivery Time (min)'},
        color='time_takenmin',
        color_continuous_scale='Blues'
    )
    fig_weather.update_layout(height=400)
    st.plotly_chart(fig_weather, use_container_width=True)

with tab3:
    st.subheader("Distance vs ETA Analysis")
    fig_distance = px.scatter(
        filtered_df.sample(min(2000, len(filtered_df)), random_state=42),
        x='distance_km',
        y='time_takenmin',
        title="Distance vs Delivery Time (sampled for performance)",
        labels={'distance_km': 'Distance (km)', 'time_takenmin': 'Delivery Time (min)'},
        trendline="ols",
        opacity=0.5,
        color_discrete_sequence=['#667eea']
    )
    fig_distance.update_layout(height=400)
    st.plotly_chart(fig_distance, use_container_width=True)

with tab4:
    st.subheader("Vehicle Type Performance")
    fig_vehicle = px.box(
        filtered_df,
        x='type_of_vehicle',
        y='time_takenmin',
        title="Delivery Time Distribution by Vehicle Type",
        labels={'type_of_vehicle': 'Vehicle Type', 'time_takenmin': 'Delivery Time (min)'},
        color='type_of_vehicle',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_vehicle.update_layout(height=400)
    st.plotly_chart(fig_vehicle, use_container_width=True)

with tab5:
    st.subheader("Peak Hour vs Non-Peak Comparison")
    peak_comparison = filtered_df.groupby('is_peak_hour')['time_takenmin'].agg(['mean', 'count']).reset_index()
    peak_comparison['is_peak_hour'] = peak_comparison['is_peak_hour'].map({0: 'Non-Peak', 1: 'Peak'})
    fig_peak = px.bar(
        peak_comparison,
        x='is_peak_hour',
        y='mean',
        title="Average Delivery Time: Peak vs Non-Peak Hours",
        labels={'is_peak_hour': 'Time Period', 'mean': 'Avg Delivery Time (min)'},
        color='is_peak_hour',
        text='count',
        color_discrete_sequence=['#48bb78', '#f56565'],
    )
    fig_peak.update_layout(height=400)
    st.plotly_chart(fig_peak, use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================
# 8. GEOSPATIAL MAP
# ====================================================

st.header("🗺️ Delivery Network Map")

col_map1, col_map2 = st.columns(2)

with col_map1:
    st.subheader("Restaurant Locations")
    map_sample = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
    fig_restaurants = px.scatter_mapbox(
        map_sample,
        lat='restaurant_latitude',
        lon='restaurant_longitude',
        color='time_takenmin',
        size='time_takenmin',
        title="Restaurant Locations (Color = Delivery Time)",
        labels={'time_takenmin': 'Delivery Time (min)'},
        color_continuous_scale='Viridis',
        zoom=10,
        height=400,
    )
    fig_restaurants.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_restaurants, use_container_width=True)

with col_map2:
    st.subheader("Delivery Locations")
    fig_deliveries = px.scatter_mapbox(
        map_sample,
        lat='delivery_location_latitude',
        lon='delivery_location_longitude',
        color='time_takenmin',
        size='time_takenmin',
        title="Delivery Locations (Color = Delivery Time)",
        labels={'time_takenmin': 'Delivery Time (min)'},
        color_continuous_scale='Plasma',
        zoom=10,
        height=400,
    )
    fig_deliveries.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_deliveries, use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================
# 9. ETA PREDICTION PANEL
# ====================================================

st.header("🔮 ETA Prediction Engine")

with st.expander("📝 Enter Delivery Parameters", expanded=True):
    col_pred1, col_pred2, col_pred3 = st.columns(3)

    with col_pred1:
        pred_distance = st.slider("Distance (km)", 0.0, 50.0, 10.0, key="pred_distance")
        pred_pickup_delay = st.slider("Pickup Delay (min)", 0.0, 60.0, 10.0, key="pred_pickup_delay")
        pred_traffic = st.selectbox("Traffic Score", [0, 1, 2, 3, 4], index=2, key="pred_traffic",
                                    help="1=Low, 2=Medium, 3=High, 4=Jam")
        pred_weather = st.selectbox("Weather Score", [0, 1, 2, 3, 4, 5], index=1, key="pred_weather",
                                    help="0=Sunny, 1=Cloudy, 2=Windy, 3=Fog, 4=Stormy, 5=Sandstorms")

    with col_pred2:
        pred_age = st.slider("Delivery Person Age", 18, 50, 30, key="pred_age")
        pred_rating = st.slider("Delivery Person Rating", 1.0, 5.0, 4.0, step=0.1, key="pred_rating")
        pred_vehicle_condition = st.slider("Vehicle Condition", 0, 2, 1, key="pred_vehicle_condition",
                                           help="0=Poor, 1=Average, 2=Good")
        pred_multiple_deliveries = st.slider("Multiple Deliveries", 0.0, 3.0, 1.0, key="pred_multiple_deliveries")

    with col_pred3:
        pred_hour = st.slider("Order Hour", 0, 23, 12, key="pred_hour")
        pred_peak = st.selectbox("Is Peak Hour?", [0, 1], key="pred_peak",
                                 help="Peak hours: 12-14, 19-22")

    if st.button("🚀 Predict ETA", type="primary"):
        input_data = {
            'distance_km': pred_distance,
            'pickup_delay_min': pred_pickup_delay,
            'traffic_score': pred_traffic,
            'weather_score': pred_weather,
            'delivery_person_age': pred_age,
            'delivery_person_ratings': pred_rating,
            'vehicle_condition': pred_vehicle_condition,
            'multiple_deliveries': pred_multiple_deliveries,
            'order_hour': pred_hour,
            'is_peak_hour': pred_peak,
        }

        result = predict_delivery_time(model, input_data)

        st.success(f"🎯 Predicted Delivery Time: **{result['prediction']} minutes**")
        st.markdown(
            f"**Delay Category:** <span style='color:{result['color']}'>{result['category']}</span>",
            unsafe_allow_html=True,
        )

        st.subheader("📋 Operational Insights")
        for insight in result['insights']:
            st.info(f"💡 {insight}")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================
# 10. MODEL INSIGHTS SECTION
# ====================================================

st.header("🧠 Model Insights & Feature Importance")

if metadata:
    # Feature importance chart
    importances = metadata.get('feature_importances', {})
    if importances:
        imp_df = pd.DataFrame([
            {'Feature': k, 'Importance': v}
            for k, v in importances.items()
        ]).sort_values('Importance', ascending=True)

        model_type = metadata.get('model_type', 'Model')
        chart_title = (
            "Feature Importance (Random Forest)"
            if 'Random' in model_type
            else "Feature Coefficients (Linear Regression)"
        )

        fig_imp = px.bar(
            imp_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=chart_title,
            labels={'Importance': 'Impact', 'Feature': 'Feature'},
            color='Importance',
            color_continuous_scale='RdBu' if 'Linear' in model_type else 'Viridis',
        )
        fig_imp.update_layout(height=450)
        st.plotly_chart(fig_imp, use_container_width=True)

    # Model comparison table
    st.subheader("📊 Model Comparison")
    all_results = metadata.get('all_results', {})
    if all_results:
        comparison_data = []
        for name, results in all_results.items():
            test = results.get('test', {})
            comparison_data.append({
                'Model': name,
                'R² Score': test.get('r2', 'N/A'),
                'MAE (min)': test.get('mae', 'N/A'),
                'RMSE (min)': test.get('rmse', 'N/A'),
                'CV R² (5-fold)': results.get('cv_r2', 'N/A'),
            })
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Key insights
    st.subheader("🔍 Key Insights")
    col_insight1, col_insight2 = st.columns(2)

    best_metrics = metadata.get('test_metrics', {})
    with col_insight1:
        st.info(
            f"**Model Performance:** {metadata.get('model_type', 'Model')} achieves "
            f"R² = {best_metrics.get('r2', 'N/A')}, explaining "
            f"{round(best_metrics.get('r2', 0) * 100)}% of delivery time variance."
        )
        # Top feature
        if importances:
            top_feature = max(importances, key=lambda k: abs(importances[k]))
            st.info(f"**Top Feature:** {top_feature} has the highest impact on delivery time predictions.")

    with col_insight2:
        st.info(
            f"**Prediction Accuracy:** Average error of "
            f"{best_metrics.get('mae', 'N/A')} minutes (MAE), "
            f"making predictions reliable for operational planning."
        )
        st.info(
            "**Actionable Insight:** Focus on high-rated drivers, vehicle maintenance, "
            "and traffic-aware routing to optimize delivery performance."
        )

else:
    st.warning("Model metadata not found. Run `python -m model.train` to generate.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================
# 11. BUSINESS INSIGHTS SECTION
# ====================================================

st.header("💼 Automated Business Insights")

col_insight_a, col_insight_b = st.columns(2)

with col_insight_a:
    # Highest delay weather
    weather_delays = filtered_df.groupby('weatherconditions')['time_takenmin'].mean()
    worst_weather = weather_delays.idxmax()
    worst_weather_time = weather_delays.max()
    st.success(f"🌧️ **Highest Delay Weather:** {worst_weather} (avg {worst_weather_time:.1f} min)")

    # Best vehicle type
    vehicle_perf = filtered_df.groupby('type_of_vehicle')['time_takenmin'].mean()
    best_vehicle = vehicle_perf.idxmin()
    best_vehicle_time = vehicle_perf.min()
    st.success(f"🏍️ **Best Vehicle Type:** {best_vehicle} (avg {best_vehicle_time:.1f} min)")

with col_insight_b:
    # Peak hour analysis
    peak_avg = filtered_df[filtered_df['is_peak_hour'] == 1]['time_takenmin'].mean()
    non_peak_avg = filtered_df[filtered_df['is_peak_hour'] == 0]['time_takenmin'].mean()
    st.warning(f"⏰ **Peak Hour ETA:** {peak_avg:.1f} min vs Non-Peak: {non_peak_avg:.1f} min")

    # City with highest delay
    if len(selected_cities) > 1:
        city_delays = filtered_df.groupby('city')['time_takenmin'].mean()
        worst_city = city_delays.idxmax()
        worst_city_time = city_delays.max()
        st.error(f"🏙️ **Highest Delay City:** {worst_city} (avg {worst_city_time:.1f} min)")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================
# 12. AI ASSISTANT (Chat Interface)
# ====================================================

st.header("🤖 DeliverIQ AI Assistant")

# Check if Groq API key is configured
has_api_key = False
try:
    has_api_key = bool(st.secrets["GROQ_API_KEY"])
except (KeyError, FileNotFoundError, Exception):
    import os
    has_api_key = bool(os.environ.get("GROQ_API_KEY"))

if has_api_key:
    st.caption("Powered by Groq LLM (llama-3.3-70b-versatile) — Ask anything about your delivery data!")
else:
    st.caption("⚠️ No GROQ_API_KEY found. Using template responses. Add your key in `.streamlit/secrets.toml` or as an environment variable.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Example questions (shown only when chat is empty)
if not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    example_cols = st.columns(3)
    examples = [
        "How does traffic affect delivery times?",
        "Which vehicle type performs best?",
        "Compare peak vs non-peak hours",
        "What's the impact of weather on delays?",
        "Show city-wise delivery performance",
        "How do driver ratings affect speed?",
    ]
    for i, question in enumerate(examples):
        col = example_cols[i % 3]
        if col.button(question, key=f"example_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# Chat input
if prompt := st.chat_input("Ask about delivery data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        if has_api_key:
            # Try streaming
            try:
                stream = ai_agent(prompt, filtered_df, stream=True)
                if hasattr(stream, '__iter__') and not isinstance(stream, str):
                    # It's a stream — use write_stream
                    def stream_text():
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content
                    response = st.write_stream(stream_text())
                else:
                    # Fallback response (string)
                    response = stream
                    st.markdown(response)
            except Exception:
                # Non-streaming fallback
                response = ai_agent(prompt, filtered_df, stream=False)
                st.markdown(response)
        else:
            response = ai_agent(prompt, filtered_df, stream=False)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # Show raw data in expander
    with st.expander("📊 View Raw Analysis Data"):
        raw_data = answer_question(prompt, filtered_df)
        st.code(raw_data, language="json")

# ====================================================
# 13. FOOTER
# ====================================================

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("### 🚚 DeliverIQ")
    st.markdown("AI-Powered Food Delivery Intelligence")

with col_footer2:
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("Streamlit · Plotly · Scikit-learn · Groq LLM")

with col_footer3:
    st.markdown("### 📊 Data")
    st.markdown(f"{len(df):,} deliveries · {df['city'].nunique()} cities")
    if metadata:
        st.markdown(f"Model: {metadata.get('model_type', 'N/A')} (R²: {metadata.get('test_metrics', {}).get('r2', 'N/A')})")