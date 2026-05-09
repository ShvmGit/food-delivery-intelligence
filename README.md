<p align="center">
  <h1 align="center">🚚 DeliverIQ — Food Delivery Intelligence</h1>
  <p align="center">
    AI-powered analytics dashboard for food delivery operations<br/>
    Built with Streamlit · Plotly · Scikit-learn · Groq LLM
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/ML-Random%20Forest-green" alt="ML"/>
  <img src="https://img.shields.io/badge/LLM-Groq%20(Llama%203.3)-orange" alt="LLM"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📊 KPI Dashboard** | Real-time metrics with dynamic filtering (5 filters) |
| **📈 Interactive Charts** | Traffic, weather, distance, vehicle & peak hour analytics (Plotly) |
| **🗺️ Geospatial Maps** | Restaurant & delivery location mapping with OpenStreetMap |
| **🔮 ETA Predictions** | ML-powered delivery time predictions with delay categorization |
| **🤖 AI Assistant** | Natural language Q&A powered by Groq LLM (Llama 3.3 70B) |
| **🧠 Model Insights** | Feature importance visualization with model comparison |
| **💼 Business Insights** | Automated operational recommendations |

## 🤖 AI Assistant

Ask natural language questions about your delivery data:

- *"How does traffic affect delivery times?"*
- *"Which vehicle type performs best?"*
- *"Compare peak vs non-peak hours"*
- *"What's the impact of weather on delays?"*
- *"How do driver ratings affect delivery speed?"*

The AI assistant uses **Groq LLM (llama-3.3-70b-versatile)** for intelligent, context-aware responses with streaming support. Falls back to template responses if API is unavailable.

## 📈 Model Performance

| Model | R² Score | MAE | RMSE | CV R² (5-fold) |
|-------|---------|-----|------|----------------|
| **Random Forest** ✅ | 0.8352 | 3.11 min | 3.85 min | 0.8298 |
| Linear Regression | 0.5149 | 5.25 min | 6.60 min | 0.5099 |

> Random Forest was auto-selected as the best performer — **62% improvement** in R² over Linear Regression.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Groq API Key](https://console.groq.com/keys) (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/food-delivery-intelligence.git
cd food-delivery-intelligence
```

### 2. Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

Or set environment variable:

```bash
# Windows
set GROQ_API_KEY=your-groq-api-key-here
# macOS/Linux
export GROQ_API_KEY=your-groq-api-key-here
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy to Streamlit Cloud

1. **Push to GitHub** — Push this repository to your GitHub account

2. **Go to [share.streamlit.io](https://share.streamlit.io)** — Sign in with GitHub

3. **Create new app:**
   - Select your repository
   - Branch: `main`
   - Main file: `app.py`

4. **Add secrets:**
   - Go to App Settings → Secrets
   - Add: `GROQ_API_KEY = "your-key-here"`

5. **Deploy** — Your app will be live!

---

## 📁 Project Structure

```
food-delivery-intelligence/
├── app.py                        # Main Streamlit dashboard
├── agents/
│   ├── __init__.py
│   ├── agent.py                  # Groq LLM integration + routing
│   └── tools.py                  # Analytics tools (9 functions)
├── model/
│   ├── __init__.py
│   ├── train.py                  # Train LR + RF, auto-select best
│   └── predict.py                # Prediction with validation
├── cleaned_delivery_data.csv     # Processed dataset (39K rows)
├── delivery_time_model.pkl       # Trained Random Forest model
├── model_metadata.json           # Model metrics & feature importances
├── requirements.txt              # Python dependencies
├── .gitignore
├── .streamlit/
│   └── secrets.toml              # API keys (gitignored)
├── data_cleaning_summary.txt     # Data cleaning documentation
├── delivery_analysis_summary.txt # Analysis summary
├── phase3_model_insights.txt     # Feature coefficient analysis
└── README.md
```

## 🔧 Retrain the Model

To retrain with updated data:

```bash
python -m model.train
```

This will:
1. Train both Linear Regression and Random Forest
2. Compare using 5-fold cross-validation
3. Auto-select and save the best model
4. Generate `model_metadata.json` with metrics

## 🛠️ Tech Stack

- **Dashboard**: [Streamlit](https://streamlit.io)
- **Visualization**: [Plotly](https://plotly.com)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org) (Random Forest)
- **LLM**: [Groq](https://groq.com) (Llama 3.3 70B Versatile)
- **Data**: [Pandas](https://pandas.pydata.org), [NumPy](https://numpy.org)

## 📊 Data Pipeline

```
FDD.txt (Raw JSON) → Code.py → FDD.csv → cod1.py → cleaned_delivery_data.csv
                                                          ↓
                                                    model/train.py → delivery_time_model.pkl
                                                          ↓
                                                      app.py (Dashboard)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 🗺️ Roadmap

- [ ] XGBoost model comparison
- [ ] Real-time data ingestion
- [ ] Multi-page Streamlit app
- [ ] A/B testing framework for delivery strategies
- [ ] Export reports as PDF

## 📄 License

This project is open source and available under the [MIT License](LICENSE).