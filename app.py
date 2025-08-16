import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="🌍 Planet Digital Twin", layout="wide")

st.title("🌍 Planet-Level Global Digital Twin")
st.markdown("Real-time data + visualizations + multi-agent AI summaries")

# -------------------------------
# DATA SOURCES
# -------------------------------
# Earthquake data (USGS)
@st.cache_data(ttl=600)
def get_earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    r = requests.get(url)
    data = r.json()
    quakes = []
    for feat in data["features"]:
        quakes.append({
            "place": feat["properties"]["place"],
            "mag": feat["properties"]["mag"],
            "time": pd.to_datetime(feat["properties"]["time"], unit="ms")
        })
    return pd.DataFrame(quakes)

# COVID-19 data (disease.sh)
@st.cache_data(ttl=600)
def get_covid():
    url = "https://disease.sh/v3/covid-19/all"
    return requests.get(url).json()

# -------------------------------
# VISUALS
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌋 Earthquakes (Past 24h)")
    try:
        eq = get_earthquakes()
        st.dataframe(eq.head(10))
        fig = px.scatter(eq, x="time", y="mag", size="mag", hover_name="place",
                         title="Recent Earthquakes", height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error fetching earthquake data: {e}")

with col2:
    st.subheader("🦠 COVID-19 Global Data")
    try:
        covid = get_covid()
        st.metric("Cases", f"{covid['cases']:,}")
        st.metric("Deaths", f"{covid['deaths']:,}")
        st.metric("Recovered", f"{covid['recovered']:,}")
    except Exception as e:
        st.error(f"Error fetching COVID data: {e}")

# -------------------------------
# MULTI-AGENT SUMMARIES
# -------------------------------
st.markdown("## 📝 Multi-Agent Summaries")
user_text = st.text_area("Paste any text (news, report, API data)", height=200)

if user_text:
    st.write("### Agent 1: Extractive Summary (Sumy LSA)")
    try:
        parser = PlaintextParser.from_string(user_text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 3)  # 3 sentences
        extractive = " ".join([str(s) for s in summary])
        st.success(extractive)
    except Exception as e:
        st.error(f"Error with Sumy summarizer: {e}")

    st.write("### Agent 2: Abstractive Summary (HuggingFace Transformers)")
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        abstractive = summarizer(user_text, max_length=100, min_length=30, do_sample=False)
        st.info(abstractive[0]['summary_text'])
    except Exception as e:
        st.error(f"Error with HuggingFace summarizer: {e}")
