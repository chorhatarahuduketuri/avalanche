# import packages
import streamlit as st
import pandas as pd
import re
import os


def clean_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    cleaned = cleaned.lower()
    return cleaned


def get_dataset_path():
    # get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # construct path to CSV file
    csv_path = os.path.join(current_dir, "..", "..", "data", "customer_reviews.csv")
    return csv_path


st.title("Hello, GenAI!")
st.write("This is your GenAI-powered data processing app.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            # st.session_state is basically a dict that gets saved between streamlit reruns/actions/etc
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset loaded successfully! Yay!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path. ")


with col2:
    if st.button("Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned successfully!")
        else:
            st.warning("Please ingest the dataset first!")


# If dataset exists and is loaded, display it
if "df" in st.session_state:
    st.subheader("Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))

    st.subheader("Dataset Preview")
    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"].head()
    st.dataframe(filtered_df)

    st.subheader("Sentiment Score by Product")
    grouped = st.session_state["df"].groupby(["PRODUCT"])["SENTIMENT_SCORE"].mean()
    st.bar_chart(grouped)
else:
    st.write("No dataset loaded yet!")