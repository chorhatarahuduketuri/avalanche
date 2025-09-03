# import packages
import os

import openai
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()
sentiment_analysis_prompt = ("Provide a sentiment analysis of the following customer review between 1 and -1, strictly "
                             "returning nothing but the decimal number between 1 and -1 and no other text or output:")

st.title("Prototype: Automated Customer Review Sentiment Analysis")
st.write("This is a prototype of an automated sentiment analysis of customer reviews for customer support staff. ")

NUMBER_OF_REVIEWS_TO_PROCESS = st.number_input(label="Please select the number of customer reviews to process via AI",
                                               min_value=1, max_value=20, step=1)


def get_reviews_dataset_csv_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "..", "..", "data", "customer_reviews.csv")
    return csv_path


@st.cache_data
def get_sentiment(product_summary: str) -> float:
    response = client.responses.create(model="gpt-4o",
        input=[{"role": "user", "content": f"{sentiment_analysis_prompt} {product_summary}"}], max_output_tokens=16)
    return float(response.output[0].content[0].text)


if st.button("Load and analyse reviews dataset"):
    try:
        csv_path = get_reviews_dataset_csv_path()
        st.session_state["df"] = pd.read_csv(csv_path).sample(NUMBER_OF_REVIEWS_TO_PROCESS)
        st.success("Customer reviews dataset loaded successfully!")
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")

    if "df" in st.session_state:
        with st.spinner("Analysing Review Dataset", show_time=True):
            st.session_state["df"]["SENTIMENT"] = st.session_state["df"]["SUMMARY"].apply(get_sentiment)
            st.success("Dataset loaded and analysed successfully!")
    else:
        st.warning("Dataset not found. Please load the dataset first.")

if "df" in st.session_state:
    # Filter displayed data by product
    st.subheader("Filter by Product")
    product = st.selectbox("Choose All Products or a specific product:",
                           ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))

    # Display data in df
    st.subheader(f"Product Reviews: {product}")
    if product == "All Products":
        filtered_df = st.session_state["df"]
    else:
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    st.dataframe(filtered_df, use_container_width=True)

    # Show graph of GenAI generated sentiment analysis scores for each customer review
    mean_sentiment_by_product_df = filtered_df.groupby("PRODUCT", as_index=False)["SENTIMENT"].mean()
    per_product_plotly_chart = px.bar(mean_sentiment_by_product_df, x="PRODUCT", y="SENTIMENT", color="PRODUCT",
                                      title="Average sentiment scores by product")
    st.plotly_chart(per_product_plotly_chart, use_container_width=True)

    sentiment_scores = pd.to_numeric(filtered_df["SENTIMENT"], errors="coerce").dropna()
    # Define the fixed set of sentiment values
    bins = np.round(np.arange(-1.0, 1.01, 0.1), 1)
    # Count sentiment score values
    counts = (
        sentiment_scores.value_counts(sort=False)
        .rename_axis("sentiment_bin")
        .reindex(bins, fill_value=0)  # ensures missing bins are included
        .reset_index(name="count")
    )
    # Make x categorical so order is preserved
    counts["sentiment_bin"] = pd.Categorical(
        counts["sentiment_bin"].map(lambda x: f"{x:.1f}"),
        categories=[f"{x:.1f}" for x in bins],
        ordered=True
    )
    all_products_plotly_chart = px.bar(
        counts,
        x="sentiment_bin",
        y="count",
        title=f"Distribution of sentiment scores for: {product}",
        labels={"sentiment_bin": "Sentiment", "count": "Review Count"},
    )
    st.plotly_chart(all_products_plotly_chart, use_container_width=True)

else:
    st.write("No dataset loaded yet. Please load a dataset first, and then click on 'Analyse Review Dataset'.")
