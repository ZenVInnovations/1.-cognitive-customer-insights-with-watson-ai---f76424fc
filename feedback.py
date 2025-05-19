import streamlit as st
import pandas as pd
import os
from datetime import datetime
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EntitiesOptions

# ---------------- Watson Setup ----------------
api_key = 'mjsRV0l_UVIkc5fIE3u9tplsTFw-7haLuk4ntICJq6IR'
service_url = 'https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/d2a98d85-7c0b-44b2-9231-c4b2c3aa06ca'

authenticator = IAMAuthenticator(api_key)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(service_url)

# ---------------- Feedback Input ----------------
st.title("Customer Feedback Sentiment Dashboard")
st.markdown("---")

user_name = st.text_input("Enter your name:")
feedback_input = st.text_area("Enter your feedback:", height=200)

if st.button("Submit Feedback"):
    if not user_name.strip():
        st.warning("Please enter your name before submitting.")
    else:
        feedback_list = [line.strip() for line in feedback_input.split('\n') if line.strip()]
        results = []

        for feedback in feedback_list:
            try:
                response = nlu.analyze(
                    text=feedback,
                    features=Features(
                        sentiment=SentimentOptions(),
                        entities=EntitiesOptions(sentiment=True, emotion=True, limit=5)
                    )
                ).get_result()

                sentiment = response['sentiment']['document']['label']
                score = response['sentiment']['document']['score']
                rating = int(((score + 1) / 2) * 9 + 1)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                results.append({
                    'Name': user_name,
                    'Timestamp': timestamp,
                    'Feedback': feedback,
                    'Sentiment': sentiment,
                    'Rating (1-10)': rating
                })

            except Exception as e:
                st.error(f"Error analyzing: {feedback}\n{e}")

        if results:
            df = pd.DataFrame(results)
            file_path = "feedback_data.xlsx"

            if os.path.exists(file_path):
                existing_df = pd.read_excel(file_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                combined_df = df

            combined_df.to_excel(file_path, index=False)
            st.success(f"Feedback saved to '{file_path}'.")

            st.markdown("### Feedback Summary")
            st.dataframe(df[['Name', 'Timestamp', 'Feedback', 'Sentiment', 'Rating (1-10)']])

            # ---------------- Rating with Progress Bar ----------------
            st.markdown("### Feedback Ratings")
            for _, row in df.iterrows():
                st.markdown(f"**{row['Name']} ({row['Timestamp']}):** {row['Feedback']}")
                st.markdown(f"**Sentiment:** {row['Sentiment']} — **Rating:** {row['Rating (1-10)']} / 10")
                st.progress(row['Rating (1-10)'] / 10)
                st.markdown("---")