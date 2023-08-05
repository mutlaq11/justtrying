import streamlit as st
import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report

# Function to load data
@st.cache
def load_data():
    df = pd.read_excel("DATASET_7G.xlsx")
    return df

# Function to classify data
@st.cache(allow_output_mutation=True)
def classify_data(df):
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    return classifier

# Streamlit app
def main():
    st.title("Text Classification App")

    # Load the data
    df = load_data()

    # Initialize the classifier
    classifier = classify_data(df)

    # Define the labels for topic classification and satisfaction classification
    topic_labels = ["transportations", "food", "WC", "guidance"]
    satisfaction_labels = ["satisfied", "unsatisfied"]

    # Input text
    input_text = st.text_input("Input Text")

    # Classify button
    if st.button("Classify"):
        # Classify the input text
        result = classifier(input_text, topic_labels + satisfaction_labels)

        # Display the results
        st.write("Topic: ", result['labels'][0])
        st.write("Satisfaction: ", result['labels'][1])

if __name__ == "__main__":
    main()
