import streamlit as st
import pandas as pd
from transformers import pipeline

# Function to classify data
@st.cache(allow_output_mutation=True)
def classify_data():
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    return classifier

# Streamlit app
def main():
    st.title("Text Classification App")

    # Initialize the classifier
    classifier = classify_data()

    # Define the labels for topic classification and satisfaction classification
    topic_labels = ["transportations", "food", "WC", "guidance"]
    satisfaction_labels = ["satisfied", "unsatisfied"]

    # File upload
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        # Select column
        selected_column = st.selectbox('Select a column', df.columns)

        # Classify button
        if st.button("Classify"):
            # Classify the input text for topic and satisfaction
            df['Topic'] = df[selected_column].apply(lambda row: classifier(row, topic_labels)['labels'][0])
            df['Satisfaction'] = df[selected_column].apply(lambda row: classifier(row, satisfaction_labels)['labels'][0])

            # Display the results
            st.dataframe(df)

if __name__ == "__main__":
    main()
