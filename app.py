# Install necessary packages
# Ensure you have these installed:
# pip install streamlit pandas transformers seaborn

# Import required libraries
import streamlit as st
import pandas as pd
import seaborn as sns
from transformers import pipeline

class EmotionDistributionApp:
    def __init__(self, file_path, sample_size=100):
        self.file_path = file_path
        self.sample_size = sample_size
        self.emotion_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased')


    def load_data(self):
        try:
            data = pd.read_csv(self.file_path)
            return data.head(self.sample_size)
        except FileNotFoundError:
            st.error(f"Error: File not found at {self.file_path}")
            return pd.DataFrame()

    def display_emotion_distribution(self, data):
        st.subheader('Emotion Distribution')
        if not data.empty:
            sns_plot = sns.countplot(data=data, y='sentiment')
            st.pyplot(sns_plot.get_figure())
        else:
            st.warning("Sample dataset could not be loaded.")

    def get_user_emotion_input(self):
        user_input = st.text_area('Enter text:', 'Product was amazing but packing was not up to the mark.')
        if st.button('Get Emotion'):
            user_emotion = self.emotion_analyzer(user_input)[0]['label']
            st.write(f'Text Emotion: {user_emotion}')

    def display_sample_dataset(self, data):
        st.subheader('Sample Dataset')
        st.dataframe(data)

    def run_app(self):
        st.title('Shopify Customer Sentiment Analysis App')

        # Load data
        sample_data = self.load_data()

        # Display emotion distribution
        self.display_emotion_distribution(sample_data)

        # Allow users to input text and get the emotion label
        self.get_user_emotion_input()

        # Display the sample dataset
        self.display_sample_dataset(sample_data)

if __name__ == "__main__":
    # Specify the file path for your dataset
    file_path = r'D:\Daniyals project\customer_id,sentiment,author,content.CSV'

    # Create an instance of the app and run it
    emotion_app = EmotionDistributionApp(file_path)
    emotion_app.run_app()
