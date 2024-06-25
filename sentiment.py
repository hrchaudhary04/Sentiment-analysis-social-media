import datetime
from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import plotly.express as px

st.header('Sentiment Analysis')

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Emotion: ', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))

    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                 stopwords=True, lowercase=True, numbers=True, punct=True))

with st.expander('Fetch From Twitter'):
    search_term = st.text_input('What you want to search: ')
    num_tweets = st.number_input('Number of tweets to fetch: ')
    starting_date = st.date_input('Starting Date: ')
    finishing_date = st.date_input('Finishing Date: ')

    if st.button('Scrape and Analyze tweets'):
        # Add the following lines before loading the data
        fetched_data = {'Search Term': [search_term], 'Date Fetched': [datetime.datetime.now()], 'Number of Tweets': [num_tweets]}
        st.write('Fetched data:')
        st.write(fetched_data)

        # ... (rest of the code for scraping and analyzing tweets)

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def score(x):
        # Check if x is a string before creating TextBlob object
        if isinstance(x, str):
            blob1 = TextBlob(x)
            return blob1.sentiment.polarity
        else:
            return 0  # Or any other value you consider appropriate for non-text inputs

    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        df = pd.read_excel(upl)

        # Drop columns containing 'unnamed' in their name (case-insensitive)
        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

        # Assuming the column name containing tweet text is 'text', change 'tweets' to 'text'
        df['score'] = df['text'].apply(score)
        df['analysis'] = df['score'].apply(analyze)

        # Create a bar chart using Plotly
        fig = px.histogram(df, x='analysis', nbins=3, title='Sentiment Analysis')
        st.plotly_chart(fig)

        st.write(df.head(10))

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )