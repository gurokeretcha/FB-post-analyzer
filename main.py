import streamlit as st
from facebook_scraper import get_posts, set_cookies
import facebook_scraper as fs
import time
from collections import defaultdict
import pandas as pd
from googletrans import Translator
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
translator = Translator()
def pos_neg_neu(compound_score):
    sentiment = None
    if compound_score > 0.05:
        sentiment='Positive'
    elif compound_score < -0.05:
        sentiment='Negative'
    else:
        sentiment='Neutral'
    return sentiment

st.header("FB Post Analytics")

text_input = st.text_input(
    "Enter post ID 👇"
)
if st.button("RUN!"):

    st.write("Please wait...")

    all_data =defaultdict(list)
    POST_ID = text_input

    MAX_COMMENTS = True

    gen = fs.get_posts(
        post_urls=[POST_ID],
        options={"comments": MAX_COMMENTS, "progress": True}
    )

    # take 1st element of the generator which is the post we requested
    post = next(gen)

    # extract the comments part
    comments = post['comments_full']

    # process comments as you want...
    for comment in comments:
        # st.write(comment)
        # e.g. ...print them
        # print(comment)
        time.sleep(2)
        # e.g. ...get the replies for them
        # for reply in comment['replies']:
            # print(' ', reply)

        all_data['commenter_id'].append(comment['comment_id'])
        all_data['commenter_name'].append(comment['commenter_name'])
        all_data['commenter_text'].append(comment['comment_text'])
        all_data['comment_reactors'].append(comment['comment_reactors'])
        all_data['comment_reaction_count'].append(comment['comment_reaction_count'])

    all_data_df = pd.DataFrame(all_data)
    # all_data_df[['commenter_id','commenter_name','commenter_text']].to_csv('post_commnets.csv',index=False)
    # st.dataframe(all_data_df[['commenter_id','commenter_name','commenter_text']].head(20))

    all_data_df['comment_translated'] = all_data_df['commenter_text'].apply(lambda x:translator.translate(x,'en').text)
    all_data_df['sentiment_score'] = all_data_df['comment_translated'].apply(lambda x:  analyzer.polarity_scores(x)['compound'])
    all_data_df['sentiment'] = all_data_df['sentiment_score'].apply(lambda x: pos_neg_neu(x))


    # Group the dataframe by category and count the frequency of each category
    freq_df = all_data_df.groupby(['sentiment']).count()[['commenter_text']]

    # Plot a bar chart of the frequency dataframe
    st.bar_chart(freq_df)

    st.write('Prediction result')
    st.dataframe(all_data_df[['commenter_id','commenter_name','commenter_text','sentiment']].head(20))

    # st.write('Examples of positive comments:')
    # st.dataframe(all_data_df[all_data_df['sentiment']=='Positive'].sample(3))
    # st.write('Examples of negative comments:')
    # st.dataframe(all_data_df[all_data_df['sentiment']=='Negative'].sample(3))
    # st.write('Examples of neutral comments:')
    # st.dataframe(all_data_df[all_data_df['sentiment']=='Neutral'].sample(3))