import streamlit as st
from model import QABot  #  with the name of the file where you defined QABot

# Create an instance of QABot
bot = QABot()

# Streamlit app layout
st.title("QA Bot")
st.write("Ask any question and the bot will try to provide an answer.")

# User input
user_query = st.text_input("Enter your question here")

# Button to get answer
if st.button("Get Answer"):
    if user_query:
        with st.spinner('Fetching answer...'):
            answer = bot.final_result(user_query)
            st.success("Answer retrieved!")
            st.write(answer)
    else:
        st.warning("Please enter a question.")