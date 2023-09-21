import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
import openai
from llama_hub.youtube_transcript import YoutubeTranscriptReader


with open("video_urls.txt") as f:
    youtube_links = f.readlines()

st.set_page_config(
    page_title="Chat with the Avenza Maps",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
openai.api_key = st.secrets.openai_key
st.title("Chat with the Avenza Maps YouTube content")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Avenza Maps YouTube content!",
        }
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(
        text="Loading and indexing the Avenza Maps docs – hang tight! This should take 1-2 minutes."
    ):
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=youtube_links[:20])
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo-16k",
                temperature=0.3,
                system_prompt="""You are an expert on Avenza Maps software product and your job is to answer technical questions.
                Do not answer the questions that are irrelevant to the information provided in the context. Keep your answers technical and based on facts - do not hallucinate features.""",
            )
        )
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        return index


index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = openai.Moderation.create(prompt)
            if response["results"][0]["flagged"]:
                content = "Please refrain from using profanity."
                st.write(content)
            else:
                response = chat_engine.chat(prompt)
                content = response.response
                st.write(content)

            message = {"role": "assistant", "content": content}
            st.session_state.messages.append(message)  # Add response to message history
