import streamlit as st

st.title("Berkshire Hathaway Annual Letters GPT")

# Define avatars
user_avatar = 'baby.png'
assistant_avatar = 'warren.png'

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the best investing advice you can give me?"):
    # Display user message in chat message container
    st.chat_message("user", avatar=user_avatar).markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "avatar": user_avatar, "content": prompt})

    response = f"Warren: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message(name="assistant", avatar=assistant_avatar):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "avatar": assistant_avatar, "content": response})



===================



import torch
import subprocess
import streamlit as st
from run_localGPT import load_model
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


def model_memory():
    # Adding history to the model.
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory

if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"


# if "result" not in st.session_state:
#     # Run the document ingestion process.
#     run_langest_commands = ["python", "ingest.py"]
#     run_langest_commands.append("--device_type")
#     run_langest_commands.append(DEVICE_TYPE)

#     result = subprocess.run(run_langest_commands, capture_output=True)
#     st.session_state.result = result

# Define the retreiver
# load the vectorstore
if "EMBEDDINGS" not in st.session_state:
    EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
    st.session_state.EMBEDDINGS = EMBEDDINGS

if "DB" not in st.session_state:
    DB = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=st.session_state.EMBEDDINGS,
        client_settings=CHROMA_SETTINGS,
    )
    st.session_state.DB = DB

if "RETRIEVER" not in st.session_state:
    RETRIEVER = DB.as_retriever()
    st.session_state.RETRIEVER = RETRIEVER

if "LLM" not in st.session_state:
    LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
    st.session_state["LLM"] = LLM


if "QA" not in st.session_state:
    prompt, memory = model_memory()

    QA = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=RETRIEVER,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    st.session_state["QA"] = QA



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    role = message["role"]
    avatar = user_avatar if role == "user" else assistant_avatar

# React to user input
if prompt := st.chat_input("What is up?"):
    # Then pass the prompt to the LLM
    response = st.session_state["QA"](prompt)
    answer = response["result"]
    # ...and write it out to the screen
    st.write(answer)

What is the investment strategy of Berkshire



# # Display user message in chat message container
# st.chat_message("user", avatar=user_avatar).markdown(prompt)
# # Add user message to chat history
# st.session_state.messages.append({"role": "user", "avatar": user_avatar, "content": prompt})

# response = f"Warren: {prompt}"
# # Display assistant response in chat message container
# with st.chat_message(name="assistant", avatar=assistant_avatar):
#     st.markdown(response)
# # Add assistant response to chat history
# st.session_state.messages.append({"role": "assistant", "avatar": assistant_avatar, "content": response})


