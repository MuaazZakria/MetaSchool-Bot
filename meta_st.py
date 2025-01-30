import streamlit as st
import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()


# Retrieve OpenAI and Pinecone API keys from environment variables
openai_api_key = os.getenv("OPEN_AI_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key)

# Set up the model for embeddings
model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

# Set the index and namespace for Pinecone
index_name = "metaschool"
namespace = "default_namespace"

# Connect to the Pinecone index
index = pinecone.Index(index_name)

# Set up the vector store to interface with Pinecone
vectorstore = PineconeVectorStore(
    index=index, embedding=model, namespace=namespace
)

# Set up the LLM (ChatGPT) model for generating responses
llm = ChatOpenAI(
    model="gpt-4",  # Ensure this model name is correct
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_api_key
)

# Define a prompt template for the model
prompt_template = """
You are an expert Chat Assistant who helps users with their queries.

Given the context, answer the question.

{context}

Question: {question}

INSTRUCTIONS:
- IF the user greets, greet back.
- DO NOT greet with every response.
- IF the context is not similar to the question, respond with 'I don't know the answer'.
- Make the answers short concise and precise.

FORMATTING INSTRUCTION:
- DO NOT add any asterisks in the response.
- Keep the response plain in simple strings.
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Set up memory to store conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history", output_key="answer", return_messages=True
)

# Create the QA chain for conversational retrieval
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(search_kwargs={"k": 3}),  # Retrieve top 3 documents
    return_source_documents=True,
    memory=memory,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

# Set up Streamlit title and interface
st.title("Chatbot for Interns/Trainees")

# Ensure session state is set up for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question..."):
    # Store the user message in session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the assistant's response using the QA chain
    with st.chat_message("assistant"):
        result = qa_chain.invoke({"question": prompt})
        response = result['answer']
        st.markdown(response)

    # Store the assistant's response in session state
    st.session_state.messages.append({"role": "assistant", "content": response})
