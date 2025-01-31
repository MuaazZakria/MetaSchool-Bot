# MetaSchool Bot

This project uses Streamlit to build a conversational AI system that retrieves relevant course information based on user queries. The system is powered by OpenAI's GPT-4 model and uses Pinecone as a vector database for storing and retrieving contextual information.

## Prerequisites

Before running the system, ensure you have the following:
- Python 3.11
- Streamlit
- OpenAI API Key
- Pinecone API Key

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/MuaazZakria/MetaSchool-Bot/
2. Install the required demendepecies:
   ```bash
   pip install -r requirements.txt
4. Run the app using:
   ```bash
   streamlit run app.py

## Example Queries:

1. Query: I want to learn solidity which course should I start from?
   Response: You can start with the following courses:

   "Writing your first ever solidity contract" - This course will help you get started with the basics of Solidity and smart contract development.

   "Build and deploy a Social Media dApp using Solidity" - This course will help you apply your Solidity knowledge in a practical project.

   "Build and deploy a Lending app on Core using Solidity" - This course will further enhance your Solidity skills by building a lending app.

2. Query: Can you help me learn about Etheruem?
   Response: Yes, I can certainly help you learn about Ethereum. Here are the top 3 most relevant courses for you:

   "Into the World of Ethereum" - This course provides an overview of Ethereum, its safety features, and why it's popular.
   "What Are We Building Today?" - This course dives deep into the Ethereum Blockchain, explaining its importance and how it works.
   "How does Ethereum work - A deepdive" - This course offers a comprehensive understanding of Ethereum, including its comparison with other blockchains.
