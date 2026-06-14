import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# DYNAMIC PATH RESOLUTION
# Ensures the script can find the database

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
CHROMA_PATH = os.path.join(BASE_DIR, "vectorstore", "chroma_db")

EMBEDDING_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen2.5-coder:3b"

def main():
    print("Initializing FactSheet-IQ Query Engine...")

    # LOADING THE EXISTING VECTOR DATABASE

    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Vector database not found at {CHROMA_PATH}")
        return

    embeddings = OllamaEmbeddings(model = EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # CONFIGURE THE DATABASE

    retriver = db.as_retriever(search_kwargs={"k": 3})

    # INITIALIZE THE LOCAL CHAT MODEL

    llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0.1
    )

    # DEFINE THE SYSTEM PROMPT

    system_prompt = (
        "You are an expert financial analyst and trading advisor specializing in global ETFs, Stocks, and Mutual Funds.\n"
        "Use the following pieces of retrieved context to answer the user's question.\n"
        "Provide clear insights, budgeting advice, or risk-to-reward metrics based strictly on the document.\n\n"
        "CRITICAL CURRENCY & LOCALIZATION RULES:\n"
        "1. Identify the country or region of the asset mentioned in the query or context (e.g., Nifty, BEES, Tata = India; S&P 500, Apple = USA; Nikkei = Japan).\n"
        "2. Dynamically apply the correct local currency symbol for ALL figures, examples, and rules of thumb:\n"
        "   - India/Indian Assets: Use Indian Rupees (₹ or INR) exclusively. Never use '$' for Indian markets.\n"
        "   - USA/Global Default: Use US Dollars ($ or USD).\n"
        "   - Japan: Use Japanese Yen (¥ or JPY).\n"
        "3. If providing a general trading rule of thumb (e.g., risk-to-reward), adapt the example numbers to the target local currency natively (e.g., 'for every ₹1,000 risked, target ₹3,000 profit').\n\n"
        "CRITICAL EXECUTION RULES:\n"
        "1. Do NOT write programming code or code blocks (e.g., no Python script answers).\n"
        "2. Answer in professional, clean financial English.\n"
        "3. If you do not know the answer or if it is not in the context, say exactly:\n"
        "   'I cannot find sufficient data in the provided financial sheets.' Do not make up facts.\n\n"
        "Context:\n{context}\n"
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}")
        ]
    )

    # FOMATING THE CONTEXT HELPER FUNC

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # CONSTRUCT THE LCEL

    rag_chain = (
        {"context" : retriver | format_docs, "question" : RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print(f"Successfully loaded {CHAT_MODEL}. Ready for queries!\n")
    print("-" * 50)

    # INTERATIVE TERMINAL LOOP FOR TESTING

    while True:
        USER_QUERY = input("\n Ask your financhial sheet a question (or type 'exit'): ")
        if USER_QUERY.strip().lower() == 'exit':
            print("Shutting down Query Engine...")
            break

        if not USER_QUERY.strip():
            continue

        print("\n Analyzing documents and calculating mertices...")

        try:
            # STREAM THE RESPONSE TOKEN IN REAL TIME IN THE TERMINAL

            response = rag_chain.stream(USER_QUERY)
            print("\n[Financial Insights]", end="", flush=True)

            for chunk in response:
                print(chunk, end="", flush=True)

            print("\n" + "-" * 50)
        
        except Exception as e:
            print(f"\n An error ocurred during the chain execution: {e}")
    
if __name__ == "__main__":
    main()