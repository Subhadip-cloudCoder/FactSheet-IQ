import os
import re
import yfinance as yf
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
CHAT_MODEL = "llama3:8b"

FALLBACK_PHRASE = "I cannat find sufficient data in the provided financhial sheets."

def get_unique_sources(db):
    """Extracts a list of unique PDF filenames currently inside the database. """
    collection = db.get()
    metadatas = collection.get("metadatas", [])
    sources = set()
    for meta in metadatas:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return list(sources)

def caluclate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) natively from the historical data"""

    if len(prices) < period + 1:
        return None
    
    deltas = prices.diff().dropna()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)

    avg_gain = gains.ewm(com=period-1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period-1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 +rs))
    return rsi.iloc[-1]

def calculated_macd(prices):
    """Calculates true MACD line, Signal Line, and Histogram using standard exchange logic"""
    if len(prices) < 35:
        return None, None, None
    
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def resolve_ticker_dynamically(query_text, llm):
    """Uses Llama3 internally to translate natural human text into exact Yahoo Finance Symbols"""

    prompt = (
        f"Extract the single financial ticker symbol for the asset requested in this question: '{query_text}'.\n"
        f"Follow these strict formatting structural rules:\n"
        f"- Indian Stocks/ETFs (e.g., Reliance, Tata Steel, Nifty Bees, Silver Bees, CPSE) -> always use uppercase and append '.NS' (e.g., RELIANCE.NS, TATASTEEL.NS, NIFTYBEES.NS, SILVERBEES.NS, CPSEETF.NS).\n"
        f"- Bank Nifty or Nifty Bank -> ^NSEBANK\n"
        f"- Nifty 50 -> ^NSEI\n"
        f"- US Stocks (e.g., Apple, Tesla) -> use standard short codes (e.g., AAPL, TSLA).\n"
        f"Return ONLY the raw ticker symbol text with nothing else. Do not use markdown blocks, punctuation, spaces, introduction sentences, or conversational remarks. If no financial asset can be identified, reply with 'NONE'."
    )

    response = llm.invoke(prompt)
    ticker = response.content.strip().split("\n")[0].replace("`", "").replace(" ", "")
    return None if "NONE" in ticker or not ticker else ticker

def fetch_realtime_data(query_text, llm):
    """
        Checks if the user explicitly wants live data/indicators. 
        If yes, dynamically extracts the ticker and fetches live exchange streams. If no, returns None to allow PDF RAG to take over.
    """

    lower_query = query_text.lower()

    # PDF OVERRIDE WORDS TO NOT SEARCH ONLINE

    pdf_override_triggers = [
        "pdf", "document", "file", "history", "feed", "dataset", "uploaded"
    ]

    if any(word in lower_query for word in pdf_override_triggers):
        return None

    # INTENT DETECTION TRIGGER WORDS

    live_triggers_indicators = [
        "rsi", "macd", "sma", "moving average", "indicator", "live", "realtime", "real-time", "current price", "today", "now",
        "highest", "lowest", "high", "low"
    ]

    if not any(trigger in lower_query for trigger in live_triggers_indicators):
        return None
    
    print(f"[Live Intent Verified] Dynamically resolving asset symbol using local intelligence...")
    selected_ticker = resolve_ticker_dynamically(query_text, llm)

    if not selected_ticker:
        return None
    
    print(f"[⚡ Live Intent Verified] Compiling live metrics for exchange code: {selected_ticker}...")

    try:
        asset = yf.Ticker(selected_ticker)
        df = asset.history(period="1y")

        if df.empty:
            return f"[Live Market Data Error] Could not retrieve data for ticker symbol {selected_ticker}"
        
        today_metrics = df.iloc[-1]
        open_price = today_metrics['Open']
        high_price = today_metrics['High']
        low_price = today_metrics['Low']
        close_price = today_metrics['Close']
        live_volume = today_metrics['Volume']

        previous_close = df['Close'].iloc[-2]
        price_change = close_price - previous_close
        pct_change = (price_change / previous_close) * 100

        highest_1y = df['High'].max()
        lowest_1y = df['Low'].min()

        sma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        sma50 = df['Close'].rolling(window=50).mean().iloc[-1]
        rsi14 = caluclate_rsi(df['Close'], period=14)
        macd_l, signal_l, hist_l = calculated_macd(df['Close'])

        live_context = (
            f"[Source: LIVE REAL-TIME EXCHANGE FEED]\n"
            f"Asset Ticker Symbol: {selected_ticker}\n"
            f"Opening Price Today: {open_price:.2f}\n"
            f"Highest Price Today: {high_price:.2f}\n"
            f"Lowest Price Today: {low_price:.2f}\n"
            f"Current Closing Price Today: {close_price:.2f}\n"
            f"Daily Absolute Change: {price_change:+.2f} ({pct_change:+.2f}%)\n"
            f"Trading Volume Today: {live_volume:,}\n"
            f"1-Year (52-Week) Highest Price Point: {highest_1y:.2f}\n"
            f"1-Year (52-Week) Lowest Price Point: {lowest_1y:.2f}\n"
            f"Technical Indicators Calculated:\n"
            f" - 20-Day Simple Moving Average (SMA_20): {sma20:.2f}\n"
            f" - 50-Day Simple Moving Average (SMA_50): {sma50:.2f}\n"
            f" - Relative Strength Index (RSI_14): {rsi14:.2f} "
            f"({'Overbought (>70)' if rsi14 > 70 else 'Oversold (<30)' if rsi14 < 30 else 'Neutral'})\n"
            f" - MACD Line (12, 26): {macd_l:.2f}\n"
            f" - Signal Line (9): {signal_l:.2f}\n"
            f" - MACD Histogram Value: {hist_l:.2f}\n"
        )
        return live_context
    except Exception as e:
        return f"[Live Market Data Error] Technical issues fetching realtime stream: {str(e)}"

def main():
    print("Initializing FactSheet-IQ Query Engine...")

    # LOADING THE EXISTING VECTOR DATABASE

    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Vector database not found at {CHROMA_PATH}")
        return

    embeddings = OllamaEmbeddings(model = EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # CONFIGURE THE DATABASE

    retriver = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 7,
            "fetch_k": 25,
            "lambda_mult" : 0.65
        }
    )

    # INITIALIZE THE LOCAL CHAT MODEL

    llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0.0,
        num_ctx=8192
    )

    # DEFINE THE SYSTEM PROMPT

    system_prompt = ("""
You are an elite institutional financial analyst and compliance officer. Your objective is to provide structured, audit-ready data analysis and reporting based strictly on the provided context. 

Context may originate from live exchange data streams [Source: LIVE REAL-TIME EXCHANGE FEED] or static document segments explicitly tagged via a [Source: filename.pdf] prefix.

CRITICAL DOCUMENT TRACKING RULES:
1. Multi-Document Clarity: Always explicitly state WHICH fund, ticker, or document you are referencing in your analysis based on the [Source] tags present in the context.
2. Source Isolation: Keep financial metrics completely separated. Never mix, average, or blend statistics (such as expense ratios, AUM, or sector allocations) between different source files.

CRITICAL CURRENCY & LOCALIZATION RULES:
1. Asset Identification: Identify the country or region of the asset mentioned in the query or context (e.g., Nifty, BEES, CPSE, Tata = India; S&P 500, Apple = USA; Nikkei = Japan).
2. Local Currency Enforcement: Dynamically apply the correct local currency symbol for ALL figures, financial calculations, and metrics:
   - India/Indian Assets: Use Indian Rupees (₹ or INR) exclusively. Never default to '$' for Indian markets.
   - USA/Global Default: Use US Dollars ($ or USD).
   - Japan: Use Japanese Yen (¥ or JPY).
3. Rule of Thumb Adaptation: If providing a general trading rule (e.g., risk-to-reward ratios or position sizing layouts), adapt the numerical examples natively to the target local currency (e.g., "for every ₹1,000 risked, target a ₹3,000 profit").

CRITICAL CHRONOLOGICAL PRIORITY & NUMERIC EXACTNESS:
1. Timeline Priority: Financial sheets frequently present data across multiple quarters or fiscal years. You must prioritize the most recent chronological data point provided in the context and explicitly state the date or period of the metric you are reporting.
2. Transcription Fidelity: Never aggressively round numbers, truncate metrics, or simplify fractional values unless explicitly commanded. Transcribe percentages, ratios, and values exactly as they are written in the context (e.g., if the sheet says 0.634%, do not report it as 0.6% or 0.63%).

CRITICAL COMPLIANCE, EXECUTION & GUARDRAIL RULES:
1. Purely Objective Analysis: You must restrict your response entirely to mathematical summaries, data trends, current asset prices, and historical or live data metrics.
2. Absolute Ban on Actionable Advice: You are strictly forbidden from giving trading suggestions, market direction advice, buy/sell/hold commands, or delivery trade parameters. Even if the user explicitly demands a "yes or no" or a "suggestion on whether to trade tomorrow," you must refuse to give a direct trade action.
3. Safe Phrasing Shift: Instead of telling the user what to do, reframe your response to analyze current trends neutrally. For example: "The metrics reflect an asset trading near its 1-year boundaries. From an analyst standpoint, trading within these zones historically implies high volatility risk, which data users should weigh against their own risk profile."
4. No Code Outputs: Do NOT write programming code or output any markdown code blocks (e.g., do not generate Python, JavaScript, or bash scripts). Respond entirely in human financial analysis.
5. Strict Grounding (Zero Tolerance for Hallucinations): Base your insights strictly on the provided context text and tables. If the exact data required to answer the question is missing or structurally insufficient to calculate a definitive answer, reply with this exact phrase and nothing else:
   "I cannot find sufficient data in the provided financial sheets."

Context:
{context}
"""
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}")
        ]
    )

    # FOMATING THE CONTEXT HELPER FUNC

    def format_docs(docs):
        formatted_chunks = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            clean_content = "\n".join([line.strip() for line in doc.page_content.split("\n") if line.strip()])
            formatted_chunks.append(f"[Source: {source}] \n {clean_content}")
        return "\n\n---\n\n".join(formatted_chunks)
    
    def hybrid_context_router(query):
        # CHECK IF THE USER IS ASKING FOR LIVE/RSI DATA
        live_context = fetch_realtime_data(query, llm)
        if live_context:
            return live_context
        
        # IF NO LIVE DATA REQUEST IS FOUND FETCH FROM THE DATA FROM PDF NATURALLY

        docs = retriver.invoke(query)

        lower_query = query.lower()

        common_words = {
            "highest", "lowest", "price", "data", "pdf", "file", "document", 
            "what", "is", "the", "of", "from", "i", "uploaded", "and", "bank", "nifty", "history",
            "want", "know", "about", "today", "tomorrow", "will", "should", "invest", "in", "it", 
            "also", "give", "me", "for", "do", "you", "trade", "buy", "sell", "any"
        }

        query_words = set(re.findall(r'\b[a-z]{3,15}', lower_query)) - common_words

        context_text = " ".join([doc.page_content.lower() for doc in docs])

        if query_words:
            has_match = any(word in context_text for word in query_words)

            if not has_match:
                print("\n[Guardrail Triggered] Chroma retrieved chunks, but they do not match the asset name requested.")
                return "I cannot find sufficient data in the provided financial sheets."
                
        return format_docs(docs)

    # CONSTRUCT THE LCEL

    rag_chain = (
        {"context" : hybrid_context_router, "question" : RunnablePassthrough()}
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

            resolved_context = hybrid_context_router(USER_QUERY)

            if resolved_context == FALLBACK_PHRASE:
                print(f"\n [Financial Insights] \n {FALLBACK_PHRASE}")
                print("\n" + "-" * 50)
                continue

            response = rag_chain.stream(USER_QUERY)
            print("\n[Financial Insights]", end="", flush=True)

            for chunk in response:
                print(chunk, end="", flush=True)

            print("\n" + "-" * 50)
        
        except Exception as e:
            print(f"\n An error ocurred during the chain execution: {e}")
    
if __name__ == "__main__":
    main()