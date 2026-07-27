# ============================================================
# src/llm_loader.py
# Loads the LLM via Ollama (CPU-friendly, no GPU required)
# ============================================================

from langchain_community.chat_models import ChatOllama


def load_llm(model_name: str = "llama3.2", temperature: float = 0.2):
    """
    Load a local LLM via Ollama.
    Ollama must be running: start it with 'ollama serve' in a terminal.
    """
    print(f"🤖 Loading LLM via Ollama: {model_name}")
    llm = ChatOllama(model=model_name, temperature=temperature)
    print(f"✅ LLM ready: {model_name} (CPU via Ollama)")
    return llm
