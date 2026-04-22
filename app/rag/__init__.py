"""Retrieval-augmented generation layer.

Everything in this package is optional — when OLLAMA_HOST is unset
OR the generator health-check fails, /chat transparently falls
back to returning the top TF-IDF KB answer verbatim. This means the
bot always works; RAG just makes answers read more naturally.
"""
