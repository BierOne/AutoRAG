supported_prompt_makers = ['fstring', 'window_replacement', 'long_context_reorder']
supported_generators = ["llama_index_llm","vllm","openai_llm","vllm_api",
                        "LlamaIndexLLM", "OpenAILLM", "Vllm", "VllmAPI"]
supported_retrievers = ["bm25", "vectordb", "hybrid_cc", "hybrid_rrf"]
supported_vectordbs = ["chroma", "couchbase", "milvus", "pinecone", "qdrant", "weaviate"]


available_model_zoo = {
    "LlamaIndexLLM":{ # TODO: available models from official package
        "openai": ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106"],
        "ollama": ["llama3"],
        "openailike": ["mistralai/Mistral-7B-Instruct-v0.2"],
    }
}

#
# import openai
#
# try:
#     client = openai.OpenAI()  # Ensure you have `OPENAI_API_KEY` set
#     models = client.models.list()
#     print("OpenAI available models:", [model.id for model in models.data])
# except Exception as e:
#     print("OpenAI model fetch failed:", e)
