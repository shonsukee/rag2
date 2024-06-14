# Automatic correction of REST API misuse using Retrieval-Augmented Generation

## About

　Retrieval-Augmented Generation (RAG) is a technique that combines text generation by Large Language Models (LLMs) with retrieval of external information to improve response accuracy. By combining the retrieval of external information, it is expected to make it easier to update the output of the LLM with the latest information, clarify the basis of the output results, and suppress the phenomenon of generating information that is not based on facts (hallucination).

## Premise

- OpenAI API (charge)
- Pinecone API Key (free)

## Package

| library                                | version |
|-------------------------------------------|------------|
| openai                                    | 1.31.2     |
| streamlit                                 | 1.35.0     |
| langchain                                 | 0.1.8      |
| llama-index                               | 0.10.43    |
| llama-index-core                          | 0.10.43    |
| llama-index-vector-stores-pinecone        | 0.1.7      |
| cassio                                    | 0.1.7      |
| pinecone-client                           | 3.2.2      |

## Usage
```
$ cd /lib
$ streamlit run chat_by_llama.py
```