# Cohere Retrieval

Improve search relevance and prepare text for similarity workflows with Cohere Retrieval. Its two actions rerank documents against a query or turn text into embedding vectors. Reranking accepts documents with your own IDs and returns their IDs, original positions and relevance scores, so later steps can select the most relevant material without losing its connection to the source.

The embedding action returns one floating-point vector per input text in the original order, together with the selected model, vector dimensions and provider request ID. Choose the input type for search documents, search queries, classification or clustering. The release exposes Cohere Rerank v3.5 and v5 model options, alongside Embed v4 and v5 options with configurable dimensions.

Use it to reorder candidate search results before an agent reads them, prepare text for a vector store, or feed similarity and classification steps. The extension supplies retrieval results rather than managing a knowledge-base index. Reranking accepts up to 1,000 documents; embedding accepts up to 96 texts per call. Text is sent to Cohere, where model access, quotas and billing apply.

Create a project-scoped Cohere API connection with an API key from https://dashboard.cohere.com/api-keys. Select it on the Extension tools step. Rerank accepts a query and 1–1000 unique `{id,text}` documents, returning IDs, indices and scores. Embed accepts 1–96 text strings and a text input type, returning ordered float vectors, dimensions, model and provider request ID. Choose model access and quota in Cohere. Text is sent to Cohere and calls may be billed. No automatic retries; inspect uncertain outcomes before retrying. These operations do not alter Flow Steward knowledge-base indexes.

See the in-extension About and Connection pages for setup and supported dimensions. Cohere references: https://docs.cohere.com/reference/rerank and https://docs.cohere.com/reference/embed.
