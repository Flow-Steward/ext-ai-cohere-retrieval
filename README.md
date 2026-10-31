# Cohere Retrieval

Requires extension runtime contract 1.4.0 or later; compatibility is checked during installation.

Create a project-scoped Cohere API connection with an API key from https://dashboard.cohere.com/api-keys. Select it on the Extension tools step. Rerank accepts a query and 1–1000 unique `{id,text}` documents, returning IDs, indices and scores. Embed accepts 1–96 text strings and a text input type, returning ordered float vectors, dimensions, model and provider request ID. Choose model access and quota in Cohere. Text is sent to Cohere and calls may be billed. No automatic retries; inspect uncertain outcomes before retrying. These operations do not alter Flow Steward knowledge-base indexes.

See the in-extension About and Connection pages for setup and supported dimensions. Cohere references: https://docs.cohere.com/reference/rerank and https://docs.cohere.com/reference/embed.
