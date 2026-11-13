# Workflow actions

Create a project Cohere API connection with a Cohere API key, then select it in an Extension tools step. Your account needs access to the selected model; requests send text to Cohere and may be billed.

For retrieval, rerank a candidate list using the question as Query. For vector storage, generate document embeddings with search_document and question embeddings with search_query, using the same model and dimensions.

Select the named action in an Extension tools step. Map workflow values to its inputs and map the returned fields needed by later steps. Examples below contain sample data; replace artifact handles and embedding values with actual earlier-step outputs.

## Rank documents by relevance to a query

Use a Cohere connection to reorder 1–1000 documents against a question or search query. Returns document IDs, original zero-based indices and relevance scores; it does not fetch documents or change an index.

- **Question or search query**: Question used to compare the supplied documents. Provide the candidate texts separately in Documents.
- **Candidate documents (JSON array)**: Supply 1–1000 objects, each with a unique nonempty id and nonempty text. Map results by ID or original zero-based index; no document retrieval happens here.
- **Reranking model**: Optional; rerank-v3.5 is the default. The selected model must be accessible to your Cohere account.
- **Maximum documents to return**: Optional integer from 1 to the number of supplied documents. Leave blank to return all ranked documents.

Example input values:

```json
{
  "query": "What is the refund deadline?",
  "documents": [
    {
      "id": "policy_1",
      "text": "Refunds are available within 30 days."
    }
  ],
  "top_n": 1
}
```

## Generate text embeddings

Use a Cohere connection to generate one numeric vector per input text, preserving input order. Choose document or query input type for retrieval and keep model and dimensions consistent with your vector store.

- **Texts to embed (JSON array)**: Nonempty text strings in input order. Maximum 96; the output-size limit may require smaller batches at high dimensions. The returned vectors use this same order.
- **Embedding purpose**: Use search_document for stored fragments and search_query for questions. Classification and clustering are separate purposes; use one consistent model for related vectors.
- **Embedding model**: Optional; embed-v4.0 is the default. Use the same model for document and query embeddings.
- **Embedding dimensions**: Optional: 256, 512, 1024 or 1536. Blank uses the model default (v4: 1536; v5: 2048). Match the vector-store dimensions exactly; large batches can exceed the result-size limit.

Example input values:

```json
{
  "texts": [
    "Refunds are available within 30 days."
  ],
  "input_type": "search_document"
}
```
