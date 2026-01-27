def search_qdrant(question, top_k=5):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_embedding = response.data[0].embedding

    results = qdrant.query_points(
        collection_name="documents",
        prefetch=[],
        query=query_embedding,
        limit=top_k
    )

    texts = []
    for matches in results.points:
        texts.append(matches.payload["text"])

    
    return texts