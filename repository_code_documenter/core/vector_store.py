from typing import List, Dict, Callable, Optional


class ChromaVectorStore:
    """Batch-based Chroma store to avoid encoding a whole repository at once."""

    def __init__(self, collection_name='repository_code', model_name='all-MiniLM-L6-v2',
                 batch_size: int = 32, progress_callback: Optional[Callable[[int, int], None]] = None):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.batch_size = max(1, batch_size)
        self.progress_callback = progress_callback
        # Keep one embedding model instance per store.
        self.model = SentenceTransformer(model_name, device='cpu')
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def build(self, nodes: Dict):
        # File-level nodes can be huge and duplicate all child function code.
        # CodeNode.text_for_embedding() supplies a compact file representation.
        items = []
        for node in nodes.values():
            doc = node.text_for_embedding()
            if not doc.strip():
                continue
            items.append((node.id, doc, {
                'file_path': node.file_path, 'name': node.name, 'kind': node.kind,
                'module': getattr(node, 'module', '') or '',
            }))

        total = len(items)
        for start in range(0, total, self.batch_size):
            batch = items[start:start + self.batch_size]
            ids = [x[0] for x in batch]
            docs = [x[1] for x in batch]
            metas = [x[2] for x in batch]
            embeddings = self.model.encode(
                docs, batch_size=min(16, len(docs)), normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True
            ).tolist()
            # Chroma limits batch sizes depending on version; keep batches modest.
            self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
            if self.progress_callback:
                self.progress_callback(min(start + len(batch), total), total)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        count = self.collection.count()
        if count == 0:
            return []
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).tolist()[0]
        result = self.collection.query(query_embeddings=[q], n_results=min(max(1, top_k), count))
        ids = result.get('ids', [[]])[0]
        docs = result.get('documents', [[]])[0]
        distances = result.get('distances', [[]])[0]
        out = []
        for i, doc in enumerate(docs):
            score = 1.0 - float(distances[i]) if i < len(distances) else 0.0
            out.append({'id': ids[i], 'document': doc, 'score': score})
        return out
