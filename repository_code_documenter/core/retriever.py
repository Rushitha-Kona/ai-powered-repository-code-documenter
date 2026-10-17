from typing import List, Dict, Tuple
from .models import RepositoryIndex, CodeNode

class HybridRetriever:
    """Combines semantic vector retrieval with repository graph context."""
    def __init__(self, index: RepositoryIndex, vector_store):
        self.index = index
        self.vector_store = vector_store

    def retrieve(self, target_id: str, query: str, top_k: int = 6) -> Tuple[CodeNode, List[Dict]]:
        target = self.index.nodes[target_id]
        semantic = self.vector_store.search(query, top_k=max(top_k * 2, 10))
        graph_ids = set(self.index.edges.get(target_id, [])) | set(self.index.reverse_edges.get(target_id, []))
        # Include second-hop graph context, limited to keep prompts small.
        for nid in list(graph_ids):
            graph_ids.update(self.index.edges.get(nid, [])[:3])
            graph_ids.update(self.index.reverse_edges.get(nid, [])[:3])
        graph_ids.discard(target_id)

        results: Dict[str, Dict] = {}
        for item in semantic:
            results[item['id']] = {**item, 'semantic_score': item['score'], 'graph_score': 0.0}
        for nid in graph_ids:
            if nid not in self.index.nodes: continue
            results.setdefault(nid, {'id': nid, 'score': 0.0, 'document': self.index.nodes[nid].text_for_embedding()})
            results[nid]['graph_score'] = 1.0 if nid in self.index.edges.get(target_id, []) or nid in self.index.reverse_edges.get(target_id, []) else 0.5
        for item in results.values():
            item['hybrid_score'] = 0.65 * item.get('semantic_score', 0.0) + 0.35 * item.get('graph_score', 0.0)
        ranked = sorted(results.values(), key=lambda x: x['hybrid_score'], reverse=True)[:top_k]
        return target, ranked
