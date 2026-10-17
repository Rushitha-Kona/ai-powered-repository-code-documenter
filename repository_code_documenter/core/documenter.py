from typing import Dict, List
from .models import CodeNode

# Character budgets keep the prompt inside the model's context window.
# (~4 chars per token; 8192-token window -> roughly 30k chars total, leave room for the answer.)
MAX_TARGET_CODE_CHARS = 9000
MAX_CONTEXT_ITEM_CHARS = 1800
MAX_TOTAL_CONTEXT_CHARS = 9000
MAX_FILES_LISTED = 60
MAX_LIST_ITEMS = 40


def _clip(text: str, limit: int, note: str = "... [truncated]") -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n" + note


def _join(items, limit=MAX_LIST_ITEMS) -> str:
    items = list(items)
    shown = ", ".join(items[:limit])
    if len(items) > limit:
        shown += f", ... (+{len(items) - limit} more)"
    return shown or "none"


class DocumentationGenerator:
    def __init__(self, llm):
        self.llm = llm

    def build_prompt(self, target: CodeNode, retrieved: List[Dict], project_files: List[str],
                     index=None) -> str:
        # Relationships from the dependency graph (names only, compact).
        callees, callers = [], []
        if index is not None:
            for nid in index.edges.get(target.id, []):
                n = index.nodes.get(nid)
                if n and n.kind != 'File':
                    callees.append(f"{n.name} ({n.file_path})")
            for nid in index.reverse_edges.get(target.id, []):
                n = index.nodes.get(nid)
                if n and n.kind != 'File':
                    callers.append(f"{n.name} ({n.file_path})")

        # Retrieved context: skip the target itself, clip each item, cap the total.
        context, used = [], 0
        for item in retrieved:
            if item.get('id') == target.id:
                continue
            doc = _clip(item.get('document', ''), MAX_CONTEXT_ITEM_CHARS)
            if used + len(doc) > MAX_TOTAL_CONTEXT_CHARS:
                break
            used += len(doc)
            context.append(f"[Context {len(context) + 1} | {item.get('id', '?')} | "
                           f"hybrid_score={item.get('hybrid_score', 0):.3f}]\n{doc}")
        context_text = "\n\n".join(context) if context else "(no additional context retrieved)"

        target_code = _clip(target.code, MAX_TARGET_CODE_CHARS,
                            "... [code truncated; document only what is shown]")

        # Task instructions come FIRST and the code comes LAST-ish so that, even if a
        # backend clips the prompt, the instructions are never the part that is lost.
        return f'''You are a repository-aware software documentation engineer.
Write accurate Markdown documentation for the TARGET code object below.
Do not invent behavior. Use only the target code and the retrieved repository context.
Clearly separate facts from uncertainty.

Return Markdown with these sections:
1. Purpose
2. Parameters / attributes
3. Return value
4. Internal logic
5. Dependencies and relationships (callers / callees / imports)
6. Side effects / exceptions visible from the code
7. Example usage (only if supported by the provided context)

TARGET
File: {target.file_path}
Type: {target.kind}
Name: {target.name}
Lines: {target.start_line}-{target.end_line}
Parameters: {_join(target.parameters)}
Imports: {_join(target.imports)}
Globals: {_join(target.globals)}
Calls: {_join(target.calls)}
Callees in repository: {_join(callees)}
Callers in repository: {_join(callers)}

TARGET CODE
```python
{target_code}
```

REPOSITORY FILES ({len(project_files)} total, first {min(len(project_files), MAX_FILES_LISTED)} shown):
{chr(10).join(project_files[:MAX_FILES_LISTED])}

RETRIEVED GRAPH + SEMANTIC CONTEXT
{context_text}

Now write the documentation for {target.name} in Markdown.
'''

    def generate(self, target, retrieved, project_files, index=None):
        prompt = self.build_prompt(target, retrieved, project_files, index)
        result = self.llm.generate(prompt)
        if not result or not result.strip():
            raise RuntimeError(
                "The model returned an empty response. Try a different model, or check that it "
                "is fully downloaded (`ollama list`)."
            )
        return result
