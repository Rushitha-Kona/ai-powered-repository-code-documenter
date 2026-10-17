from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class CodeNode:
    id: str
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    code: str
    module: str
    parent: str = ""
    parameters: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    globals: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)

    def text_for_embedding(self) -> str:
        # File nodes can contain thousands of lines. Their embedding should represent
        # the file, not duplicate the complete source alongside every AST child.
        code = self.code
        if self.kind == "File":
            code = code[:2500]
            if len(self.code) > 2500:
                code += "\n... [file content truncated for embedding; select a symbol for full code]"
        return (
            f"File: {self.file_path}\nType: {self.kind}\nName: {self.name}\n"
            f"Parent: {self.parent}\nImports: {', '.join(self.imports[:80])}\n"
            f"Globals: {', '.join(self.globals[:80])}\nCalls: {', '.join(self.calls[:100])}\n"
            f"Bases: {', '.join(self.bases[:40])}\nCode:\n{code}"
        )

@dataclass
class RepositoryIndex:
    nodes: Dict[str, CodeNode] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)
    reverse_edges: Dict[str, List[str]] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)

    def add_edge(self, source: str, target: str):
        if source == target:
            return
        self.edges.setdefault(source, [])
        if target not in self.edges[source]:
            self.edges[source].append(target)
        self.reverse_edges.setdefault(target, [])
        if source not in self.reverse_edges[target]:
            self.reverse_edges[target].append(source)
