import ast
from pathlib import Path
from typing import Dict, List, Optional, Set
from .models import CodeNode, RepositoryIndex

IGNORED_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', '.idea', '.vscode', 'dist', 'build', '.tox', '.mypy_cache', '.pytest_cache', 'site-packages', 'htmlcov'}
MAX_PYTHON_FILE_BYTES = 2_000_000

class RepositoryAnalyzer:
    """AST-based repository analyzer for Python repositories."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.index = RepositoryIndex()
        self.symbol_to_nodes: Dict[str, List[str]] = {}
        self.module_to_file: Dict[str, str] = {}

    def analyze(self) -> RepositoryIndex:
        files = [p for p in self.root.rglob('*.py') if not any(x in IGNORED_DIRS for x in p.relative_to(self.root).parts) and p.is_file() and p.stat().st_size <= MAX_PYTHON_FILE_BYTES]
        self.index.files = [str(p.relative_to(self.root)) for p in files]
        for path in files:
            self._parse_file(path)
        self._resolve_edges()
        return self.index

    def _module_name(self, path: Path) -> str:
        rel = path.relative_to(self.root).with_suffix('')
        parts = list(rel.parts)
        if parts and parts[-1] == '__init__':
            parts.pop()
        return '.'.join(parts)

    def _parse_file(self, path: Path):
        try:
            source = path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, OSError):
            return
        module = self._module_name(path)
        self.module_to_file[module] = str(path.relative_to(self.root))
        lines = source.splitlines()
        imports = self._imports(tree)
        globals_ = self._globals(tree)

        # File-level node keeps repository context available to retrieval.
        file_id = f"file::{path.relative_to(self.root).as_posix()}"
        self.index.nodes[file_id] = CodeNode(
            id=file_id, file_path=str(path.relative_to(self.root)), name=path.name,
            kind='File', start_line=1, end_line=max(1, len(lines)), code=source,
            module=module, imports=imports, globals=globals_
        )
        self.symbol_to_nodes.setdefault(path.stem, []).append(file_id)

        class Visitor(ast.NodeVisitor):
            def __init__(inner):
                inner.stack: List[str] = []
                inner.current: Optional[CodeNode] = None

            def visit_ClassDef(inner, node):
                inner._add(node, 'Class')
                inner.stack.append(node.name)
                self.symbol_to_nodes.setdefault(node.name, []).append(inner.current.id)
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        inner.current.bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        inner.current.bases.append(self._attr_name(base))
                inner._visit_body(node)
                inner.stack.pop()

            def visit_FunctionDef(inner, node):
                inner._function(node, 'Function')

            def visit_AsyncFunctionDef(inner, node):
                inner._function(node, 'AsyncFunction')

            def _function(inner, node, kind):
                inner._add(node, kind)
                self.symbol_to_nodes.setdefault(node.name, []).append(inner.current.id)
                inner.current.parameters = [self._arg_name(a) for a in node.args.args]
                inner.current.calls = self._calls(node)
                inner._visit_body(node)

            def _visit_body(inner, node):
                old = inner.current
                for child in node.body:
                    inner.current = old
                    inner.visit(child)
                inner.current = old

            def _add(inner, node, kind):
                parent = '::'.join(inner.stack)
                prefix = str(path.relative_to(self.root)).replace('/', '::')
                name = f"{prefix}::{parent + '::' if parent else ''}{node.name}"
                node_id = f"symbol::{name}"
                code = ast.get_source_segment(source, node) or '\n'.join(lines[node.lineno-1:node.end_lineno])
                obj = CodeNode(
                    id=node_id, file_path=str(path.relative_to(self.root)), name=node.name,
                    kind=kind, start_line=node.lineno, end_line=getattr(node, 'end_lineno', node.lineno),
                    code=code, module=module, parent=parent, imports=list(imports), globals=list(globals_)
                )
                self.index.nodes[node_id] = obj
                self.index.add_edge(file_id, node_id)
                inner.current = obj

        Visitor().visit(tree)

    def _resolve_edges(self):
        # Resolve calls and imports against known symbols/modules. This is deliberately
        # conservative: unresolved external-library calls are retained in metadata but
        # do not become graph nodes.
        for node in self.index.nodes.values():
            for call in node.calls:
                short = call.split('.')[-1]
                candidates = self.symbol_to_nodes.get(short, [])
                for target in candidates[:3]:
                    self.index.add_edge(node.id, target)
            for imp in node.imports:
                module = imp.split(' as ')[0].strip()
                if module in self.module_to_file:
                    target = f"file::{self.module_to_file[module]}"
                    self.index.add_edge(node.id, target)
                else:
                    top = module.split('.')[0]
                    for known, rel in self.module_to_file.items():
                        if known == top or known.startswith(top + '.'):
                            self.index.add_edge(node.id, f"file::{rel}")
                            break
            for base in node.bases:
                for target in self.symbol_to_nodes.get(base.split('.')[-1], [])[:2]:
                    self.index.add_edge(node.id, target)

    @staticmethod
    def _imports(tree) -> List[str]:
        result = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                result.extend(a.name + (f" as {a.asname}" if a.asname else '') for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                mod = n.module or ''
                for a in n.names:
                    result.append(f"{mod}.{a.name}" if mod else a.name)
        return sorted(set(result))

    @staticmethod
    def _globals(tree) -> List[str]:
        result = []
        for n in tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Name): result.append(t.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
        return sorted(set(result))

    @staticmethod
    def _calls(node) -> List[str]:
        calls = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name): calls.append(n.func.id)
                elif isinstance(n.func, ast.Attribute): calls.append(RepositoryAnalyzer._attr_name(n.func))
        return sorted(set(calls))

    @staticmethod
    def _attr_name(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr); node = node.value
        if isinstance(node, ast.Name): parts.append(node.id)
        return '.'.join(reversed(parts))

    @staticmethod
    def _arg_name(arg):
        return arg.arg
