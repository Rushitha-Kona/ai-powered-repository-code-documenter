import tempfile
from pathlib import Path
from core.ast_analyzer import RepositoryAnalyzer

def test_python_repository_analysis():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / 'a.py').write_text('''CONST = 1\ndef helper(x):\n    return x + CONST\n''')
        (p / 'b.py').write_text('''from a import helper\ndef main():\n    return helper(2)\n''')
        idx = RepositoryAnalyzer(d).analyze()
        assert any(n.name == 'helper' for n in idx.nodes.values())
        assert any(n.name == 'main' for n in idx.nodes.values())
        main = next(n for n in idx.nodes.values() if n.name == 'main')
        assert 'helper' in main.calls
