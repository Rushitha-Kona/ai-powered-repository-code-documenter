import os
import tempfile
import zipfile
from pathlib import Path
import streamlit as st

from core.ast_analyzer import RepositoryAnalyzer
from core.vector_store import ChromaVectorStore
from core.retriever import HybridRetriever
from core.documenter import DocumentationGenerator
from core.llm import OllamaLLM, MockLLM

st.set_page_config(page_title='AI Repository-Aware Code Documenter', layout='wide')
st.title('AI-Powered Repository-Aware Code Documenter')
st.caption('AST + Dependency/Knowledge Graph + Hybrid RAG + LLM')

if 'workspace' not in st.session_state:
    st.session_state.workspace = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retrieved' not in st.session_state:
    st.session_state.retrieved = []
if 'retrieved_for' not in st.session_state:
    st.session_state.retrieved_for = None
if 'docs' not in st.session_state:
    st.session_state.docs = ''
if 'docs_for' not in st.session_state:
    st.session_state.docs_for = None

with st.sidebar:
    st.header('1. Repository')
    uploaded = st.file_uploader('Upload a ZIP repository', type=['zip'])
    model = st.text_input('Ollama model', value=os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:3b'))
    use_mock = st.checkbox('Use demo/mock LLM', value=False)
    st.divider()
    st.header('Pipeline')
    st.write('✓ Repository analysis')
    st.write('✓ Python AST extraction')
    st.write('✓ Dependency / knowledge graph')
    st.write('✓ Embeddings + Chroma')
    st.write('✓ Hybrid graph + semantic retrieval')
    st.write('✓ LLM documentation')

if uploaded and st.button('Analyze Repository', type='primary'):
    MAX_UPLOAD_BYTES = 200 * 1024 * 1024
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error('ZIP is larger than 200 MB. Please exclude datasets, virtual environments, build outputs, and other generated files, then upload the source repository.')
        st.stop()

    tmp = tempfile.mkdtemp(prefix='repo_doc_')
    zip_path = Path(tmp) / 'repo.zip'
    zip_path.write_bytes(uploaded.getvalue())
    extract_dir = Path(tmp) / 'repo'
    extract_dir.mkdir()
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = z.infolist()
            if len(members) > 100_000:
                st.error('ZIP contains too many entries (over 100,000). Exclude generated/vendor folders and retry.')
                st.stop()
            uncompressed = sum(info.file_size for info in members)
            if uncompressed > 1_000_000_000:
                st.error('ZIP expands to more than 1 GB. Exclude datasets, virtual environments, and generated files.')
                st.stop()
            # Prevent ZIP path traversal and extract files incrementally.
            root_resolved = extract_dir.resolve()
            for info in members:
                destination = (extract_dir / info.filename).resolve()
                if destination != root_resolved and root_resolved not in destination.parents:
                    raise ValueError(f'Unsafe path in ZIP: {info.filename}')
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, destination.open('wb') as dst:
                        while True:
                            block = src.read(1024 * 1024)
                            if not block:
                                break
                            dst.write(block)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        st.error(f'Could not safely extract this ZIP: {exc}')
        st.stop()

    # Handle ZIPs that contain a single top-level directory.
    children = [p for p in extract_dir.iterdir()]
    root = children[0] if len(children) == 1 and children[0].is_dir() else extract_dir

    progress = st.progress(0, text='Scanning repository files...')
    with st.spinner('Analyzing repository with AST...'):
        index = RepositoryAnalyzer(str(root)).analyze()
    progress.progress(35, text=f'Parsed {len(index.files)} Python files; building embeddings in batches...')
    def update_embedding_progress(done, total):
        progress.progress(35 + int(65 * done / max(total, 1)), text=f'Embedding repository nodes: {done:,}/{total:,}')
    try:
        vector_store = ChromaVectorStore(collection_name='repository_code', batch_size=32, progress_callback=update_embedding_progress)
        vector_store.build(index.nodes)
    except Exception as exc:
        st.error(f'Indexing failed: {exc}')
        st.info('For large repositories, exclude datasets, .venv, node_modules, build/dist folders, and generated files before uploading.')
        st.stop()
    st.session_state.workspace = root
    st.session_state.index = index
    st.session_state.vector_store = vector_store
    st.session_state.retrieved = []
    st.session_state.retrieved_for = None
    st.session_state.docs = ''
    st.session_state.docs_for = None
    progress.progress(100, text='Repository index ready')
    st.success(f'Analyzed {len(index.files):,} Python files and {len(index.nodes):,} repository nodes.')

index = st.session_state.index
if index:
    st.sidebar.header('2. Target')
    symbol_query = st.sidebar.text_input('Find a function / class / file', placeholder='e.g. parse_config')
    options = [(nid, f"{n.file_path} :: {n.name} [{n.kind}]") for nid, n in index.nodes.items()]
    if symbol_query.strip():
        needle = symbol_query.casefold()
        options = [item for item in options if needle in item[1].casefold()]
    if not options:
        st.warning('No matching symbols. Try a shorter search term.')
        st.stop()
    # Avoid rendering a huge selectbox for repositories with thousands of symbols.
    if len(options) > 500:
        st.sidebar.info(f'{len(options):,} matches. Type a more specific name to narrow the list (showing first 500).')
        options = options[:500]
    labels = [x[1] for x in options]
    selected_label = st.sidebar.selectbox('Select function / class / file', labels)
    target_id = options[labels.index(selected_label)][0]
    target = index.nodes[target_id]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader('Target code')
        st.code(target.code, language='python')
        st.caption(f'Lines {target.start_line}-{target.end_line} · {target.file_path}')
    with c2:
        st.subheader('Repository relationships')
        callees = [index.nodes[x] for x in index.edges.get(target_id, []) if x in index.nodes]
        callers = [index.nodes[x] for x in index.reverse_edges.get(target_id, []) if x in index.nodes]
        st.write('**Callees / dependencies**')
        for n in callees[:20]: st.write(f'→ `{n.name}` ({n.file_path})')
        st.write('**Callers / dependents**')
        for n in callers[:20]: st.write(f'← `{n.name}` ({n.file_path})')
        if not callees and not callers:
            st.info('No internal graph relationships were resolved for this target.')

    st.subheader('3. Hybrid retrieval')
    query = st.text_input('Retrieval query', value=f'Explain {target.name} and its role in the repository')
    top_k = st.slider('Retrieved context', 3, 12, 6)
    def run_retrieval():
        retriever = HybridRetriever(index, st.session_state.vector_store)
        _, items = retriever.retrieve(target_id, query, top_k)
        st.session_state.retrieved = items
        st.session_state.retrieved_for = target_id

    if st.button('Retrieve Context'):
        run_retrieval()

    # Only show retrieval results that belong to the currently selected target.
    retrieved = st.session_state.retrieved if st.session_state.retrieved_for == target_id else []
    for item in retrieved:
        st.markdown(f"**{item['id']}** · hybrid `{item.get('hybrid_score', 0):.3f}` · semantic `{item.get('semantic_score', 0):.3f}` · graph `{item.get('graph_score', 0):.3f}`")
        with st.expander('View retrieved context'):
            st.code(item['document'], language='text')

    st.subheader('4. Generate documentation')
    if st.button('Generate Documentation', type='primary'):
        # Retrieval is automatic if the user has not run it for this target yet.
        if st.session_state.retrieved_for != target_id:
            with st.spinner('Retrieving repository context...'):
                run_retrieval()
        retrieved = st.session_state.retrieved
        llm = MockLLM() if use_mock else OllamaLLM(model=model)
        generator = DocumentationGenerator(llm)
        with st.spinner('Generating repository-aware documentation (slow models can take a few minutes)...'):
            try:
                st.session_state.docs = generator.generate(target, retrieved, index.files, index)
                st.session_state.docs_for = target_id
            except Exception as e:
                st.session_state.docs = ''
                st.session_state.docs_for = None
                st.error(f'Documentation generation failed: {e}')
                st.info('Check that Ollama is running and the model is pulled, or enable "Use demo/mock LLM".')

    if st.session_state.docs and st.session_state.docs_for == target_id:
        st.subheader('Generated documentation')
        st.markdown(st.session_state.docs)
        st.download_button('Download as Markdown', st.session_state.docs,
                           file_name=f'{target.name}.md', mime='text/markdown')
else:
    st.info('Upload a ZIP repository and click Analyze Repository to begin.')
