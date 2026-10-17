# AI-Powered Repository-Aware Code Documenter

A project implementation based on the RepoAgent idea, redesigned around the proposed project flow:

**Repository → AST → Dependency/Knowledge Graph → Chunking/Embeddings → Vector DB → Hybrid Retrieval → LLM → Documentation → Streamlit**

## Run

Use Python 3.10+.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload a ZIP containing a Python repository.

### Local LLM with Ollama

Install/start Ollama and pull a model that fits your machine, for example:

```bash
ollama pull qwen2.5-coder:3b
```

Then use the model name in the sidebar.

If your laptop cannot run the model, enable **Use demo/mock LLM** first. The AST, graph, embedding and retrieval stages still run, allowing you to validate the pipeline without a large model.

## Project flow

1. Repository upload
2. Python AST extraction of files/classes/functions/parameters/imports/globals/calls
3. Caller-callee/import/inheritance graph
4. Code representation and embeddings
5. Chroma vector database
6. Hybrid retrieval: semantic similarity + graph relationships
7. LLM receives target code and retrieved repository context
8. Documentation is generated in Markdown
9. Streamlit displays target, relationships, retrieved evidence and documentation

## Important scope

This version focuses on Python repositories. Java/TypeScript parsers can be added later without changing the retrieval or generation layers.

## Large-repository notes (updated build)

This build encodes repository nodes in batches on CPU instead of creating embeddings for the entire repository in one call. File-level embeddings are compact summaries (to avoid embedding the full file again when its AST symbols are also indexed), generated/build/vendor directories are excluded, and Python files larger than 2 MB are skipped by the AST scanner. The Streamlit UI displays indexing progress and narrows large symbol lists with a search field.

For best results, upload source code only. Exclude `.git`, `.venv`, `venv`, `node_modules`, datasets, model weights, generated code, `dist`, `build`, and caches. The upload limit is 200 MB compressed and the extraction limit is 1 GB uncompressed. This version is still Python-focused and the Chroma client is in-memory; re-indexing after restarting the app is expected.
