# Transformers >= 4.45.0 (required for ColQwen2.5)
uv add "transformers>=4.45.0"
uv add ag2[openai] 
uv add pandas 
uv add PyMuPDF 
uv add torch==2.8.0
uv add tqdm 
uv add pillow 
uv add pdfminer.six==20250506 
uv add qwen-agent[rag]


# Colpali Engine (install from source, latest version)
uv add git+https://github.com/illuin-tech/colpali
