"""HuggingFace Spaces entry point.

Spaces auto-detects `app.py` at the repo root. The actual app lives in
`app/streamlit_app.py` so it's importable for tests.

Run locally:
    streamlit run app.py
"""
from app.streamlit_app import main

if __name__ == "__main__":
    main()
