This document contains critical information about working with this codebase. Follow these guidelines precisely.

# Core Development Rules
Package Management
    - ONLY use uv, NEVER pip
    - Installation: uv add package
    - Running tools: uv run tool
    - Running python code : uv run file.py
    FORBIDDEN: uv pip install, python file.pe

Running code
    - To run python do uv run python 
    - Never do python3 -c, python -c etc

# Code Formatting
    Format: uv run ruff format .
    Check: uv run ruff check .
    Fix: uv run ruff check . --fix

    