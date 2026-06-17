"""Thin entry point. Delegates to the experiment driver in `src.run_experiment`.

Prefer the module commands documented in CLAUDE.md (e.g. `uv run python -m src.run_experiment`);
this file exists only so `python main.py` also works.
"""

from src.run_experiment import main

if __name__ == "__main__":
    main()
