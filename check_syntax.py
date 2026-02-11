"""Quick syntax check for all bezier_doge modules."""

import ast
import sys
from pathlib import Path

files = [
    "bezier_doge/__init__.py",
    "bezier_doge/__main__.py",
    "bezier_doge/data_loader.py",
    "bezier_doge/rasterizer.py",
    "bezier_doge/tiling.py",
    "bezier_doge/bezier_graph.py",
    "bezier_doge/diff_renderer.py",
    "bezier_doge/diffalign.py",
    "bezier_doge/topoadapt.py",
    "bezier_doge/doge_optimizer.py",
    "bezier_doge/run.py",
]

errors = []
for f in files:
    try:
        with open(f, encoding="utf-8") as fh:
            ast.parse(fh.read())
        print(f"  OK: {f}")
    except SyntaxError as e:
        print(f"  FAIL: {f} -> {e}")
        errors.append(f)

if errors:
    print(f"\n{len(errors)} files have syntax errors!")
    sys.exit(1)
else:
    print(f"\nAll {len(files)} files passed syntax check.")
