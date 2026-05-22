# Server AI Instructions

You are running on a remote GPU server. Your job is to run the existing repository exactly as provided and report results back. Do not edit source code.

## Absolute Rules

1. Do not modify any tracked source file.
2. Do not commit, push, format, refactor, or rewrite code.
3. Do not change `A_solution/src/run_all.py`.
4. You may create a conda environment, install packages, run scripts, inspect generated outputs, and create a compressed archive of outputs.
5. If something fails, report the exact command and error. Do not patch code.

## Commands To Run

```bash
set -e

WORKDIR="$HOME/cup_run"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

git clone https://github.com/Seven-creater/cup.git
cd cup

conda create -y -n cup-a python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cup-a

python -m pip install --upgrade pip
python -m pip install pandas numpy openpyxl pillow reportlab pypdf

python -X utf8 A_solution/src/run_all.py
```

## Optional LaTeX Compile

Only run this if `xelatex` exists on the server:

```bash
if command -v xelatex >/dev/null 2>&1; then
  mkdir -p A_solution/outputs/report/latex_build
  xelatex -interaction=nonstopmode -output-directory=A_solution/outputs/report/latex_build A_solution/outputs/report/main.tex
  xelatex -interaction=nonstopmode -output-directory=A_solution/outputs/report/latex_build A_solution/outputs/report/main.tex
  cp A_solution/outputs/report/latex_build/main.pdf A_solution/outputs/report/A_latex_server.pdf
fi
```

## Verification Commands

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
import pypdf

base = Path("A_solution/outputs")
required = [
    base / "A题_求解结果汇总.xlsx",
    base / "report" / "A题_电氢氨园区优化运行报告.pdf",
    base / "report" / "main.tex",
    base / "tables" / "validation_checks.csv",
]
for path in required:
    print(path, "exists=", path.exists(), "size=", path.stat().st_size if path.exists() else 0)

checks = pd.read_csv(base / "tables" / "validation_checks.csv")
print(checks.to_string(index=False))

pdf = base / "report" / "A题_电氢氨园区优化运行报告.pdf"
reader = pypdf.PdfReader(str(pdf))
print("report_pages=", len(reader.pages))

p1 = pd.read_csv(base / "tables" / "problem1_summary.csv")
p2 = pd.read_csv(base / "tables" / "problem2_typical_summary.csv")
p3 = pd.read_csv(base / "tables" / "problem3_year_summary.csv")
p4 = pd.read_csv(base / "tables" / "problem4_storage_config.csv")

print("problem1_summary:")
print(p1.to_string(index=False))
print("problem2_best_typical:")
print(p2.sort_values("comprehensive_cost_per_ton").head(1).to_string(index=False))
print("problem3_best_year:")
print(p3.sort_values("annual_avg_cost_per_ton").head(1).to_string(index=False))
print("problem4_storage_config:")
print(p4.to_string(index=False))
PY
```

## Package Results

```bash
tar -czf cup_outputs.tar.gz A_solution/outputs
sha256sum cup_outputs.tar.gz
git status --short
```

## What To Send Back

Send back exactly these items:

1. The full terminal output from the run command.
2. The full terminal output from the verification commands.
3. The `sha256sum` line for `cup_outputs.tar.gz`.
4. The output of `git status --short`.
5. Whether `A_solution/outputs/report/A_latex_server.pdf` was generated.

Remember: do not edit source code. If a command fails, stop and report the error.
