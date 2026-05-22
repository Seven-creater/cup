# A题最终交付包说明

## 直接提交/查看
- paper/A题_LaTeX编译版.pdf：主论文 PDF，已用严格 MILP 服务器结果重新编译。
- paper/main.tex：可继续人工编辑的 LaTeX 源文件。
- paper/formal_paper.md：同内容的 Markdown 正文，便于快速改写和审阅。
- results/A题_求解结果汇总.xlsx：全部问题结果汇总工作簿。
- tables/：各问题 CSV 明细表。
- figures/：论文图表，paper/main.tex 通过 ../figures/ 引用。

## 可复现工程
- reproduce/A_solution/src/run_all.py：最终求解脚本。
- reproduce/A_solution/data/raw/：题面 PDF 和 8 个附件原始数据。
- reproduce/requirements.txt：Python 依赖。
- reproduce/SERVER_AI_INSTRUCTIONS.md：服务器复现说明。

## 本地复现命令
```bash
cd reproduce
python -m pip install -r requirements.txt
python -X utf8 A_solution/src/run_all.py
```

输出会生成在 reproduce/A_solution/outputs/。

## 编译论文
进入 paper 文件夹，运行两轮：
```bash
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

注意：LaTeX 需要支持中文的 XeLaTeX/ctex 环境。
