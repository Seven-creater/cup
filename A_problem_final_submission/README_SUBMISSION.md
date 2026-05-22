# A题最终交付包说明

## 直接提交/查看
- paper/A题_LaTeX编译版.pdf：按 2025 年 LaTeX 模板格式编译的主论文 PDF。
- paper/main.tex：可继续人工编辑的 LaTeX 源文件，已使用 gmcmthesis 模板类。
- paper/gmcmthesis.cls：比赛模板类文件。
- paper/figures/logo2025.png、paper/figures/title2025.pdf：模板封面资源。
- paper/formal_paper.md：Markdown 正文，便于快速审阅和改写。
- results/A题_求解结果汇总.xlsx：全部问题结果汇总工作簿。
- tables/：各问题 CSV 明细表。
- figures/：论文正文图表，paper/main.tex 通过 ../figures/ 引用。

## 可复现工程
- reproduce/A_solution/src/run_all.py：最终求解脚本。
- reproduce/A_solution/data/raw/：题面 PDF 和 8 个原始附件。
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
注意：LaTeX 需要支持中文的 XeLaTeX/ctex 环境；队号、学校和队员姓名在 main.tex 顶部仍为“待填写”，提交前按比赛要求修改。
