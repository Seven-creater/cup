# A题电氢氨园区优化求解工程

本目录包含 A 题的可复现求解脚本、结果表、图表和论文草稿生成逻辑。

## 运行方式

```powershell
& "C:\Users\29785\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -X utf8 A_solution/src/run_all.py
```

脚本会自动定位 `C:\Users\29785\Desktop\A题` 中的题面附件，并把结果写入：

- `A_solution/outputs/tables`
- `A_solution/outputs/figures`
- `A_solution/outputs/report`

如果仓库内存在 `A_solution/data/raw`，脚本会优先读取该目录中的题面附件；也可以通过环境变量 `A_PROBLEM_DIR` 指定其他数据目录。

## 口径说明

- 问题一至四完全按题面三项绿电指标公式和阈值判定。
- 24 个风光场景按每个 15 天构成 360 天代表年。
- 72 吨/日扩容只放大 ALK、PEM 和合成氨装置，不放大常规负荷和既有风光装机。
- 问题三为连续 LP 的解析贪心实现；问题四在无外部 MILP 求解器条件下使用可复现的离散网格储能调度近似，并输出约束检查。
