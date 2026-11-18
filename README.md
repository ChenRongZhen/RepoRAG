# RepoRAG

面向开源项目文档的问答 Agent：先检索本地文档，资料不足时搜索 GitHub Issue，答案带可核对出处，没有依据就明确拒答。

用 LangGraph 编排，BGE 中文向量检索，模型走任意 OpenAI 兼容接口。**本地可跑、可评测、可离线测试。**

## 数据来源与致谢

本项目的第一版知识库是 **[CoreCoder](https://github.com/he-yufeng/CoreCoder)** 的中文文档（README_CN 与 article/*.md）。

- 原作者：**Yufeng He**（[@he-yufeng](https://github.com/he-yufeng)）
- 原项目许可：**MIT License**，版权归原作者所有
- 本仓库**不复制**这些文档，运行时从你本地克隆的 CoreCoder 读取；文档内容的著作权归原作者
- 本项目与 CoreCoder 作者无隶属关系，只是把它当作真实语料

如果你换掉 `CORECODER_DOCS_DIR`，就不再依赖 CoreCoder，可以指向任何含 `README_CN.md` 与 `article/*.md` 的目录。

## 它做什么

```text
提问
  → 向量检索 Top-K 文档
  → 相关性判定（分数分段，临界才问模型）
      ├─ 资料充足 → 带 [D1] 引用作答
      └─ 资料不足 → 搜索该仓库的 GitHub Issue
                      ├─ 有结果 → 带 [I1] 引用作答
                      └─ 无结果/失败 → 固定拒答
```

- 引用编号由代码校验，模型编造的编号会被丢掉
- GitHub 超时、限流、token 无效都映射成状态并降级提示，不中断主流程
- 不执行资料里的命令；资料用 `<doc id="D1">` 包裹并转义，降低提示注入
- 空问题、检索为空、模型异常都有确定的出口

## 实测效果

单次运行，模型 `deepseek-chat`，网关为 OpenAI 兼容地址，11 篇中文文档、229 个片段。

| 场景 | 结果 |
|---|---|
| 4 道文档题 | 4/4 作答，引用均能映射回原文 |
| 4 道文档没写的题 | 4/4 固定拒答，无编造 |
| 2 道已知 Issue 题 | 2/2 作答，其中 1 题真实检索到 Issue #59 并引用 `[I1]` |
| 模糊问题（「上下文怎么处理？」） | 给出部分答案并声明不完整 |
| 空问题 | 返回「问题不能为空。」 |

36 题离线评测（24 可答 / 8 应拒 / 4 Issue）：

| 指标 | 结果 | 目标 |
|---|---|---|
| Top-1 / Top-3 / Top-5 命中率 | 62.5% / 87.5% / 100% | Top-3 ≥ 70%、Top-5 ≥ 80% |
| 拒答精确率 | 100% | ≥ 80% |
| 误拒率 | 0% | — |
| 引用覆盖率 | 96.4%–100% | ≥ 90% |

延迟：首次冷启动曾达 101 秒；模型与索引预热后单题 9–31 秒，中位数 21 秒。拒答题同样要花一次判定模型和一次 Issue 搜索，所以并不更快。

52 个测试全部离线，不读 key、不联网。

## 安装

需要 Python 3.11+。先取语料，两个目录并排放：

```bash
git clone https://github.com/he-yufeng/CoreCoder.git
git clone <this-repo> RepoRAG
```

```bash
cd RepoRAG
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

编辑 `.env`，至少填 `OPENAI_API_KEY`。首次运行会自动下载 BGE 模型并构建索引；模型约 90 MB，索引约 3 MB。

## 配置

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | 必填。OpenAI 兼容服务的 key |
| `OPENAI_BASE_URL` | 默认 `https://api.deepseek.com`；换服务只改这里 |
| `CHAT_MODEL` | 默认 `deepseek-chat`，需支持工具调用 |
| `GITHUB_TOKEN` | 可选。只读 token，仅影响限流；无 token 仍可匿名搜索 |
| `GITHUB_REPO` | Issue 搜索范围，默认原项目仓库 |
| `CORECODER_DOCS_DIR` | 文档目录，默认 `../CoreCoder` |
| `TOP_K` / `SCORE_THRESHOLD` / `BORDERLINE_LOW` | 检索数与两段阈值，默认 3 / 0.72 / 0.45 |

阈值来自真实分数分布：可答题最低 0.499，跑题题最高 0.708，区间重叠，所以只让 0.72 以上直通、0.45 以下直拒，中间交给模型判断。换模型或换语料后必须重跑评测再调。

## 使用

```bash
python main.py "上下文压缩分哪三层？"
python main.py                     # 交互模式
python main.py --fake --no-github "上下文怎么处理？"   # 完全离线，不花钱
python main.py --json --debug "..."                  # 机器可读 + 路由轨迹
python -m streamlit run ui/streamlit_app.py          # 网页
```

`--fake` 用确定性假模型验证链路，不代表真实回答质量。

## 评测

```bash
python -m app.eval --retrieval-only   # 只跑本地检索，零模型调用
python -m app.eval --strict           # 完整评测，低于门槛退出码 1
python -m pytest -q                   # 全部离线测试
```

题集在 `tests/questions.json`。报告写到 `eval/report.json` 与 `eval/report.md`。人工抽检用 `eval/manual_rubric.md`。

## 结构

```text
app/load_docs.py      读取与切分，内容指纹
app/embedding.py      BGE 编码（文档不加前缀，查询加）
app/build_index.py    joblib 索引，原子写
app/retrieve.py       索引缓存、检索、LangChain Retriever
app/relevance.py      分数分段 + 临界结构化判定
app/prompts.py        提示词、拒答文案、上下文分隔
app/rag.py            引用解析与来源映射
app/github_issues.py  Issue 搜索与状态化降级
app/graph.py          LangGraph 节点、条件边、路由
app/eval.py           检索与回答评测
main.py / ui/         CLI 与 Streamlit
```

## 限制

- 单轮问答，不保留上下文；每次提问独立检索
- 知识范围等于你给的目录；本项目只验证过 CoreCoder 的 11 篇文档
- GitHub 对中文分词很差，检索词优先取问题里的英文词，纯中文问题常 0 命中
- Issue 只搜标题和正文，不搜评论、不分页、不用 GraphQL
- 没有权限体系、增量索引、reranker、向量数据库；适合个人与小语料
- 阈值绑模型，换模型需重标定
- 评测集自建，存在过拟合自己题集的风险

## License

代码 MIT，见 [LICENSE](LICENSE)。作为语料的 CoreCoder 文档同样为 MIT，版权归原作者 Yufeng He。
