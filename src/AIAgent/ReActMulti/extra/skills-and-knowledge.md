# Skills 与知识检索（第五阶段）

第五阶段给 Agent 两样按需能力：磁盘上的领域流程（Skill），以及隔壁 RAG
项目的只读检索（`knowledge_search`）。两者都不改已经冻结的 system prompt。

## 为什么不能改 system prompt

`Agent` 只在会话第一条消息时构造 system prompt，之后永不重建。工具清单、
记忆静态指令都冻在那里。如果把 skill 正文写进 system prompt：

- `unload_skill` 无法真正生效，模型下一轮仍能看见旧流程；
- 多轮 REPL 会把用过的流程永久堆在最前面；
- 子 Agent / 恢复会话会继承一份过期的“系统指令”。

因此 Skill 走和 `plan_reminder` 相同的范式：每轮临时 append 到 `wire_messages`
尾部，**不进 transcript**。卸载后下一轮自然消失。

## Skill 生命周期

```text
磁盘  workspace/skills/<id>/SKILL.md
        │  SkillRegistry 扫描 / 缓存 / 失效
        ▼
会话开始  若至少有一个合法 skill
        ├── 工具：list_skills / load_skill / unload_skill（写入冻结的工具清单）
        └── 清单层：每轮注入 id + description（有数量和字符上限）

load_skill
        │  激活 id 写入 SessionState.active_skill_ids
        ▼
正文层    每轮按 id 重新从磁盘读正文并临时注入
        │  不写 checkpoint 正文，不写 transcript
        ▼
unload_skill 或 新的 user turn（Agent.run）
        └── 激活集合清空；下一轮只剩清单层
```

`continue_run` 恢复的是同一个 user turn，必须保留激活集合。`run_runtime_event`
也不是新的用户目标，同样保留。

## 所有权边界

| 状态 | 所有者 | 不放在 |
|---|---|---|
| skill 文件 / 正文 | 磁盘 + `SkillRegistry`（进程级只读缓存） | Session、checkpoint、system prompt |
| 激活了哪些 skill | `SessionState.active_skill_ids` | 全局 registry、Memory |
| 计划 | `SessionState.plan_manager` | Skill |
| 跨会话事实 | Memory | Skill |

激活状态跟 `PlanManager` 一样挂在 `SessionState` 上，所以主 Agent 和每个子
Agent 天然隔离。父 Agent 加载的流程不会漏到子任务里；子任务也不该自己
`load_skill`——委派时把需要的步骤写进任务描述，子 Agent 才能保持“自包含、
可独立完成”。

Checkpoint 只存 skill id 列表。恢复时重新读磁盘：人改了 SKILL.md，恢复后
看到的是新正文，而不是崩溃前的副本。旧 checkpoint 没有该字段时当成空列表。

## knowledge_search

默认关闭。未设置 `REACT_KNOWLEDGE_ENABLED=1` 时，工具根本不进工具集，避免
每个新会话都被一个不可用的检索工具占位。

启用后走 `KnowledgeProvider` 协议。ReActMulti 的类型是 `KnowledgeHit`，不
依赖 RAG 的 `SearchResult`。`RagKnowledgeProvider` 在第一次 `search()` 时才
把 RAG 目录插入 `sys.path`、导入 `RAGChain` 并 `load_index()`。导入
ReActMulti 不会触发 RAG 导入、模型加载或网络请求。

初始化失败（缺索引、缺 `SILICONFLOW_API_KEY`、RAG 导入失败）返回
`ToolResult.fail`，说明缺什么、怎么补；同一 provider 实例会缓存失败原因，
不再反复重初始化。查询期的瞬时网络错误不缓存。

权限声明 `accesses_network`，与 `http_request` 一样走 `ask`，避免默认放行。

环境变量：

| 变量 | 含义 | 默认 |
|---|---|---|
| `REACT_KNOWLEDGE_ENABLED` | `1`/`true`/`yes`/`on` 才把工具加入工具集 | 关闭 |
| `REACT_KNOWLEDGE_INDEX` | 索引文件路径 | `src/AIAgent/RAG/simple_index.json` |
| `REACT_KNOWLEDGE_RETRIEVER` | `dense` 或 `hybrid` | `dense` |
| `REACT_KNOWLEDGE_RERANKER` | 是否启用 reranker | 关闭 |
| `SILICONFLOW_API_KEY` | embedding API；缺省时可回退 `LLM_API_KEY` | 无 |

检索结果带来源，正文包在 `<untrusted-knowledge>` 里，并有“未经验证”的警告。
`top_k` 限制 1..10，单条 content 截断 2000 字符，总输出 8000 字符。

## 子 Agent 与 durable run

- **knowledge_search 给子 Agent，不给 durable run。** 它是只读检索，没有跨会话
  副作用，子任务常常需要查资料；但无人值守运行会消耗 embedding 额度、依赖
  网络，且权限是 `ask`——fail-closed 下调用必被拒，放进工具集只会误导模型。
- **Skill 工具不给子 Agent，也不给 durable run。** 子 Agent 的契约是自包含任务；
  父 Agent 应在 spawn 描述里写清流程。durable run 把步骤写进调度 prompt，避免
  用步数去发现和加载 skill。

这两处分别写进 `_child_base_tools` 和 `_DURABLE_EXCLUDED_TOOLS`。漏一处就会
让隔离上下文或无人值守任务拿到不该有的能力。

## 有意没做的事

- **不给模型 `create_skill` / `update_skill`。** Skill 由人维护。让模型自动沉淀
  流程会和 Memory 抢职责：Memory 存“跨会话为真的事实”，Skill 存“完成某类
  任务的做法”。自动写入会把一次性对话习惯写进仓库级流程。
- **`allowed_tools` 只是提示，不动态增删工具集。** 模型看见的工具清单在会话
  开始时冻结。运行期改 executor 注册表会造成 prompt 与可执行集合不一致。
- **不把 Skills 塞进 Memory。** 召回的是事实，加载的是流程；两者的失效策略、
  注入时机和所有权都不同。
- **不用 `RAGChain.query()`。** 那条路径会再调 LLM 生成答案。这里只要检索。
- **不在会话中途把新出现的 skill 目录补进工具清单。** system prompt 已经冻结；
  启动时一个 skill 都没有，则本会话没有 skill 工具。新增文件后开新会话即可。
- **不把 skill 正文写入 checkpoint。** 见上。
