# Phase 6: 综合系统 — 把所有能力整合成一个完整的 Agent

## 回顾：我们走过的路

经过 Phase 1-5 的学习，你已经掌握了构建一个完整 AI Agent 的所有核心能力：

| Phase | 能力 | 核心学习 |
|:------|:-----|:---------|
| 1 | 经典 RAG | 文档 → 分块 → Embedding → 向量检索 → 生成 |
| 2 | 进阶 RAG | 混合检索 + Reranking + 查询改写 + RAGAS 评估 |
| 3 | Agentic RAG | Agent 自主控制检索 + Self-RAG 质量评估 + 多跳检索 |
| 4 | 记忆系统 | 短期记忆 + 结构化工作记忆 + 情景记忆 + 语义记忆 |
| 5 | GraphRAG | 知识图谱 + 社区检测 + Local/Global Search |

现在的问题是：**这些能力分散在不同的 `phase_main.py` 中，无法协同工作。**

Phase 6 的目标就是把它们整合成一个统一入口，让 Agent 同时拥有所有能力，自主决策何时使用哪种能力。

---

## 为什么需要综合系统？

### 单一能力的局限性

每种检索方式都有盲区：

```
向量检索 (Phase 1+2):
  ✅ "BGE-M3 怎么配置？"           ← 精确的文档片段匹配
  ❌ "A 和 B 有什么关系？"          ← 关系推理力弱
  ❌ "这些文档的主要趋势？"          ← 只返回局部 top-k

图检索 (Phase 5):
  ✅ "Transformer 和 BERT 的关系？" ← 沿关系边精准追溯
  ✅ "核心技术主题有哪些？"          ← 社区摘要全局覆盖
  ❌ "注意力机制的计算步骤？"        ← 图里没有文档原文细节

无记忆的 Agent:
  ✅ 当前问题回答得好
  ❌ 不记得你是谁
  ❌ 不从历史经验中学习
  ❌ 重复犯同样的错误
```

### 综合系统的价值

综合系统不是简单的 1+1=2，而是 **让每种能力互补**：

- Agent 自主判断用向量检索还是图检索（而不是硬编码路由）
- 情景记忆让 Agent 从历史失败中学习，下次遇到类似问题换策略
- 语义记忆让 Agent 记住用户偏好，个性化回答
- 结构化工作记忆在对话中自动提取用户信息，无感持久化

---

## 系统架构

### 整体视图

```
┌──────────────────────────────────────────────────────────────┐
│                 Phase 6: 综合 AI 助手                        │
│                                                              │
│  用户问题                                                     │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────────────────────────┐                    │
│  │ 预处理层                              │                    │
│  │   语义记忆预召回 → 注入已知用户事实     │                    │
│  │   情景记忆预召回 → 注入相似历史经验     │                    │
│  └──────────────┬───────────────────────┘                    │
│                 │                                            │
│                 ▼                                            │
│  ┌──────────────────────────────────────┐                    │
│  │ Agent ReAct 循环 (Phase 3)            │                    │
│  │                                      │                    │
│  │  LLM 思考 → 选择工具 → 执行 → 反馈   │                    │
│  │       ↓                              │                    │
│  │  ┌─────────────────────────────────┐ │                    │
│  │  │ 可用工具池                       │ │                    │
│  │  │                                 │ │                    │
│  │  │ 📄 search_knowledge_base        │ │  ← Phase 1+2      │
│  │  │    向量混合检索 + Reranking      │ │                    │
│  │  │    + Self-RAG 评估 + CRAG 重试  │ │  ← Phase 3        │
│  │  │                                 │ │                    │
│  │  │ 📊 search_knowledge_graph       │ │  ← Phase 5        │
│  │  │    Local / Global / Hybrid 模式  │ │                    │
│  │  │                                 │ │                    │
│  │  │ 🧠 search_semantic_memory       │ │  ← Phase 4        │
│  │  │    长期事实二次检索              │ │                    │
│  │  │                                 │ │                    │
│  │  │ 🕵️ multi_hop_search             │ │  ← Phase 3        │
│  │  │    复杂多跳推理                  │ │                    │
│  │  │                                 │ │                    │
│  │  │ 💡 direct_answer                │ │  ← Phase 3        │
│  │  │    常识问题直接回答              │ │                    │
│  │  └─────────────────────────────────┘ │                    │
│  └──────────────┬───────────────────────┘                    │
│                 │                                            │
│                 ▼                                            │
│  ┌──────────────────────────────────────┐                    │
│  │ 后处理层                              │                    │
│  │   短期记忆更新 → 摘要缓冲管理          │  ← Phase 4        │
│  │   结构化抽取 → 自动提取用户信息        │  ← Phase 4        │
│  │   语义记忆写入 → 持久化新事实          │  ← Phase 4        │
│  │   情景记忆记录 → 保存任务经验          │  ← Phase 4        │
│  └──────────────────────────────────────┘                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 关键设计决策

#### 1. GraphRAG 作为「工具」而非「替换」

Phase 5 的 `HybridGraphRAG` 有自己的 Router（判断问题类型），但在综合系统中，我们 **不** 用 Phase 5 的 Router 来决定用不用向量检索。而是把 GraphRAG 封装成 Agent 的一个工具，让 Agent 自己决定：

```
❌ 方案 A（不好）: Phase 5 Router 决定一切
   问题 → Router → 向量 or 图 → 回答
   问题: Router 是规则型的，不能考虑上下文、记忆、工具调用历史

✅ 方案 B（实际采用）: Agent 自主选择
   问题 → Agent 思考 → 自主选择工具 → 执行 → 评估 → 回答
   优势: Agent 能结合上下文、记忆、之前的失败经验来选择最优策略
```

为什么这样更好？因为 Agent 有 **上下文感知能力**：

- 如果之前向量检索没找到好结果，Agent 可以换成图检索
- 如果问题既涉及具体细节又涉及关系，Agent 可以先搜图再搜向量
- 如果语义记忆里已有相关事实，Agent 可以直接用，不需要检索

#### 2. 记忆系统的分层注入

记忆不是一股脑塞进 prompt 的，而是分层注入：

```
System Prompt (固定)
   │
   ├── 语义记忆 (自动预召回，放在历史后面)
   │    "用户叫小林，喜欢蓝色，住在北京"
   │    → 保护 Prompt 前缀缓存，不打断上下文
   │
   ├── 情景记忆 (自动预召回，放在系统消息中)
   │    "之前类似问题用 Local Search 效果更好"
   │    → 让 Agent 从经验中学习
   │
   ├── 短期记忆 (对话历史，摘要 + 近期原文)
   │    → 保持对话连贯性
   │
   └── 当前问题
```

这个分层设计来自 Phase 4 的 `SemanticAgent` 和 `EpisodicAgent` 装饰器——它们在调用 Agent 前自动注入相关记忆，Agent 本身不需要知道记忆系统的存在。

#### 3. 向量检索 vs 图检索的分工

Agent 在 system prompt 中被告知两者的区别：

| 场景 | 最佳工具 | 原因 |
|:-----|:---------|:-----|
| "BGE-M3 怎么配置？" | `search_knowledge_base` | 需要文档原文细节 |
| "Transformer 和 BERT 的关系" | `search_knowledge_graph` (local) | 需要追溯实体关系 |
| "核心技术主题有哪些？" | `search_knowledge_graph` (global) | 需要全局概览 |
| "RAG 和 Agent 的关系以及各自细节" | 两个都用 | 先图检索理清关系，再向量检索补充细节 |
| "1+1=?" | `direct_answer` | 常识，不需要检索 |

---

## 端到端示例

### 示例 1：关系推理 + 事实细节

```
用户: Transformer 和 BERT 有什么关系？它们的核心区别是什么？

Agent 思考:
  这个问题有两部分：
  1. "关系" — 需要知识图谱追溯
  2. "核心区别" — 需要文档细节

  → 先调用 search_knowledge_graph(query="Transformer 和 BERT 的关系", mode="local")
  → 得到关系链: Transformer → 基础架构 → BERT; BERT 只用了 Encoder 部分
  → 再调用 search_knowledge_base(query="Transformer 和 BERT 的核心区别")
  → 得到具体技术细节

Agent 回答:
  "BERT 基于 Transformer 的 Encoder 部分构建..."（综合两路信息）
```

### 示例 2：记忆 + 检索协作

```
# 第 1 轮
用户: 我叫小林，我在做一个知识管理项目

Agent: 好的小林！知识管理项目...
→ 结构化记忆自动抽取: {user.name: "小林", user.project: "知识管理"}
→ 语义记忆持久化: fact.user_name = "小林"

# 第 2 轮（可能是另一天）
用户: 对了，你还记得我在做什么吗？

→ 语义记忆预召回: user_name=小林, user_project=知识管理
Agent: 记得，小林！你在做一个知识管理项目。
```

### 示例 3：情景记忆驱动策略选择

```
# 经验库中有一条记录:
  task: "查找 Transformer 和 GPT 的关系"
  outcome: partial
  reflection: "向量检索只返回了文档片段，缺少关系上下文。
               如果用 GraphRAG 的 Local Search 效果会更好。"

# 新查询
用户: Transformer 和 BERT 有什么联系？

→ 情景记忆预召回: 找到上面的相似经验
→ Agent 在 system prompt 中看到这条经验
→ 直接选择 search_knowledge_graph 而不是 search_knowledge_base
→ 效果更好！
```

---

## 运行方式

### 快速启动（所有功能默认开启）

```bash
uv run python phase6_main.py
```

### 自定义配置

```bash
# 不使用图检索（加载更快）
uv run python phase6_main.py --no-graph

# 自定义记忆参数
uv run python phase6_main.py \
  --token-budget 2000 \
  --summary-token-budget 600

# 最小配置（只有向量检索 + 短期记忆）
uv run python phase6_main.py \
  --no-graph \
  --no-episodic \
  --no-semantic \
  --no-structured \
  --strategy turns
```

### 前置要求

1. **环境变量**：
   - `LLM_API_KEY`（DeepSeek API Key）
   - `SILICONFLOW_API_KEY`（SiliconFlow API Key，用于 Embedding）

2. **向量索引**：首次运行时自动构建（或加载 Phase 3 已构建的缓存）

3. **GraphRAG 索引**：需要预先通过 Phase 5 构建：
   ```bash
   uv run python phase5_main.py build
   ```
   或使用 `--rebuild-graph` 选项在 Phase 6 中构建。

---

## 与前面 Phase 的关系

Phase 6 **不是重写**，而是 **整合**。代码层面：

```
phase6_graph_tool.py   — 新增：GraphRAG 工具适配层（~180 行）
phase6_main.py         — 新增：统一入口（~500 行）

复用的已有模块（零修改）：
  phase3_agentic_rag.py   → Agent 核心 + ReAct 循环 + 工具执行
  phase3_main.py          → ensure_index() 函数
  phase4_episodic_memory.py → EpisodicAgent + EpisodicMemory
  phase4_semantic_memory.py → SemanticAgent + SemanticMemory
  phase4_structured_memory.py → StructuredWorkingMemory
  phase4_summary_memory.py → SummaryBufferMemory
  phase4_token_memory.py  → TokenBudgetMemory + Tokenizer
  phase5_hybrid_graphrag.py → HybridGraphRAG
  phase5_local_search.py  → LocalSearch
  phase5_global_search.py → GlobalSearch
  phase5_community.py     → CommunityDetector
  phase5_knowledge_graph.py → KnowledgeGraph
```

Phase 3 的 `AgenticRAG.register_tool()` 机制是整合的关键——它允许我们把任何新能力（GraphRAG、语义记忆等）作为工具注册进去，而不需要修改 Agent 核心代码。这就是 **可扩展架构** 的价值。

---

## 总结：从零到一的完整路径

```
Phase 1: 你学会了让 LLM "看到" 外部知识
Phase 2: 你学会了让检索结果更精准
Phase 3: 你学会了让 Agent 自主控制检索
Phase 4: 你学会了让 Agent 拥有记忆
Phase 5: 你学会了用知识图谱做关系推理
Phase 6: 你把所有能力整合成了一个完整的 AI Agent

→ 这就是一个生产级 AI 助手的核心架构。
```

接下来可以探索的方向：
- **多 Agent 协作**：多个 Agent 分工合作（规划者 + 执行者 + 审核者）
- **实时数据源**：接入 API、数据库、实时搜索引擎
- **用户界面**：Web UI、API 服务化
- **评估体系**：端到端的自动化评估管线
- **安全治理**：访问控制、输入过滤、输出审核
