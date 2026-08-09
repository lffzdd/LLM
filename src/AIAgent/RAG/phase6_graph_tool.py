"""
Phase 6 — GraphRAG 工具适配层

职责：将 Phase 5 的 HybridGraphRAG 封装为 AgenticRAG 可注册的外部工具

为什么需要这一层？
  Phase 3 的 AgenticRAG 有一个优雅的外部工具注册机制 (register_tool)：
    - tool_spec(): 返回 OpenAI function calling 格式的工具定义
    - handler(args, verbose): 执行工具并返回文本结果

  Phase 5 的 HybridGraphRAG 有 query() 方法，但接口不匹配。
  GraphRAGTool 做的就是适配：

    HybridGraphRAG.query(question, force_route)
        ↕  适配
    AgenticRAG.register_tool(spec, handler)

  这样 Agent 在 ReAct 循环中就能自主选择：
    - search_knowledge_base → 向量检索（Phase 1+2 的混合检索）
    - search_knowledge_graph → 图检索（Phase 5 的 Local/Global/Hybrid）

  Agent 根据问题类型自己判断用哪个，不需要硬编码路由规则。

架构位置：
  ┌─────────────────────────────────────────────┐
  │        AgenticRAG (Phase 3)                 │
  │                                             │
  │  内置工具:                                   │
  │    search_knowledge_base (向量检索)           │
  │    multi_hop_search (多跳检索)                │
  │    direct_answer (直接回答)                   │
  │                                             │
  │  外部注册工具:                                │
  │    search_semantic_memory (Phase 4 语义记忆)  │
  │    search_knowledge_graph ← 本文件           │
  │                                             │
  └─────────────────────────────────────────────┘
"""

from typing import Any

from phase5_hybrid_graphrag import HybridGraphRAG


class GraphRAGTool:
    """
    将 HybridGraphRAG 适配为 AgenticRAG 的注册工具。

    使用方式：
        hybrid = HybridGraphRAG()
        hybrid.load_graph_index()

        tool = GraphRAGTool(hybrid)
        agent.register_tool(
            tool.tool_spec(),
            tool.execute,
            prompt_instruction=tool.prompt_instruction(),
        )
    """

    TOOL_NAME = "search_knowledge_graph"

    def __init__(self, hybrid_graphrag: HybridGraphRAG):
        """
        Args:
            hybrid_graphrag: 已加载索引的 HybridGraphRAG 实例
        """
        self.hybrid = hybrid_graphrag

    # ========== 工具定义 ==========

    @staticmethod
    def tool_spec() -> dict[str, Any]:
        """
        返回 OpenAI function calling 格式的工具定义。

        为什么有 search_mode 参数？
          让 Agent 可以根据问题类型选择最优检索策略：
          - auto:   交给 GraphRAG 内部 Router 自动判断（默认）
          - local:  关系追溯 — 从实体出发沿关系边走
          - global: 全局概览 — 社区摘要 Map-Reduce
          - hybrid: 向量+图双路检索

          实际测试中，Agent 通常用 auto 就行，
          但显式暴露模式让 Agent 在明确场景下可以跳过 Router 开销。
        """
        return {
            "type": "function",
            "function": {
                "name": GraphRAGTool.TOOL_NAME,
                "description": (
                    "在知识图谱中检索结构化信息。"
                    "适用于：实体关系追溯（A 和 B 有什么关系）、"
                    "多跳关联（X 的上游技术是什么）、"
                    "全局概览（这些文档的主要主题/趋势是什么）。"
                    "不适用于查找具体的文档片段或操作步骤（那些用 search_knowledge_base）。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "检索查询。应该是关于实体关系、技术关联"
                                "或全局性问题的清晰表述。"
                            ),
                        },
                        "search_mode": {
                            "type": "string",
                            "enum": ["auto", "local", "global", "hybrid"],
                            "description": (
                                "检索模式。auto=自动判断（默认），"
                                "local=实体关系追溯，"
                                "global=社区摘要全局概览，"
                                "hybrid=向量+图双路。"
                                "不确定时用 auto。"
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    @staticmethod
    def prompt_instruction() -> str:
        """
        返回注入 Agent system prompt 的工具使用说明。

        这段说明帮助 Agent 区分何时用向量检索 vs 图检索：
          - 向量检索 (search_knowledge_base)：找最匹配的文档片段
          - 图检索 (search_knowledge_graph)：追溯实体关系、全局概览
        """
        return (
            "- **search_knowledge_graph**: "
            "当问题涉及实体间的关系、技术关联、多跳推理、"
            "或需要全局概览时使用。"
            "向量检索擅长找具体文档片段，"
            "图检索擅长追溯关系和总结全局。"
            "两者互补，按需选择。"
        )

    # ========== 工具执行 ==========

    def execute(self, args: dict[str, Any], verbose: bool = True) -> str:
        """
        执行图检索并返回文本结果。

        这个方法的签名 (args, verbose) -> str 与 AgenticRAG
        的 register_tool handler 接口一致。

        Args:
            args: function calling 解析出的参数
                  {"query": "...", "search_mode": "auto"}
            verbose: 是否打印过程

        Returns:
            格式化的检索结果文本，直接作为 tool message 反馈给 LLM
        """
        query = str(args.get("query", "")).strip()
        if not query:
            return "错误：查询不能为空。"

        search_mode = str(args.get("search_mode", "auto")).strip().lower()
        valid_modes = {"auto", "local", "global", "hybrid"}
        if search_mode not in valid_modes:
            search_mode = "auto"

        if verbose:
            mode_emoji = {
                "auto": "🤖", "local": "🔗",
                "global": "🌐", "hybrid": "🔀",
            }
            print(
                f"\n  📊 search_knowledge_graph("
                f"query=\"{query}\", "
                f"mode={mode_emoji.get(search_mode, '❓')}{search_mode})"
            )

        # 执行检索
        force_route = None if search_mode == "auto" else search_mode
        try:
            answer = self.hybrid.query(
                query,
                force_route=force_route,
                verbose=verbose,
            )
        except Exception as e:
            error_msg = f"知识图谱检索失败: {type(e).__name__}: {e}"
            if verbose:
                print(f"     ⚠️ {error_msg}")
            return error_msg

        if not answer or not answer.strip():
            return "知识图谱中没有找到与该问题相关的信息。"

        # 包装结果，让 LLM 知道这是图检索结果
        return (
            f"【知识图谱检索结果 | 模式: {search_mode}】\n\n"
            f"{answer}"
        )
