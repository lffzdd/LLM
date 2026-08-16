"""把隔壁 RAG 项目的检索能力适配成 KnowledgeProvider。

硬约束：import ReActMulti 不能触发 RAG 导入、模型加载或网络请求。
RAGChain 构造和 load_index 推迟到第一次 search()。
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading

from .provider import (
    KnowledgeHit,
    KnowledgeUnavailable,
    knowledge_hit_from_search_result,
)


_RAG_DIR = Path(__file__).resolve().parents[2] / "RAG"
_DEFAULT_INDEX = _RAG_DIR / "simple_index.json"

_TRUTHY = {"1", "true", "yes", "on"}


def knowledge_enabled() -> bool:
    """默认关闭：未显式启用时 knowledge_search 不进工具集。"""
    return os.environ.get("REACT_KNOWLEDGE_ENABLED", "").strip().lower() in _TRUTHY


def knowledge_index_path() -> Path:
    override = os.environ.get("REACT_KNOWLEDGE_INDEX", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_INDEX.resolve()


def knowledge_retriever_type() -> str:
    value = os.environ.get("REACT_KNOWLEDGE_RETRIEVER", "dense").strip().lower()
    if value not in {"dense", "hybrid"}:
        return "dense"
    return value


def knowledge_use_reranker() -> bool:
    return os.environ.get("REACT_KNOWLEDGE_RERANKER", "").strip().lower() in _TRUTHY


class RagKnowledgeProvider:
    """懒加载、失败缓存的 RAG 检索适配器。"""

    def __init__(
        self,
        *,
        index_path: Path | None = None,
        retriever_type: str | None = None,
        use_reranker: bool | None = None,
        api_key: str | None = None,
        rag_dir: Path | None = None,
    ) -> None:
        self.index_path = (
            Path(index_path).expanduser().resolve()
            if index_path is not None
            else knowledge_index_path()
        )
        self.retriever_type = retriever_type or knowledge_retriever_type()
        self.use_reranker = (
            knowledge_use_reranker() if use_reranker is None else use_reranker
        )
        self._api_key = api_key
        self._rag_dir = (rag_dir or _RAG_DIR).resolve()
        self._lock = threading.RLock()
        self._chain: object | None = None
        self._init_error: str | None = None
        self._init_attempts = 0

    @classmethod
    def from_env(cls) -> RagKnowledgeProvider:
        return cls()

    def search(self, query: str, top_k: int) -> list[KnowledgeHit]:
        error = self._ensure_ready()
        if error is not None:
            raise KnowledgeUnavailable(error)
        chain = self._chain
        assert chain is not None
        try:
            raw_results = self._retrieve(chain, query, top_k)
        except KnowledgeUnavailable:
            raise
        except Exception as exc:
            # 查询期网络抖动不永久禁用；只有初始化失败才缓存。
            raise KnowledgeUnavailable(
                f"知识库检索失败: {type(exc).__name__}: {exc}"
            ) from exc
        return [knowledge_hit_from_search_result(item) for item in raw_results]

    def _ensure_ready(self) -> str | None:
        with self._lock:
            if self._init_error is not None:
                return self._init_error
            if self._chain is not None:
                return None
            self._init_attempts += 1
            try:
                self._chain = self._initialize()
            except KnowledgeUnavailable as exc:
                self._init_error = str(exc)
                return self._init_error
            except Exception as exc:
                self._init_error = (
                    f"知识库初始化失败: {type(exc).__name__}: {exc}"
                )
                return self._init_error
            return None

    def _initialize(self) -> object:
        api_key = self._resolved_api_key()
        if not api_key:
            raise KnowledgeUnavailable(
                "缺少 SILICONFLOW_API_KEY。knowledge_search 使用 SiliconFlow "
                "embedding API，请设置该环境变量后重开会话；"
                "也可以设置 LLM_API_KEY 作为后备。"
            )
        if not self.index_path.is_file():
            raise KnowledgeUnavailable(
                f"知识库索引不存在: {self.index_path}。"
                "请先在 RAG 项目中构建索引，或用 REACT_KNOWLEDGE_INDEX "
                "指向已有的 simple_index.json。"
            )
        rag_chain_cls = self._import_rag_chain()
        chain = rag_chain_cls(
            embedder_type="api",
            store_type="simple",
            retriever_type=self.retriever_type,
            use_reranker=self.use_reranker,
            query_rewrite="none",
        )
        loaded = chain.load_index(self.index_path)
        if not loaded:
            raise KnowledgeUnavailable(
                f"无法加载知识库索引: {self.index_path}。"
                "文件可能损坏或为空，请重新构建索引。"
            )
        return chain

    def _resolved_api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key.strip()
        return (
            os.environ.get("SILICONFLOW_API_KEY", "").strip()
            or os.environ.get("LLM_API_KEY", "").strip()
        )

    def _import_rag_chain(self):
        rag_dir = str(self._rag_dir)
        if rag_dir not in sys.path:
            sys.path.insert(0, rag_dir)
        try:
            from rag_chain import RAGChain
        except Exception as exc:
            raise KnowledgeUnavailable(
                "无法导入 RAG 模块。确认 src/AIAgent/RAG 存在且依赖已安装，"
                f"原始错误: {type(exc).__name__}: {exc}"
            ) from exc
        return RAGChain

    def _retrieve(self, chain: object, query: str, top_k: int) -> list[object]:
        hybrid = getattr(chain, "hybrid", None)
        if hybrid is not None:
            results = hybrid.search(query, top_k=top_k)
        else:
            dense = getattr(chain, "dense_retriever", None)
            if dense is None:
                raise KnowledgeUnavailable("RAGChain 没有可用的检索器")
            results = dense.search(query, top_k=top_k)
        reranker = getattr(chain, "reranker", None)
        if reranker is not None:
            results = reranker.rerank(query=query, results=results, top_n=top_k)
        if not isinstance(results, list):
            return []
        return results
