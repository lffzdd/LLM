import sys

import pytest

from ...knowledge.provider import KnowledgeUnavailable
from ...knowledge.rag_provider import RagKnowledgeProvider, knowledge_enabled
from ...tools.knowledge_tools import build_knowledge_tools


def test_constructing_provider_does_not_import_rag(monkeypatch, tmp_path):
    calls = {"n": 0}

    def boom(self):
        calls["n"] += 1
        raise AssertionError("不应在构造时导入 RAG")

    monkeypatch.setattr(RagKnowledgeProvider, "_import_rag_chain", boom)
    RagKnowledgeProvider(index_path=tmp_path / "missing.json", api_key="k")
    assert calls["n"] == 0
    assert "rag_chain" not in sys.modules


def test_missing_index_returns_actionable_failure(tmp_path):
    provider = RagKnowledgeProvider(
        index_path=tmp_path / "no-index.json", api_key="k"
    )
    tools = build_knowledge_tools(provider)
    result = tools[0].call({"query": "什么是 Transformer"}, None)
    assert result.ok is False
    assert "索引不存在" in result.err
    assert "REACT_KNOWLEDGE_INDEX" in result.err


def test_missing_api_key_returns_actionable_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    index = tmp_path / "simple_index.json"
    index.write_text("{}", encoding="utf-8")
    provider = RagKnowledgeProvider(index_path=index, api_key="")
    result = build_knowledge_tools(provider)[0].call({"query": "q"}, None)
    assert result.ok is False
    assert "SILICONFLOW_API_KEY" in result.err


def test_import_failure_returns_actionable_failure(tmp_path, monkeypatch):
    index = tmp_path / "simple_index.json"
    index.write_text("{}", encoding="utf-8")

    def boom(self):
        raise KnowledgeUnavailable(
            "无法导入 RAG 模块。确认 src/AIAgent/RAG 存在且依赖已安装，"
            "原始错误: ModuleNotFoundError: rag_chain"
        )

    monkeypatch.setattr(RagKnowledgeProvider, "_import_rag_chain", boom)
    provider = RagKnowledgeProvider(index_path=index, api_key="k")
    result = build_knowledge_tools(provider)[0].call({"query": "q"}, None)
    assert result.ok is False
    assert "无法导入 RAG 模块" in result.err


def test_init_failure_is_cached(tmp_path):
    provider = RagKnowledgeProvider(
        index_path=tmp_path / "missing.json", api_key="k"
    )
    with pytest.raises(KnowledgeUnavailable, match="索引不存在"):
        provider.search("q", 3)
    assert provider._init_attempts == 1
    with pytest.raises(KnowledgeUnavailable, match="索引不存在"):
        provider.search("q", 3)
    assert provider._init_attempts == 1


def test_knowledge_disabled_by_default(monkeypatch):
    monkeypatch.delenv("REACT_KNOWLEDGE_ENABLED", raising=False)
    assert knowledge_enabled() is False
