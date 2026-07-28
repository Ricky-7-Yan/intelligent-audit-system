"""Agentic RAG implementation for audit knowledge."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import LLM_CONFIG, RAG_CONFIG
from services.llm_client import LLMClient
from services.security import current_principal, current_tenant_id

TfidfVectorizer = None
cosine_similarity = None

logger = logging.getLogger(__name__)


AUDIT_QUERY_SYNONYMS: Dict[str, List[str]] = {
    "权限": ["访问控制", "账号授权", "职责分离", "最小权限", "privileged access", "access review"],
    "访问": ["身份认证", "授权审批", "定期复核", "用户生命周期"],
    "备份": ["恢复演练", "灾备", "RPO", "RTO", "data recovery"],
    "日志": ["审计轨迹", "操作留痕", "监控告警", "audit log"],
    "财务": ["财务报告", "凭证", "SOX", "内部控制", "ITGC"],
    "数据": ["数据分类分级", "敏感数据", "加密", "脱敏", "personal information"],
    "变更": ["上线审批", "回退方案", "测试验证", "change management"],
    "合规": ["控制要求", "法规", "标准", "审计证据", "compliance"],
    "Agent": ["tool calling", "planning", "memory", "RAG", "evaluation", "MCP", "Skill"],
    "RAG": ["retrieval", "citation", "faithfulness", "answer relevance", "检索增强"],
}


BUILTIN_KNOWLEDGE = [
    {
        "source": "builtin:COBIT2019",
        "text": "COBIT 2019 关注企业 IT 治理和管理目标，强调价值交付、风险优化、资源优化、绩效度量和责任分工。审计时应将业务目标映射到治理目标，并检查流程责任、关键控制、指标和证据。",
        "type": "standard",
    },
    {
        "source": "builtin:ISO27001",
        "text": "ISO/IEC 27001 要求组织建立信息安全管理体系，围绕风险评估、控制选择、运行监控和持续改进形成闭环。常见审计证据包括资产清单、访问权限复核、风险处置计划、事件记录和管理评审。",
        "type": "standard",
    },
    {
        "source": "builtin:SOX",
        "text": "SOX 审计重点关注财务报告相关内部控制，包括职责分离、变更管理、访问控制、日志留存、接口对账和管理层复核。审计结论需要能追溯到抽样、审批和复核证据。",
        "type": "standard",
    },
    {
        "source": "builtin:DataSecurity",
        "text": "数据安全审计应检查数据分类分级、敏感数据访问授权、传输和存储加密、脱敏处理、共享审批、日志审计和应急处置机制，确保数据处理活动有制度、有记录、可追溯。",
        "type": "standard",
    },
]


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]


@dataclass
class StoredChunk:
    id: str
    content: str
    metadata: Dict[str, Any]


def _looks_corrupt(text: Any) -> bool:
    value = str(text or "")
    if "?" * 3 in value:
        return True
    return any(mark in value for mark in ["\u93c1", "\u7487", "\u20ac", "\ufffd", "\u6d93", "\u6942", "\u6d63"])


class DocumentProcessor:
    def __init__(self, chunk_size: int = RAG_CONFIG["chunk_size"], chunk_overlap: int = RAG_CONFIG["chunk_overlap"]) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, max(0, chunk_size // 2))

    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        metadata = dict(metadata or {})
        metadata.setdefault("source", "manual")
        metadata.setdefault("tenant_id", current_tenant_id())
        metadata["processed_at"] = datetime.now().isoformat()
        return [
            Document(page_content=chunk, metadata={**metadata, "chunk_id": index, "chunk_size": len(chunk)})
            for index, chunk in enumerate(self._split_text(text))
        ]

    def process_file(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="ignore")
        return self.process_text(
            content,
            {
                "source": f"upload:{path.name}",
                "file_name": path.name,
                "file_type": path.suffix.lower(),
                "file_size": path.stat().st_size,
            },
        )

    def _split_text(self, text: str) -> List[str]:
        text = re.sub(r"\r\n?", "\n", text).strip()
        if not text:
            return []
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        chunks: List[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 2 <= self.chunk_size:
                current = f"{current}\n\n{paragraph}".strip()
                continue
            if current:
                chunks.append(current)
            if len(paragraph) <= self.chunk_size:
                current = paragraph
            else:
                chunks.extend(self._window_split(paragraph))
                current = ""
        if current:
            chunks.append(current)
        return chunks

    def _window_split(self, text: str) -> List[str]:
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class PersistentDocumentStore:
    def __init__(self, store_file: Path = RAG_CONFIG["store_file"]) -> None:
        self.store_file = store_file
        self.store_file.parent.mkdir(parents=True, exist_ok=True)
        self.chunks: List[StoredChunk] = []
        self.load()
        self._ensure_seed_knowledge()

    def load(self) -> None:
        if not self.store_file.exists():
            self.chunks = []
            return
        try:
            payload = json.loads(self.store_file.read_text(encoding="utf-8"))
            self.chunks = [StoredChunk(**item) for item in payload.get("chunks", [])]
        except Exception as exc:
            logger.warning("Failed to load RAG store, starting empty: %s", exc)
            self.chunks = []

    def persist(self) -> None:
        payload = {"chunks": [asdict(chunk) for chunk in self.chunks], "updated_at": datetime.now().isoformat()}
        self.store_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_documents(self, documents: Iterable[Document], persist: bool = True) -> int:
        existing_ids = {chunk.id for chunk in self.chunks}
        added = 0
        for document in documents:
            chunk_id = self._document_id(document)
            if chunk_id in existing_ids:
                continue
            metadata = dict(document.metadata or {})
            metadata.setdefault("source", "manual")
            self.chunks.append(StoredChunk(id=chunk_id, content=document.page_content, metadata=metadata))
            existing_ids.add(chunk_id)
            added += 1
        if added and persist:
            self.persist()
        return added

    def _ensure_seed_knowledge(self) -> None:
        builtin_sources = {item["source"] for item in BUILTIN_KNOWLEDGE}
        before = len(self.chunks)
        self.chunks = [
            chunk
            for chunk in self.chunks
            if not (chunk.metadata.get("source") in builtin_sources and _looks_corrupt(chunk.content))
        ]
        seed_items = list(BUILTIN_KNOWLEDGE)
        seed_dir = self.store_file.parents[1] / "seed_knowledge"
        for path in sorted(seed_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load seed knowledge %s: %s", path, exc)
                continue
            if isinstance(payload, list):
                seed_items.extend(item for item in payload if isinstance(item, dict))

        documents = []
        for item in seed_items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": item.get("source", "seed"),
                        "type": item.get("type", "seed"),
                        "title": item.get("title", ""),
                        "tenant_id": "*",
                        "visibility": "public_seed",
                        "authority_level": item.get("authority_level", "reference"),
                        "seed": True,
                    },
                )
            )
        added = self.add_documents(documents, persist=False)
        if added or len(self.chunks) != before:
            self.persist()

    def _document_id(self, document: Document) -> str:
        metadata = document.metadata or {}
        identity = {
            "tenant_id": metadata.get("tenant_id") or current_tenant_id(),
            "project_id": metadata.get("project_id") or "",
            "source": metadata.get("source") or "manual",
            "document_version": metadata.get("document_version") or metadata.get("version") or "1",
            "page": metadata.get("page"),
            "section": metadata.get("section"),
            "chunk_id": metadata.get("chunk_id"),
            "content": document.page_content,
        }
        raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]


class HybridRetriever:
    def __init__(self, store: PersistentDocumentStore) -> None:
        self.store = store
        self.embedding_model = self._load_embedding_model()
        self.embedding_matrix = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.rebuild()

    def _load_embedding_model(self) -> Any:
        if not RAG_CONFIG.get("enable_embeddings"):
            logger.info("Embedding model disabled; TF-IDF + keyword hybrid retrieval enabled")
            return None
        model_name = str(RAG_CONFIG["embedding_model"])
        if "\\" in model_name or "/" in model_name:
            path = Path(model_name)
            if not path.exists():
                logger.info("Local embedding model not found: %s", path)
                return None
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(model_name)
        except Exception as exc:
            logger.info("Embedding model unavailable, fallback retrieval enabled: %s", exc)
            return None

    def rebuild(self) -> None:
        texts = [chunk.content for chunk in self.store.chunks]
        if not texts:
            return
        if self.embedding_model:
            try:
                import numpy as np

                embeddings = self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                self.embedding_matrix = np.asarray(embeddings, dtype=np.float32)
            except Exception as exc:
                logger.warning("Embedding index rebuild failed: %s", exc)
                self.embedding_matrix = None
        self._ensure_tfidf_dependencies()
        if TfidfVectorizer:
            try:
                self.tfidf_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            except Exception as exc:
                logger.warning("TF-IDF index rebuild failed: %s", exc)
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None

    def retrieve(self, queries: List[str], k: int, context: Optional[Dict[str, Any]] = None) -> List[Document]:
        scored: Dict[str, Dict[str, Any]] = {}
        active_channels: set[str] = set()
        for query in queries:
            channel_results = {
                "semantic": self._semantic_scores(query, k * 4),
                "tfidf": self._tfidf_scores(query, k * 4),
                "keyword": self._keyword_scores(query, k * 4),
            }
            for channel, results in channel_results.items():
                if results:
                    active_channels.add(channel)
                for rank, (chunk_id, score) in enumerate(results, start=1):
                    data = scored.setdefault(chunk_id, {"channel_scores": {}, "rrf": 0.0})
                    data["channel_scores"][channel] = max(
                        float(data["channel_scores"].get(channel, 0.0)),
                        float(score),
                    )
                    data["rrf"] += 1.0 / (60 + rank)

        chunks_by_id = {chunk.id: chunk for chunk in self.store.chunks}
        weights = {"semantic": 0.45, "tfidf": 0.35, "keyword": 0.20}
        weight_total = sum(weights[channel] for channel in active_channels) or 1.0
        query_text = " ".join(queries)
        candidates = []
        for chunk_id, data in scored.items():
            chunk = chunks_by_id.get(chunk_id)
            if not chunk or not self._metadata_allowed(chunk.metadata, context):
                continue
            channel_score = sum(
                weights[channel] * float(data["channel_scores"].get(channel, 0.0))
                for channel in active_channels
            ) / weight_total
            rerank_score = self._lexical_coverage(query_text, chunk.content)
            channel_coverage = len(data["channel_scores"]) / max(len(active_channels), 1)
            final_score = min(0.99, channel_score * 0.72 + rerank_score * 0.18 + channel_coverage * 0.10)
            data["score"] = final_score
            candidates.append((chunk_id, data))
        ranked = sorted(candidates, key=lambda item: (item[1]["score"], item[1]["rrf"]), reverse=True)
        documents = []
        for chunk_id, data in ranked[:k]:
            chunk = chunks_by_id[chunk_id]
            metadata = dict(chunk.metadata)
            metadata["retrieval_score"] = round(float(data["score"]), 4)
            metadata["retrieval_channels"] = {
                key: round(float(value), 4) for key, value in data["channel_scores"].items()
            }
            metadata["rank_fusion_score"] = round(float(data["rrf"]), 6)
            metadata["chunk_id"] = chunk.id
            documents.append(Document(page_content=chunk.content, metadata=metadata))
        return documents

    def _metadata_allowed(self, metadata: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        context = context or {}
        tenant = str(metadata.get("tenant_id") or "")
        visibility = str(metadata.get("visibility") or "").lower()
        if not tenant:
            return False
        if tenant not in {"*", current_tenant_id()}:
            return False
        if tenant == "*" and visibility != "public_seed":
            return False

        document_project = str(metadata.get("project_id") or "")
        requested_project = str(context.get("project_id") or "")
        principal_projects = set(current_principal().project_ids)
        if document_project:
            if requested_project:
                if document_project != requested_project:
                    return False
            elif "*" not in principal_projects and document_project not in principal_projects:
                return False
        elif requested_project and visibility not in {"public_seed", "tenant_shared"}:
            return False

        filters = dict(context.get("filters") or {})
        for key in ("project_id", "standard", "source_type", "effective_year"):
            if context.get(key) not in (None, ""):
                filters.setdefault(key, context[key])
        for key, expected in filters.items():
            actual = metadata.get(key)
            if actual in (None, ""):
                if key == "project_id" and visibility in {"public_seed", "tenant_shared"}:
                    continue
                return False
            allowed = expected if isinstance(expected, (list, tuple, set)) else [expected]
            if str(actual).lower() not in {str(item).lower() for item in allowed}:
                return False
        return True

    def _lexical_coverage(self, query: str, content: str) -> float:
        terms = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,6}", query.lower()))
        if not terms:
            return 0.0
        lowered = content.lower()
        return min(1.0, sum(1 for term in terms if term in lowered) / max(len(terms), 1))

    def _semantic_scores(self, query: str, limit: int) -> List[tuple[str, float]]:
        if self.embedding_model is None or self.embedding_matrix is None:
            return []
        try:
            import numpy as np

            query_vec = self.embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = self.embedding_matrix @ np.asarray(query_vec, dtype=np.float32)
            top_indices = np.argsort(scores)[::-1][:limit]
            return [(self.store.chunks[index].id, float(scores[index])) for index in top_indices if scores[index] > 0]
        except Exception as exc:
            logger.warning("Semantic retrieval failed: %s", exc)
            return []

    def _tfidf_scores(self, query: str, limit: int) -> List[tuple[str, float]]:
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None or cosine_similarity is None:
            return []
        try:
            import numpy as np

            query_vec = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).ravel()
            top_indices = np.argsort(scores)[::-1][:limit]
            return [(self.store.chunks[index].id, float(scores[index])) for index in top_indices if scores[index] > 0]
        except Exception as exc:
            logger.warning("TF-IDF retrieval failed: %s", exc)
            return []

    def _keyword_scores(self, query: str, limit: int) -> List[tuple[str, float]]:
        query_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
        if not query_terms:
            return []
        scored = []
        for chunk in self.store.chunks:
            content = chunk.content.lower()
            metadata_text = " ".join(str(value).lower() for value in chunk.metadata.values())
            hits = sum(1 for term in query_terms if term and (term in content or term in metadata_text))
            char_hits = len(set(query.lower()).intersection(set(content))) / max(len(set(query.lower())), 1)
            score = hits * 0.18 + char_hits * 0.35
            if score > 0:
                scored.append((chunk.id, min(score, 1.0)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _ensure_tfidf_dependencies(self) -> None:
        global TfidfVectorizer, cosine_similarity
        if TfidfVectorizer is not None or not RAG_CONFIG.get("enable_tfidf", True):
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity

            TfidfVectorizer = _TfidfVectorizer
            cosine_similarity = _cosine_similarity
        except Exception as exc:  # pragma: no cover
            logger.info("sklearn unavailable, keyword retrieval fallback enabled: %s", exc)


class AgenticRetriever:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def generate_queries(self, original_query: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        queries = [original_query]
        context = context or {}
        for value in [context.get("audit_item"), *(context.get("standards") or []), *(context.get("audit_types") or [])]:
            if value:
                queries.append(f"{original_query} {value}")
        for keyword, synonyms in AUDIT_QUERY_SYNONYMS.items():
            if keyword in original_query:
                queries.append(f"{original_query} {' '.join(synonyms)}")
        return self._dedupe(queries)[:6]

    def retrieve_documents(self, query: str, context: Optional[Dict[str, Any]] = None, k: int = 5) -> List[Document]:
        return self.retriever.retrieve(self.generate_queries(query, context), k=k, context=context)

    def _dedupe(self, values: Iterable[str]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            normalized = re.sub(r"\s+", " ", value).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result


class RAGPipeline:
    def __init__(self) -> None:
        self.document_processor = DocumentProcessor()
        self.store = PersistentDocumentStore()
        self.hybrid_retriever = HybridRetriever(self.store)
        self.retriever = AgenticRetriever(self.hybrid_retriever)
        self.llm = self._init_llm()

    def _init_llm(self) -> Any:
        if os.getenv("RAG_DISABLE_LLM", "1").lower() in {"1", "true", "yes"}:
            return None
        if not LLM_CONFIG.get("enabled"):
            return None
        try:
            return LLMClient(
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
                model=LLM_CONFIG["model"],
                temperature=0.1,
                max_tokens=LLM_CONFIG["max_tokens"],
            )
        except Exception as exc:
            logger.info("RAG LLM unavailable, extractive answers enabled: %s", exc)
            return None

    def add_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        scoped_metadata = dict(metadata or {})
        scoped_metadata["tenant_id"] = current_tenant_id()
        scoped_metadata.setdefault(
            "visibility",
            "project" if scoped_metadata.get("project_id") else "tenant_shared",
        )
        scoped_metadata.setdefault("document_version", "1")
        documents = self.document_processor.process_text(text, scoped_metadata)
        added = self.store.add_documents(documents)
        if added:
            self.hybrid_retriever.rebuild()
        return {"added_chunks": added, "total_chunks": len(self.store.chunks)}

    def add_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        documents = self.document_processor.process_file(file_path)
        for document in documents:
            document.metadata.update(metadata or {})
            document.metadata["tenant_id"] = current_tenant_id()
            document.metadata.setdefault(
                "visibility",
                "project" if document.metadata.get("project_id") else "tenant_shared",
            )
            document.metadata.setdefault("document_version", "1")
        added = self.store.add_documents(documents)
        if added:
            self.hybrid_retriever.rebuild()
        return {"added_chunks": added, "total_chunks": len(self.store.chunks)}

    def query(self, question: str, context: Optional[Dict[str, Any]] = None, k: int = RAG_CONFIG["top_k"]) -> Dict[str, Any]:
        scoped_context = dict(context or {})
        scoped_context["tenant_id"] = current_tenant_id()
        documents = self.retriever.retrieve_documents(question, scoped_context, k=k)
        if not documents:
            return {
                "answer": "没有检索到足够相关的知识。建议先补充制度、流程、底稿或控制要求。",
                "sources": [],
                "confidence": 0.0,
                "retrieval_confidence": 0.0,
                "confidence_semantics": "retrieval_support_not_answer_probability",
                "conflicts": [],
                "retrieved_docs_count": 0,
            }
        answer = self._generate_answer(question, documents)
        confidence = self._calculate_confidence(documents)
        conflicts = self._detect_conflicts(documents)
        return {
            "answer": answer,
            "sources": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "content": doc.page_content[:260],
                    "score": doc.metadata.get("retrieval_score", 0.0),
                    "channels": doc.metadata.get("retrieval_channels", {}),
                    "page": doc.metadata.get("page"),
                    "section": doc.metadata.get("section"),
                    "authority_level": doc.metadata.get("authority_level"),
                }
                for doc in documents
            ],
            "confidence": confidence,
            "retrieval_confidence": confidence,
            "confidence_semantics": "retrieval_support_not_answer_probability",
            "conflicts": conflicts,
            "requires_human_review": bool(conflicts) or confidence < 0.55,
            "retrieved_docs_count": len(documents),
        }

    def _detect_conflicts(self, documents: List[Document]) -> List[Dict[str, Any]]:
        conflict_pairs = [
            (("必须", "应当", "required"), ("无需", "不需要", "optional")),
            (("允许", "可访问", "permit"), ("禁止", "不得", "deny")),
            (("保留", "留存"), ("删除", "销毁")),
        ]
        conflicts = []
        for positive, negative in conflict_pairs:
            positive_sources = [
                doc for doc in documents if any(term in doc.page_content.lower() for term in positive)
            ]
            negative_sources = [
                doc for doc in documents if any(term in doc.page_content.lower() for term in negative)
            ]
            if positive_sources and negative_sources:
                conflicts.append(
                    {
                        "topic": f"{positive[0]} / {negative[0]}",
                        "sources": sorted(
                            {
                                str(doc.metadata.get("source") or "unknown")
                                for doc in [*positive_sources, *negative_sources]
                            }
                        ),
                        "resolution": "存在方向相反的证据表述，需要核对版本、生效日期与适用范围。",
                    }
                )
        return conflicts

    def _generate_answer(self, question: str, documents: List[Document]) -> str:
        context_text = "\n\n".join(f"[{index + 1}] 来源：{doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for index, doc in enumerate(documents))
        if self.llm:
            prompt = (
                "你是审计知识库问答助手。仅基于检索上下文回答；若证据不足，要说明缺口。"
                "答案需要包含直接结论、审计依据、建议动作和引用来源编号。\n\n"
                f"问题：{question}\n\n检索上下文：\n{context_text}"
            )
            try:
                return self.llm.complete(
                    system="你是审计知识库问答助手，仅根据给定检索上下文回答。",
                    user=prompt,
                )
            except Exception as exc:
                logger.warning("RAG LLM answer failed: %s", exc)
                self.llm = None
        highlights = []
        for index, doc in enumerate(documents[:3], start=1):
            sentence = self._best_sentence(question, doc.page_content)
            highlights.append(f"{index}. {sentence}（来源：{doc.metadata.get('source', 'unknown')}）")
        return "基于知识库检索，相关依据如下：\n" + "\n".join(highlights)

    def _best_sentence(self, question: str, content: str) -> str:
        sentences = [part.strip() for part in re.split(r"[。！？；;.!?]\s*", content) if part.strip()]
        if not sentences:
            return content[:220]
        query_chars = set(question)
        return max(sentences, key=lambda sentence: len(query_chars.intersection(set(sentence))))[:220]

    def _calculate_confidence(self, documents: List[Document]) -> float:
        scores = [float(doc.metadata.get("retrieval_score", 0.0)) for doc in documents]
        if not scores:
            return 0.0
        top_score = max(scores)
        coverage = min(len(documents) / max(1, RAG_CONFIG["top_k"]), 1.0)
        return round(min(1.0, top_score * 0.75 + coverage * 0.25), 3)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.store.chunks),
            "store_file": str(self.store.store_file),
            "embedding_model": str(RAG_CONFIG["embedding_model"]),
            "semantic_retrieval": self.hybrid_retriever.embedding_model is not None,
            "tfidf_retrieval": self.hybrid_retriever.tfidf_vectorizer is not None,
            "keyword_retrieval": True,
            "rank_fusion": True,
            "metadata_filtering": True,
            "conflict_detection": True,
            "confidence_semantics": "retrieval_support_not_answer_probability",
            "chunk_size": RAG_CONFIG["chunk_size"],
            "chunk_overlap": RAG_CONFIG["chunk_overlap"],
        }
