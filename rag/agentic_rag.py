"""
Agentic RAG implementation for audit knowledge.

The pipeline is intentionally resilient:
- persistent JSON document store for user-added knowledge
- semantic retrieval when sentence-transformers is available
- TF-IDF fallback when embedding dependencies or model files are unavailable
- query expansion, deduplication, lightweight reranking and source attribution
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from langchain_core.documents import Document

from config import LLM_CONFIG, RAG_CONFIG

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None


logger = logging.getLogger(__name__)


AUDIT_QUERY_SYNONYMS: Dict[str, List[str]] = {
    "权限": ["访问控制", "账号授权", "职责分离", "最小权限"],
    "备份": ["恢复演练", "灾备", "RPO", "RTO"],
    "日志": ["审计轨迹", "操作留痕", "监控告警"],
    "财务": ["财务报告", "凭证", "SOX", "内部控制"],
    "数据": ["数据分类分级", "敏感数据", "加密", "脱敏"],
    "变更": ["上线审批", "回退方案", "测试验证"],
    "合规": ["控制要求", "法规", "标准", "审计证据"],
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
class StoredChunk:
    id: str
    content: str
    metadata: Dict[str, Any]


class DocumentProcessor:
    def __init__(self, chunk_size: int = RAG_CONFIG["chunk_size"], chunk_overlap: int = RAG_CONFIG["chunk_overlap"]) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, max(0, chunk_size // 2))

    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        metadata = dict(metadata or {})
        metadata.setdefault("source", "manual")
        metadata["processed_at"] = datetime.now().isoformat()
        chunks = self._split_text(text)
        documents = []
        for index, chunk in enumerate(chunks):
            doc_metadata = {**metadata, "chunk_id": index, "chunk_size": len(chunk)}
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        return documents

    def process_file(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="ignore")
        return self.process_text(
            content,
            {
                "source": str(path),
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
        if not self.chunks:
            self.add_documents(
                [
                    Document(page_content=item["text"], metadata={"source": item["source"], "type": item["type"]})
                    for item in BUILTIN_KNOWLEDGE
                ],
                persist=True,
            )

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

    def _document_id(self, document: Document) -> str:
        source = str(document.metadata.get("source", "manual"))
        raw = f"{source}\n{document.page_content}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]


class HybridRetriever:
    def __init__(self, store: PersistentDocumentStore) -> None:
        self.store = store
        self.embedding_model = self._load_embedding_model()
        self.embedding_matrix: Optional[np.ndarray] = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.rebuild()

    def _load_embedding_model(self) -> Any:
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
            logger.info("Embedding model unavailable, TF-IDF fallback enabled: %s", exc)
            return None

    def rebuild(self) -> None:
        texts = [chunk.content for chunk in self.store.chunks]
        if not texts:
            return
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                self.embedding_matrix = np.asarray(embeddings, dtype=np.float32)
            except Exception as exc:
                logger.warning("Embedding index rebuild failed: %s", exc)
                self.embedding_matrix = None

        if TfidfVectorizer:
            try:
                self.tfidf_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            except Exception as exc:
                logger.warning("TF-IDF index rebuild failed: %s", exc)
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None

    def retrieve(self, queries: List[str], k: int) -> List[Document]:
        scored: Dict[str, Dict[str, Any]] = {}
        for query in queries:
            for chunk_id, score in self._semantic_scores(query, k * 3):
                scored.setdefault(chunk_id, {"score": 0.0})
                scored[chunk_id]["score"] = max(scored[chunk_id]["score"], score * 0.65)
            for chunk_id, score in self._tfidf_scores(query, k * 3):
                scored.setdefault(chunk_id, {"score": 0.0})
                scored[chunk_id]["score"] = max(scored[chunk_id]["score"], score)

        chunks_by_id = {chunk.id: chunk for chunk in self.store.chunks}
        ranked = sorted(scored.items(), key=lambda item: item[1]["score"], reverse=True)
        documents = []
        for chunk_id, data in ranked[:k]:
            chunk = chunks_by_id[chunk_id]
            metadata = dict(chunk.metadata)
            metadata["retrieval_score"] = round(float(data["score"]), 4)
            metadata["chunk_id"] = chunk.id
            documents.append(Document(page_content=chunk.content, metadata=metadata))
        return documents

    def _semantic_scores(self, query: str, limit: int) -> List[tuple[str, float]]:
        if self.embedding_model is None or self.embedding_matrix is None:
            return []
        try:
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
            query_vec = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).ravel()
            top_indices = np.argsort(scores)[::-1][:limit]
            return [(self.store.chunks[index].id, float(scores[index])) for index in top_indices if scores[index] > 0]
        except Exception as exc:
            logger.warning("TF-IDF retrieval failed: %s", exc)
            return []


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
        queries = self.generate_queries(query, context)
        return self.retriever.retrieve(queries, k=k)

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
        if not LLM_CONFIG.get("enabled"):
            return None
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
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
        documents = self.document_processor.process_text(text, metadata)
        added = self.store.add_documents(documents)
        if added:
            self.hybrid_retriever.rebuild()
        return {"added_chunks": added, "total_chunks": len(self.store.chunks)}

    def add_file(self, file_path: str) -> Dict[str, Any]:
        documents = self.document_processor.process_file(file_path)
        added = self.store.add_documents(documents)
        if added:
            self.hybrid_retriever.rebuild()
        return {"added_chunks": added, "total_chunks": len(self.store.chunks)}

    def query(self, question: str, context: Optional[Dict[str, Any]] = None, k: int = RAG_CONFIG["top_k"]) -> Dict[str, Any]:
        documents = self.retriever.retrieve_documents(question, context, k=k)
        if not documents:
            return {
                "answer": "没有检索到足够相关的知识。建议先在知识库中补充制度、流程、审计底稿或控制要求。",
                "sources": [],
                "confidence": 0.0,
                "retrieved_docs_count": 0,
            }

        answer = self._generate_answer(question, documents)
        confidence = self._calculate_confidence(documents)
        return {
            "answer": answer,
            "sources": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "content": doc.page_content[:260],
                    "score": doc.metadata.get("retrieval_score", 0.0),
                }
                for doc in documents
            ],
            "confidence": confidence,
            "retrieved_docs_count": len(documents),
        }

    def _generate_answer(self, question: str, documents: List[Document]) -> str:
        context_text = "\n\n".join(
            f"[{index + 1}] 来源：{doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for index, doc in enumerate(documents)
        )
        if self.llm:
            from langchain_core.messages import HumanMessage

            prompt = (
                "你是审计知识库问答助手。仅基于检索上下文回答，若证据不足要说明缺口。"
                "答案需要包含直接结论、审计依据、建议动作和引用来源编号。\n\n"
                f"问题：{question}\n\n检索上下文：\n{context_text}"
            )
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as exc:
                logger.warning("RAG LLM answer failed: %s", exc)
                self.llm = None

        highlights = []
        for index, doc in enumerate(documents[:3], start=1):
            sentence = self._best_sentence(question, doc.page_content)
            highlights.append(f"{index}. {sentence}（来源：{doc.metadata.get('source', 'unknown')}）")
        return "基于知识库检索，相关依据如下：\n" + "\n".join(highlights)

    def _best_sentence(self, question: str, content: str) -> str:
        sentences = [part.strip() for part in re.split(r"[。！？!?]\s*", content) if part.strip()]
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
        confidence = min(1.0, top_score * 0.75 + coverage * 0.25)
        return round(confidence, 3)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.store.chunks),
            "store_file": str(self.store.store_file),
            "embedding_model": str(RAG_CONFIG["embedding_model"]),
            "semantic_retrieval": self.hybrid_retriever.embedding_model is not None,
            "tfidf_retrieval": self.hybrid_retriever.tfidf_vectorizer is not None,
            "chunk_size": RAG_CONFIG["chunk_size"],
            "chunk_overlap": RAG_CONFIG["chunk_overlap"],
        }
