"""
Lightweight audit knowledge graph builder.

Neo4j is optional. When it is unavailable the builder still returns extracted
entities and relations and keeps an in-memory NetworkX graph for the current
process.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import networkx as nx

from config import NEO4J_CONFIG

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None


logger = logging.getLogger(__name__)


AUDIT_PATTERNS: Dict[str, List[str]] = {
    "AUDIT_STANDARD": [r"COBIT\s*\d*", r"ISO\s*/?IEC?\s*27001", r"ISO\s*27001", r"SOX", r"数据安全法", r"网络安全法", r"个人信息保护法"],
    "RISK_TYPE": [r"数据泄露", r"系统故障", r"合规风险", r"操作风险", r"技术风险", r"业务风险", r"财务风险", r"权限滥用"],
    "CONTROL_TYPE": [r"访问控制", r"数据加密", r"审计日志", r"备份恢复", r"变更管理", r"事件响应", r"权限管理", r"职责分离"],
    "PROCESS_TYPE": [r"用户管理", r"财务报告", r"数据备份", r"系统变更", r"安全监控", r"风险评估", r"合规检查"],
    "SYSTEM_TYPE": [r"ERP系统", r"CRM系统", r"财务系统", r"人力资源系统", r"OA系统", r"邮件系统", r"数据库系统"],
}

RELATION_PATTERNS: Dict[str, List[str]] = {
    "COVERS": [r"(.{1,40}?)(?:标准|规范|制度)?(?:覆盖|适用于|要求)(.{1,40}?)(?:。|；|;|$)"],
    "MITIGATES": [r"(.{1,40}?)(?:缓解|降低|减少|控制)(.{1,40}?)(?:。|；|;|$)"],
    "IMPLEMENTS": [r"(.{1,40}?)(?:实施|执行|建立|落地)(.{1,40}?)(?:。|；|;|$)"],
    "RELATED_TO": [r"(.{1,40}?)(?:涉及|影响|关联)(.{1,40}?)(?:。|；|;|$)"],
}


class EntityExtractor:
    def extract_entities(self, text: str, language: str = "auto") -> List[Dict[str, Any]]:
        entities = []
        for entity_type, patterns in AUDIT_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(
                        {
                            "text": match.group().strip(),
                            "label": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.9,
                        }
                    )
        return self._dedupe(entities)

    def _dedupe(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        result = []
        for entity in entities:
            key = (entity["text"], entity["label"])
            if key in seen:
                continue
            seen.add(key)
            result.append(entity)
        return result


class RelationExtractor:
    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relations = []
        for relation_type, patterns in RELATION_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    source = self._find_entity(match.group(1), entities)
                    target = self._find_entity(match.group(2), entities)
                    if source and target and source["text"] != target["text"]:
                        relations.append(
                            {
                                "source": source,
                                "target": target,
                                "relation": relation_type,
                                "confidence": 0.72,
                                "context": match.group().strip(),
                            }
                        )
        return relations

    def _find_entity(self, fragment: str, entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        fragment = fragment.strip()
        for entity in entities:
            if entity["text"] in fragment or fragment in entity["text"]:
                return entity
        return None


class KnowledgeGraphBuilder:
    def __init__(self) -> None:
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.driver = self._init_neo4j()
        self.graph = nx.DiGraph()

    def _init_neo4j(self):
        if not GraphDatabase or not NEO4J_CONFIG.get("password"):
            return None
        try:
            driver = GraphDatabase.driver(
                NEO4J_CONFIG["uri"],
                auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"]),
                connection_timeout=NEO4J_CONFIG.get("timeout", 3),
            )
            driver.verify_connectivity()
            return driver
        except Exception as exc:
            logger.info("Neo4j unavailable for graph builder: %s", exc)
            return None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def build_from_text(self, text: str, language: str = "auto") -> Dict[str, Any]:
        entities = self.entity_extractor.extract_entities(text, language)
        relations = self.relation_extractor.extract_relations(text, entities)
        self._build_networkx_graph(entities, relations)
        if self.driver:
            self._store_to_neo4j(entities, relations)
        return {
            "entities": entities,
            "relations": relations,
            "graph_stats": {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()},
            "neo4j_enabled": bool(self.driver),
        }

    def _build_networkx_graph(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> None:
        for entity in entities:
            self.graph.add_node(entity["text"], label=entity["label"], confidence=entity["confidence"])
        for relation in relations:
            self.graph.add_edge(
                relation["source"]["text"],
                relation["target"]["text"],
                relation=relation["relation"],
                confidence=relation["confidence"],
                context=relation["context"],
            )

    def _store_to_neo4j(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> None:
        try:
            with self.driver.session() as session:
                for entity in entities:
                    session.run(
                        """
                        MERGE (e:AuditEntity {text: $text})
                        SET e.label = $label,
                            e.confidence = $confidence,
                            e.updated_at = datetime()
                        """,
                        text=entity["text"],
                        label=entity["label"],
                        confidence=entity["confidence"],
                    )
                for relation in relations:
                    session.run(
                        """
                        MATCH (source:AuditEntity {text: $source_text})
                        MATCH (target:AuditEntity {text: $target_text})
                        MERGE (source)-[r:RELATES_TO {type: $relation_type}]->(target)
                        SET r.confidence = $confidence,
                            r.context = $context,
                            r.updated_at = datetime()
                        """,
                        source_text=relation["source"]["text"],
                        target_text=relation["target"]["text"],
                        relation_type=relation["relation"],
                        confidence=relation["confidence"],
                        context=relation["context"],
                    )
        except Exception as exc:
            logger.warning("Failed to persist graph to Neo4j: %s", exc)

    def build_from_documents(self, documents: List[str], language: str = "auto") -> Dict[str, Any]:
        all_entities: List[Dict[str, Any]] = []
        all_relations: List[Dict[str, Any]] = []
        for text in documents:
            result = self.build_from_text(text, language)
            all_entities.extend(result["entities"])
            all_relations.extend(result["relations"])
        return {
            "entities": all_entities,
            "relations": all_relations,
            "graph_stats": {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()},
            "neo4j_enabled": bool(self.driver),
        }

    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        if self.driver:
            with self.driver.session() as session:
                records = session.run(
                    """
                    MATCH (n:AuditEntity)-[r]->(m:AuditEntity)
                    WHERE n.text CONTAINS $query OR m.text CONTAINS $query
                    RETURN n.text AS source, r.type AS relation, m.text AS target, r.context AS context
                    LIMIT 50
                    """,
                    query=query,
                )
                return [dict(record) for record in records]

        results = []
        for source, target, data in self.graph.edges(data=True):
            if query in source or query in target or query in data.get("context", ""):
                results.append({"source": source, "relation": data.get("relation"), "target": target, "context": data.get("context")})
        return results

    def get_graph_statistics(self) -> Dict[str, Any]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "neo4j_enabled": bool(self.driver),
        }
