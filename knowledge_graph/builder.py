"""
知识图谱构建模块
Knowledge Graph Construction Module
"""

import spacy
import stanza
import networkx as nx
from neo4j import GraphDatabase
from typing import List, Dict, Tuple, Any
import json
import logging
from pathlib import Path
import jieba
import pkuseg
from config import NEO4J_CONFIG, PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """实体抽取器"""

    def __init__(self):
        # 初始化spaCy模型（英文）
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy英文模型未安装，请运行: python -m spacy download en_core_web_sm")
            self.nlp_en = None

        # 初始化Stanza模型（中文）
        try:
            stanza.download('zh', package='default', processors='tokenize,pos,lemma,ner')
            self.nlp_zh = stanza.Pipeline('zh', processors='tokenize,pos,lemma,ner')
        except Exception as e:
            logger.warning(f"Stanza中文模型初始化失败: {e}")
            self.nlp_zh = None

        # 初始化jieba分词
        jieba.initialize()

        # 初始化pkuseg（北大分词）
        try:
            self.pku_seg = pkuseg.pkuseg()
        except Exception as e:
            logger.warning(f"pkuseg初始化失败: {e}")
            self.pku_seg = None

    def extract_entities_en(self, text: str) -> List[Dict[str, Any]]:
        """英文实体抽取"""
        if not self.nlp_en:
            return []

        doc = self.nlp_en(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0
            })

        return entities

    def extract_entities_zh(self, text: str) -> List[Dict[str, Any]]:
        """中文实体抽取"""
        if not self.nlp_zh:
            return []

        doc = self.nlp_zh(text)
        entities = []

        for sent in doc.sentences:
            for ent in sent.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.type,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0
                })

        return entities

    def extract_custom_entities(self, text: str) -> List[Dict[str, Any]]:
        """自定义实体抽取（审计领域特定）"""
        entities = []

        # 审计相关实体模式
        audit_patterns = {
            'AUDIT_STANDARD': [
                r'COBIT\s*\d*', r'ISO\s*27001', r'SOX', r'GDPR',
                r'数据安全法', r'网络安全法', r'个人信息保护法'
            ],
            'RISK_TYPE': [
                r'数据泄露', r'系统故障', r'合规风险', r'操作风险',
                r'技术风险', r'业务风险', r'财务风险'
            ],
            'CONTROL_TYPE': [
                r'访问控制', r'数据加密', r'审计日志', r'备份恢复',
                r'变更管理', r'事件响应', r'权限管理'
            ],
            'PROCESS_TYPE': [
                r'用户管理', r'财务报告', r'数据备份', r'系统变更',
                r'安全监控', r'风险评估', r'合规检查'
            ],
            'SYSTEM_TYPE': [
                r'ERP系统', r'CRM系统', r'财务系统', r'人力资源系统',
                r'OA系统', r'邮件系统', r'数据库系统'
            ]
        }

        import re
        for entity_type, patterns in audit_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9
                    })

        return entities

    def extract_entities(self, text: str, language: str = 'auto') -> List[Dict[str, Any]]:
        """综合实体抽取"""
        entities = []

        # 检测语言
        if language == 'auto':
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                language = 'zh'
            else:
                language = 'en'

        # 根据语言选择抽取方法
        if language == 'zh':
            entities.extend(self.extract_entities_zh(text))
        else:
            entities.extend(self.extract_entities_en(text))

        # 添加自定义实体
        entities.extend(self.extract_custom_entities(text))

        # 去重
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity['text'], entity['start'], entity['end'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities


class RelationExtractor:
    """关系抽取器"""

    def __init__(self):
        self.relation_patterns = {
            'COVERS': [
                r'(.+?)\s*标准\s*覆盖\s*(.+?)',
                r'(.+?)\s*规范\s*(.+?)',
                r'(.+?)\s*适用于\s*(.+?)'
            ],
            'MITIGATES': [
                r'(.+?)\s*缓解\s*(.+?)',
                r'(.+?)\s*降低\s*(.+?)',
                r'(.+?)\s*减少\s*(.+?)'
            ],
            'ADDRESSES': [
                r'(.+?)\s*解决\s*(.+?)',
                r'(.+?)\s*应对\s*(.+?)',
                r'(.+?)\s*处理\s*(.+?)'
            ],
            'IMPLEMENTS': [
                r'(.+?)\s*实施\s*(.+?)',
                r'(.+?)\s*执行\s*(.+?)',
                r'(.+?)\s*实现\s*(.+?)'
            ],
            'RELATED_TO': [
                r'(.+?)\s*与\s*(.+?)\s*相关',
                r'(.+?)\s*涉及\s*(.+?)',
                r'(.+?)\s*影响\s*(.+?)'
            ]
        }

    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """关系抽取"""
        relations = []

        import re
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # 查找匹配的实体
                    source_entity = self._find_entity_by_text(match.group(1), entities)
                    target_entity = self._find_entity_by_text(match.group(2), entities)

                    if source_entity and target_entity:
                        relations.append({
                            'source': source_entity,
                            'target': target_entity,
                            'relation': relation_type,
                            'confidence': 0.8,
                            'context': match.group()
                        })

        return relations

    def _find_entity_by_text(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """根据文本查找实体"""
        text = text.strip()
        for entity in entities:
            if entity['text'] in text or text in entity['text']:
                return entity
        return None


class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.driver = GraphDatabase.driver(
            NEO4J_CONFIG['uri'],
            auth=(NEO4J_CONFIG['user'], NEO4J_CONFIG['password'])
        )
        self.graph = nx.DiGraph()

    def close(self):
        self.driver.close()

    def build_from_text(self, text: str, language: str = 'auto') -> Dict[str, Any]:
        """从文本构建知识图谱"""
        # 实体抽取
        entities = self.entity_extractor.extract_entities(text, language)
        logger.info(f"抽取到 {len(entities)} 个实体")

        # 关系抽取
        relations = self.relation_extractor.extract_relations(text, entities)
        logger.info(f"抽取到 {len(relations)} 个关系")

        # 构建NetworkX图
        self._build_networkx_graph(entities, relations)

        # 存储到Neo4j
        self._store_to_neo4j(entities, relations)

        return {
            'entities': entities,
            'relations': relations,
            'graph_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges()
            }
        }

    def _build_networkx_graph(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]):
        """构建NetworkX图"""
        # 添加节点
        for entity in entities:
            self.graph.add_node(
                entity['text'],
                label=entity['label'],
                confidence=entity['confidence'],
                start=entity['start'],
                end=entity['end']
            )

        # 添加边
        for relation in relations:
            source = relation['source']['text']
            target = relation['target']['text']
            self.graph.add_edge(
                source, target,
                relation=relation['relation'],
                confidence=relation['confidence'],
                context=relation['context']
            )

    def _store_to_neo4j(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]):
        """存储到Neo4j"""
        with self.driver.session() as session:
            # 存储实体
            for entity in entities:
                session.run("""
                    MERGE (e:Entity {text: $text})
                    SET e.label = $label,
                        e.confidence = $confidence,
                        e.start = $start,
                        e.end = $end,
                        e.created_at = datetime()
                """, **entity)

            # 存储关系
            for relation in relations:
                session.run("""
                    MATCH (source:Entity {text: $source_text})
                    MATCH (target:Entity {text: $target_text})
                    MERGE (source)-[r:RELATES_TO {type: $relation_type}]->(target)
                    SET r.confidence = $confidence,
                        r.context = $context,
                        r.created_at = datetime()
                """,
                            source_text=relation['source']['text'],
                            target_text=relation['target']['text'],
                            relation_type=relation['relation'],
                            confidence=relation['confidence'],
                            context=relation['context']
                            )

    def build_from_documents(self, documents: List[str], language: str = 'auto') -> Dict[str, Any]:
        """从多个文档构建知识图谱"""
        all_entities = []
        all_relations = []

        for i, doc in enumerate(documents):
            logger.info(f"处理文档 {i + 1}/{len(documents)}")
            result = self.build_from_text(doc, language)
            all_entities.extend(result['entities'])
            all_relations.extend(result['relations'])

        # 合并重复实体
        merged_entities = self._merge_entities(all_entities)

        # 重新构建图谱
        self.graph.clear()
        self._build_networkx_graph(merged_entities, all_relations)

        return {
            'entities': merged_entities,
            'relations': all_relations,
            'graph_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges()
            }
        }

    def _merge_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并重复实体"""
        entity_map = {}

        for entity in entities:
            key = entity['text'].lower()
            if key in entity_map:
                # 合并属性
                existing = entity_map[key]
                existing['confidence'] = max(existing['confidence'], entity['confidence'])
                if entity['label'] != existing['label']:
                    existing['label'] = f"{existing['label']}|{entity['label']}"
            else:
                entity_map[key] = entity.copy()

        return list(entity_map.values())

    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """查询知识图谱"""
        with self.driver.session() as session:
            # 简单的Cypher查询
            result = session.run("""
                MATCH (n)-[r]->(m)
                WHERE n.text CONTAINS $query OR m.text CONTAINS $query
                RETURN n, r, m
                LIMIT 50
            """, query=query)

            results = []
            for record in result:
                results.append({
                    'source': dict(record['n']),
                    'relation': dict(record['r']),
                    'target': dict(record['m'])
                })

            return results

    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        with self.driver.session() as session:
            # 节点统计
            node_stats = session.run("""
                MATCH (n)
                RETURN labels(n) as label, count(n) as count
                ORDER BY count DESC
            """)

            # 关系统计
            relation_stats = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relation_type, count(r) as count
                ORDER BY count DESC
            """)

            return {
                'nodes': [dict(record) for record in node_stats],
                'relations': [dict(record) for record in relation_stats]
            }


# 使用示例
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()

    try:
        # 示例文本
        sample_text = """
        COBIT 2019框架为IT治理和管理提供全面的指导。该框架覆盖了用户权限管理、
        财务报告流程、数据备份与恢复等关键业务流程。通过实施访问控制和数据加密等
        控制措施，可以有效缓解数据泄露风险和系统故障风险。
        """

        # 构建知识图谱
        result = builder.build_from_text(sample_text, language='zh')

        print(f"构建完成！")
        print(f"实体数量: {len(result['entities'])}")
        print(f"关系数量: {len(result['relations'])}")
        print(f"图谱统计: {result['graph_stats']}")

        # 查询示例
        query_results = builder.query_graph("COBIT")
        print(f"查询结果: {len(query_results)} 条")

    except Exception as e:
        logger.error(f"知识图谱构建失败: {e}")
    finally:
        builder.close()

