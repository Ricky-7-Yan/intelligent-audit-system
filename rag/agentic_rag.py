"""
Agentic RAG系统
Agentic Retrieval-Augmented Generation System
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from config import PATHS, LLM_CONFIG
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup
import jieba
import pkuseg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器"""

    def __init__(self):
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

        # 初始化中文分词器
        jieba.initialize()
        try:
            self.pku_seg = pkuseg.pkuseg()
        except:
            self.pku_seg = None

    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """处理文本，分割成文档块"""
        if not metadata:
            metadata = {}

        # 添加时间戳
        metadata['processed_at'] = datetime.now().isoformat()

        # 分割文本
        chunks = self.text_splitter.split_text(text)

        # 创建文档对象
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata['chunk_id'] = i
            doc_metadata['chunk_size'] = len(chunk)

            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))

        return documents

    def process_file(self, file_path: str) -> List[Document]:
        """处理文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = {
                'source': file_path,
                'file_type': os.path.splitext(file_path)[1],
                'file_size': len(content)
            }

            return self.process_text(content, metadata)

        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return []

    def process_web_content(self, url: str) -> List[Document]:
        """处理网页内容"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # 提取文本内容
            text_content = soup.get_text()

            # 清理文本
            lines = text_content.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            cleaned_text = '\n'.join(cleaned_lines)

            metadata = {
                'source': url,
                'file_type': 'web',
                'title': soup.title.string if soup.title else '',
                'scraped_at': datetime.now().isoformat()
            }

            return self.process_text(cleaned_text, metadata)

        except Exception as e:
            logger.error(f"处理网页内容失败 {url}: {e}")
            return []

class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        # 初始化嵌入模型 - 优先使用本地模型
        try:
            import os
            # 检查本地模型路径
            local_model_path = PATHS['models'] / 'sentence-transformers' / 'paraphrase-multilingual-MiniLM-L12-v2'

            if os.path.exists(local_model_path):
                logger.info(f"使用本地模型: {local_model_path}")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=str(local_model_path),
                    model_kwargs={'device': 'cpu'}
                )
            else:
                # 尝试从镜像站下载
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                logger.info("尝试从镜像站下载模型...")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={'device': 'cpu'}
                )
        except Exception as e:
            logger.warning(f"无法加载嵌入模型，使用简化版本: {e}")
            # 使用简化的嵌入模型
            self.embedding_model = None

        # 初始化ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(PATHS['data'] / 'chroma_db'),
            settings=Settings(anonymized_telemetry=False)
        )

        # 初始化FAISS索引
        self.faiss_index = None
        self.faiss_documents = []

        # 初始化向量存储
        self.vectorstore = Chroma(
            collection_name="audit_knowledge",
            embedding_function=self.embedding_model,
            persist_directory=str(PATHS['data'] / 'chroma_db')
        )

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量存储"""
        try:
            # 添加到ChromaDB
            self.vectorstore.add_documents(documents)

            # 添加到FAISS索引
            if not self.faiss_index:
                # 创建新的FAISS索引
                embeddings = self.embedding_model.embed_documents([doc.page_content for doc in documents])
                dimension = len(embeddings[0])
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(np.array(embeddings).astype('float32'))
                self.faiss_documents.extend(documents)
            else:
                # 更新现有索引
                embeddings = self.embedding_model.embed_documents([doc.page_content for doc in documents])
                self.faiss_index.add(np.array(embeddings).astype('float32'))
                self.faiss_documents.extend(documents)

            logger.info(f"成功添加 {len(documents)} 个文档到向量存储")

        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {e}")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """相似性搜索"""
        try:
            # 使用ChromaDB搜索
            chroma_results = self.vectorstore.similarity_search(query, k=k)

            # 使用FAISS搜索
            faiss_results = []
            if self.faiss_index and self.faiss_documents:
                query_embedding = self.embedding_model.embed_query(query)
                scores, indices = self.faiss_index.search(
                    np.array([query_embedding]).astype('float32'), k
                )

                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.faiss_documents):
                        doc = self.faiss_documents[idx]
                        doc.metadata['faiss_score'] = float(score)
                        faiss_results.append(doc)

            # 合并结果并去重
            all_results = chroma_results + faiss_results
            unique_results = []
            seen_content = set()

            for doc in all_results:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_results.append(doc)

            return unique_results[:k]

        except Exception as e:
            logger.error(f"相似性搜索失败: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            collection = self.chroma_client.get_collection("audit_knowledge")
            count = collection.count()

            return {
                'total_documents': count,
                'faiss_index_size': len(self.faiss_documents) if self.faiss_documents else 0,
                'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2'
            }

        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {}

class AgenticRetriever:
    """智能检索器"""

    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.vectorstore_manager = vectorstore_manager
        self.llm = ChatOpenAI(
            api_key=LLM_CONFIG['api_key'],
            base_url=LLM_CONFIG['base_url'],
            model=LLM_CONFIG['model'],
            temperature=0.1
        )

    def generate_queries(self, original_query: str) -> List[str]:
        """生成多个查询变体"""
        prompt = f"""
        基于以下原始查询，生成3个不同的查询变体，用于检索相关信息：
        
        原始查询: {original_query}
        
        请生成3个查询变体，每个查询应该：
        1. 保持原始查询的核心意图
        2. 使用不同的表达方式
        3. 包含相关的同义词或相关术语
        4. 适合审计领域的专业术语
        
        请只返回3个查询，每行一个，不要其他内容。
        """

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            queries = [original_query]  # 包含原始查询

            # 解析生成的查询
            generated_queries = response.content.strip().split('\n')
            for query in generated_queries:
                query = query.strip()
                if query and query != original_query:
                    queries.append(query)

            return queries[:4]  # 最多4个查询

        except Exception as e:
            logger.error(f"生成查询变体失败: {e}")
            return [original_query]

    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """检索相关文档"""
        try:
            # 生成查询变体
            queries = self.generate_queries(query)

            # 对每个查询进行检索
            all_results = []
            for q in queries:
                results = self.vectorstore_manager.similarity_search(q, k=k)
                all_results.extend(results)

            # 去重和排序
            unique_results = []
            seen_content = set()

            for doc in all_results:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_results.append(doc)

            # 按相关性排序（如果有分数）
            unique_results.sort(
                key=lambda x: x.metadata.get('faiss_score', 0),
                reverse=True
            )

            return unique_results[:k]

        except Exception as e:
            logger.error(f"检索文档失败: {e}")
            return []

    def contextual_retrieval(self, query: str, context: Dict[str, Any], k: int = 5) -> List[Document]:
        """基于上下文的检索"""
        try:
            # 构建上下文增强的查询
            enhanced_query = f"{query}"

            if context.get('audit_type'):
                enhanced_query += f" {context['audit_type']}"

            if context.get('standard_type'):
                enhanced_query += f" {context['standard_type']}"

            if context.get('risk_level'):
                enhanced_query += f" {context['risk_level']}"

            return self.retrieve_documents(enhanced_query, k)

        except Exception as e:
            logger.error(f"上下文检索失败: {e}")
            return self.retrieve_documents(query, k)

class RAGPipeline:
    """RAG管道"""

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vectorstore_manager = VectorStoreManager()
        self.retriever = AgenticRetriever(self.vectorstore_manager)

        # 初始化LLM
        self.llm = ChatOpenAI(
            api_key=LLM_CONFIG['api_key'],
            base_url=LLM_CONFIG['base_url'],
            model=LLM_CONFIG['model'],
            temperature=LLM_CONFIG['temperature']
        )

        # 创建提示模板
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""基于以下上下文信息，回答用户的问题。请提供准确、专业的回答，并引用相关的标准和法规。

上下文信息:
{context}

用户问题: {question}

请基于上下文信息提供详细的回答，包括：
1. 直接回答用户的问题
2. 引用相关的标准和法规
3. 提供具体的建议和措施
4. 如果上下文信息不足，请说明需要补充的信息

回答:"""
        )

    def add_knowledge(self, text: str, metadata: Dict[str, Any] = None) -> None:
        """添加知识到RAG系统"""
        try:
            documents = self.document_processor.process_text(text, metadata)
            self.vectorstore_manager.add_documents(documents)
            logger.info(f"成功添加知识: {len(documents)} 个文档块")

        except Exception as e:
            logger.error(f"添加知识失败: {e}")

    def add_file(self, file_path: str) -> None:
        """添加文件到RAG系统"""
        try:
            documents = self.document_processor.process_file(file_path)
            if documents:
                self.vectorstore_manager.add_documents(documents)
                logger.info(f"成功添加文件: {file_path}")

        except Exception as e:
            logger.error(f"添加文件失败: {e}")

    def add_web_content(self, url: str) -> None:
        """添加网页内容到RAG系统"""
        try:
            documents = self.document_processor.process_web_content(url)
            if documents:
                self.vectorstore_manager.add_documents(documents)
                logger.info(f"成功添加网页内容: {url}")

        except Exception as e:
            logger.error(f"添加网页内容失败: {e}")

    def query(self, question: str, context: Dict[str, Any] = None, k: int = 5) -> Dict[str, Any]:
        """查询RAG系统"""
        try:
            # 检索相关文档
            if context:
                documents = self.retriever.contextual_retrieval(question, context, k)
            else:
                documents = self.retriever.retrieve_documents(question, k)

            if not documents:
                return {
                    'answer': '抱歉，没有找到相关的信息来回答您的问题。',
                    'sources': [],
                    'confidence': 0.0
                }

            # 构建上下文
            context_text = "\n\n".join([
                f"来源: {doc.metadata.get('source', '未知')}\n内容: {doc.page_content}"
                for doc in documents
            ])

            # 生成回答
            prompt = self.prompt_template.format(
                context=context_text,
                question=question
            )

            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])

            # 计算置信度
            confidence = self._calculate_confidence(documents, question)

            return {
                'answer': response.content,
                'sources': [
                    {
                        'source': doc.metadata.get('source', '未知'),
                        'content': doc.page_content[:200] + '...',
                        'score': doc.metadata.get('faiss_score', 0.0)
                    }
                    for doc in documents
                ],
                'confidence': confidence,
                'retrieved_docs_count': len(documents)
            }

        except Exception as e:
            logger.error(f"RAG查询失败: {e}")
            return {
                'answer': f'查询失败: {str(e)}',
                'sources': [],
                'confidence': 0.0
            }

    def _calculate_confidence(self, documents: List[Document], question: str) -> float:
        """计算回答的置信度"""
        try:
            if not documents:
                return 0.0

            # 基于文档数量和相关性分数计算置信度
            scores = [doc.metadata.get('faiss_score', 0.0) for doc in documents]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            # 基于文档数量调整置信度
            doc_count_factor = min(len(documents) / 5.0, 1.0)

            confidence = (avg_score * 0.7 + doc_count_factor * 0.3)
            return min(confidence, 1.0)

        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.5

    def get_statistics(self) -> Dict[str, Any]:
        """获取RAG系统统计信息"""
        try:
            vectorstore_stats = self.vectorstore_manager.get_collection_stats()

            return {
                'vectorstore_stats': vectorstore_stats,
                'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
                'chunk_size': 1000,
                'chunk_overlap': 200
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

# 使用示例
if __name__ == "__main__":
    rag_pipeline = RAGPipeline()

    try:
        # 添加示例知识
        sample_knowledge = """
        COBIT 2019是ISACA发布的IT治理和管理框架。该框架包含5个治理原则和40个管理流程，
        分为治理和管理两个领域。COBIT 2019帮助组织建立有效的IT治理体系，
        确保IT投资与业务目标对齐，并管理IT相关的风险。
        
        ISO/IEC 27001是信息安全管理体系的国际标准。该标准要求组织建立、
        实施、维护和持续改进信息安全管理体系，包括93个控制措施，
        分为4个主要类别：组织控制、人员控制、物理控制和技术控制。
        """

        rag_pipeline.add_knowledge(sample_knowledge, {
            'source': 'audit_standards',
            'type': 'standard',
            'language': 'zh'
        })

        # 测试查询
        test_questions = [
            "COBIT 2019框架包含哪些内容？",
            "ISO 27001标准有哪些控制措施？",
            "如何建立IT治理体系？"
        ]

        for question in test_questions:
            print(f"\n问题: {question}")
            result = rag_pipeline.query(question)
            print(f"回答: {result['answer']}")
            print(f"置信度: {result['confidence']:.2f}")
            print(f"检索文档数: {result['retrieved_docs_count']}")

        # 获取统计信息
        stats = rag_pipeline.get_statistics()
        print(f"\nRAG系统统计: {stats}")

    except Exception as e:
        logger.error(f"RAG系统测试失败: {e}")
