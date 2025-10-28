"""
SFT、RLHF训练管道和Benchmark测评系统
SFT, RLHF Training Pipeline and Benchmark Evaluation System
"""

import os
import json
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import (
    SFTTrainer, RewardTrainer, RewardConfig,
    PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
)
from accelerate import Accelerator
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import requests
from bs4 import BeautifulSoup
import jieba
import pkuseg
from config import PATHS, LLM_CONFIG, TRAINING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    """训练数据结构"""
    instruction: str
    input: str
    output: str
    category: str
    difficulty: str
    source: str


class DataCollector:
    """数据收集器"""

    def __init__(self):
        self.collected_data = []
        jieba.initialize()
        try:
            self.pku_seg = pkuseg.pkuseg()
        except:
            self.pku_seg = None

    def collect_audit_standards_data(self) -> List[TrainingData]:
        """收集审计标准数据"""
        data = []

        # COBIT 2019数据
        cobit_data = [
            {
                "instruction": "解释COBIT 2019框架的核心原则",
                "input": "",
                "output": "COBIT 2019框架包含5个核心原则：1)满足利益相关者需求，2)覆盖企业端到端，3)应用单一集成框架，4)实现整体方法，5)区分治理和管理。这些原则指导组织建立有效的IT治理体系。",
                "category": "governance",
                "difficulty": "medium",
                "source": "cobit_2019"
            },
            {
                "instruction": "COBIT 2019包含多少个管理流程？",
                "input": "",
                "output": "COBIT 2019包含40个管理流程，分为5个领域：1)APO（调整、规划和组织）包含13个流程，2)BAI（构建、获取和实施）包含10个流程，3)DSS（交付、服务和支持）包含6个流程，4)MEA（监控、评估和评估）包含3个流程，5)EDM（评估、指导和监控）包含5个流程。",
                "category": "governance",
                "difficulty": "easy",
                "source": "cobit_2019"
            }
        ]

        # ISO 27001数据
        iso27001_data = [
            {
                "instruction": "ISO 27001标准的主要控制措施类别有哪些？",
                "input": "",
                "output": "ISO 27001标准包含93个控制措施，分为4个主要类别：1)A.5信息安全策略，2)A.6信息安全组织，3)A.7人力资源安全，4)A.8资产管理，5)A.9访问控制，6)A.10密码学，7)A.11物理和环境安全，8)A.12操作安全，9)A.13通信安全，10)A.14系统获取、开发和维护，11)A.15供应商关系，12)A.16信息安全事件管理，13)A.17业务连续性管理，14)A.18合规性。",
                "category": "security",
                "difficulty": "hard",
                "source": "iso27001"
            }
        ]

        # SOX法案数据
        sox_data = [
            {
                "instruction": "SOX法案对内部控制的主要要求是什么？",
                "input": "",
                "output": "SOX法案第404条要求上市公司管理层评估和报告内部控制的充分性，包括：1)管理层必须评估内部控制的有效性，2)外部审计师必须对管理层的评估进行审计，3)必须建立和维护充分的内部控制结构，4)必须定期评估内部控制的有效性，5)必须披露内部控制中的重大缺陷。",
                "category": "compliance",
                "difficulty": "medium",
                "source": "sox"
            }
        ]

        # 数据安全法
        data_security_law_data = [
            {
                "instruction": "数据安全法对数据分类的要求是什么？",
                "input": "",
                "output": "数据安全法要求对数据进行分类分级保护：1)一般数据：指不涉及国家安全、公共安全、经济安全、社会稳定的数据，2)重要数据：指一旦泄露可能直接影响国家安全、经济安全、社会稳定、公共健康和安全的数据，3)核心数据：指关系国家安全、国民经济命脉、重要民生、重大公共利益的数据。不同级别的数据需要采取不同的保护措施。",
                "category": "compliance",
                "difficulty": "medium",
                "source": "data_security_law"
            }
        ]

        all_data = cobit_data + iso27001_data + sox_data + data_security_law_data

        for item in all_data:
            data.append(TrainingData(**item))

        return data

    def collect_risk_assessment_data(self) -> List[TrainingData]:
        """收集风险评估数据"""
        data = []

        risk_scenarios = [
            {
                "instruction": "如何评估IT系统的数据泄露风险？",
                "input": "系统：ERP系统，用户数：1000，数据敏感度：高",
                "output": "评估ERP系统数据泄露风险需要考虑：1)技术风险：系统漏洞、弱密码、未加密传输，2)管理风险：权限管理不当、访问控制缺失，3)人员风险：内部威胁、社会工程攻击，4)环境风险：物理安全、自然灾害。建议采取多层防护措施，包括数据加密、访问控制、监控审计、员工培训等。",
                "category": "risk_assessment",
                "difficulty": "hard",
                "source": "risk_scenarios"
            },
            {
                "instruction": "评估系统可用性风险的方法",
                "input": "系统：财务系统，业务重要性：关键，停机容忍度：4小时",
                "output": "评估财务系统可用性风险：1)识别关键业务功能，2)分析系统依赖关系，3)评估单点故障风险，4)计算RTO和RPO要求，5)评估备份和恢复能力。建议实施：冗余架构、负载均衡、自动故障转移、定期备份测试、灾难恢复计划。",
                "category": "risk_assessment",
                "difficulty": "medium",
                "source": "risk_scenarios"
            }
        ]

        for scenario in risk_scenarios:
            data.append(TrainingData(**scenario))

        return data

    def collect_compliance_check_data(self) -> List[TrainingData]:
        """收集合规性检查数据"""
        data = []

        compliance_scenarios = [
            {
                "instruction": "如何检查用户权限管理是否符合COBIT要求？",
                "input": "当前权限管理：基于角色的访问控制，定期审查：每季度",
                "output": "检查用户权限管理合规性：1)APO07.01：确保IT人员具备适当技能，2)APO07.02：管理IT人员，3)APO07.03：管理IT人员分离，4)APO07.04：管理IT人员访问权限，5)APO07.05：管理IT人员变更。建议：实施最小权限原则、定期权限审查、权限变更审批流程、访问日志监控。",
                "category": "compliance_check",
                "difficulty": "medium",
                "source": "compliance_scenarios"
            },
            {
                "instruction": "检查数据备份流程是否符合ISO 27001要求",
                "input": "当前备份：每日全量备份，保留期：30天，测试频率：每月",
                "output": "检查数据备份流程合规性：1)A.12.3.1：信息备份，2)A.12.3.2：信息恢复，3)A.17.1.2：业务连续性管理。当前流程基本符合要求，建议改进：1)增加增量备份，2)延长保留期至90天，3)增加备份测试频率至每周，4)建立异地备份，5)实施备份加密。",
                "category": "compliance_check",
                "difficulty": "medium",
                "source": "compliance_scenarios"
            }
        ]

        for scenario in compliance_scenarios:
            data.append(TrainingData(**scenario))

        return data

    def collect_web_data(self) -> List[TrainingData]:
        """从网络收集审计相关数据"""
        data = []

        # 审计相关网站
        audit_sites = [
            "https://www.isaca.org/resources/cobit",
            "https://www.iso.org/isoiec-27001-information-security.html",
            "https://www.sec.gov/spotlight/sarbanes-oxley.htm"
        ]

        for site in audit_sites:
            try:
                response = requests.get(site, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')

                # 提取文本内容
                text_content = soup.get_text()

                # 生成训练数据
                if "cobit" in site.lower():
                    data.append(TrainingData(
                        instruction="什么是COBIT框架？",
                        input="",
                        output=text_content[:500] + "...",
                        category="governance",
                        difficulty="easy",
                        source="web_cobit"
                    ))

            except Exception as e:
                logger.warning(f"无法访问网站 {site}: {e}")

        return data

    def collect_all_data(self) -> List[TrainingData]:
        """收集所有训练数据"""
        all_data = []

        logger.info("收集审计标准数据...")
        all_data.extend(self.collect_audit_standards_data())

        logger.info("收集风险评估数据...")
        all_data.extend(self.collect_risk_assessment_data())

        logger.info("收集合规性检查数据...")
        all_data.extend(self.collect_compliance_check_data())

        logger.info("收集网络数据...")
        all_data.extend(self.collect_web_data())

        logger.info(f"总共收集了 {len(all_data)} 条训练数据")
        return all_data

    def save_data(self, data: List[TrainingData], file_path: str):
        """保存训练数据"""
        try:
            data_dict = []
            for item in data:
                data_dict.append({
                    'instruction': item.instruction,
                    'input': item.input,
                    'output': item.output,
                    'category': item.category,
                    'difficulty': item.difficulty,
                    'source': item.source
                })

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"训练数据已保存到 {file_path}")

        except Exception as e:
            logger.error(f"保存训练数据失败: {e}")


class SFTTrainer:
    """监督微调训练器"""

    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def setup_model(self):
        """设置模型和分词器"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            logger.info(f"模型 {self.model_name} 加载成功")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def prepare_dataset(self, data: List[TrainingData]) -> Dataset:
        """准备训练数据集"""
        try:
            # 格式化数据
            formatted_data = []
            for item in data:
                # 构建训练文本
                text = f"<|im_start|>user\n{item.instruction}\n{item.input}<|im_end|>\n<|im_start|>assistant\n{item.output}<|im_end|>"

                formatted_data.append({
                    'text': text,
                    'instruction': item.instruction,
                    'input': item.input,
                    'output': item.output,
                    'category': item.category,
                    'difficulty': item.difficulty
                })

            # 创建数据集
            dataset = Dataset.from_list(formatted_data)

            # 分割数据集
            train_test_split = dataset.train_test_split(test_size=0.1, seed=42)

            logger.info(
                f"数据集准备完成，训练集: {len(train_test_split['train'])} 条，测试集: {len(train_test_split['test'])} 条")

            return train_test_split

        except Exception as e:
            logger.error(f"数据集准备失败: {e}")
            raise

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """执行SFT训练"""
        try:
            # 设置训练参数
            training_args = TrainingArguments(
                output_dir=str(PATHS['models'] / 'sft_model'),
                num_train_epochs=TRAINING_CONFIG['num_epochs'],
                per_device_train_batch_size=TRAINING_CONFIG['batch_size'],
                per_device_eval_batch_size=TRAINING_CONFIG['batch_size'],
                warmup_steps=TRAINING_CONFIG['warmup_steps'],
                learning_rate=TRAINING_CONFIG['learning_rate'],
                logging_steps=50,
                save_steps=TRAINING_CONFIG['save_steps'],
                eval_steps=TRAINING_CONFIG['eval_steps'],
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
                run_name=f"sft_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # 数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # 创建训练器
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            # 开始训练
            logger.info("开始SFT训练...")
            self.trainer.train()

            # 保存模型
            self.trainer.save_model()
            self.tokenizer.save_pretrained(str(PATHS['models'] / 'sft_model'))

            logger.info("SFT训练完成")

        except Exception as e:
            logger.error(f"SFT训练失败: {e}")
            raise


class RLHFTrainer:
    """强化学习人类反馈训练器"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.reward_model = None
        self.ppo_trainer = None

    def setup_reward_model(self):
        """设置奖励模型"""
        try:
            # 使用预训练的奖励模型
            self.reward_model = AutoModelForCausalLM.from_pretrained(
                "OpenAssistant/reward-model-deberta-v3-large-v2",
                torch_dtype=torch.float16,
                device_map="auto"
            )

            logger.info("奖励模型加载成功")

        except Exception as e:
            logger.error(f"奖励模型加载失败: {e}")
            raise

    def train_reward_model(self, preference_data: List[Dict[str, Any]]):
        """训练奖励模型"""
        try:
            # 准备偏好数据
            dataset = Dataset.from_list(preference_data)

            # 设置训练参数
            training_args = RewardConfig(
                output_dir=str(PATHS['models'] / 'reward_model'),
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                warmup_steps=100,
                learning_rate=1e-5,
                logging_steps=50,
                save_steps=500,
                eval_steps=100,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False
            )

            # 创建奖励训练器
            reward_trainer = RewardTrainer(
                model=self.reward_model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
            )

            # 开始训练
            logger.info("开始奖励模型训练...")
            reward_trainer.train()

            # 保存模型
            reward_trainer.save_model()

            logger.info("奖励模型训练完成")

        except Exception as e:
            logger.error(f"奖励模型训练失败: {e}")
            raise

    def train_with_ppo(self, dataset: Dataset):
        """使用PPO训练"""
        try:
            # 加载SFT模型
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # 设置PPO配置
            ppo_config = PPOConfig(
                model_name=self.model_path,
                learning_rate=1e-5,
                batch_size=4,
                mini_batch_size=1,
                gradient_accumulation_steps=4,
                optimize_cuda_cache=True,
                early_stopping=True,
                target_kl=0.1,
                ppo_epochs=4,
                seed=42
            )

            # 创建PPO训练器
            self.ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=model,
                ref_model=None,
                tokenizer=AutoTokenizer.from_pretrained(self.model_path)
            )

            # 准备训练数据
            for item in dataset:
                query = item['instruction']
                response = item['output']

                # 计算奖励
                reward = self._calculate_reward(query, response)

                # PPO训练步骤
                self.ppo_trainer.step([query], [response], [reward])

            # 保存模型
            self.ppo_trainer.save_model(str(PATHS['models'] / 'rlhf_model'))

            logger.info("PPO训练完成")

        except Exception as e:
            logger.error(f"PPO训练失败: {e}")
            raise

    def _calculate_reward(self, query: str, response: str) -> float:
        """计算奖励分数"""
        try:
            # 简化的奖励计算
            # 实际应用中应该使用训练好的奖励模型

            # 基于长度的奖励
            length_reward = min(len(response) / 100, 1.0)

            # 基于关键词的奖励
            audit_keywords = ['审计', '风险', '合规', '控制', '标准', '流程']
            keyword_reward = sum(1 for keyword in audit_keywords if keyword in response) / len(audit_keywords)

            # 综合奖励
            total_reward = (length_reward * 0.3 + keyword_reward * 0.7)

            return total_reward

        except Exception as e:
            logger.error(f"计算奖励失败: {e}")
            return 0.5


class BenchmarkEvaluator:
    """基准测试评估器"""

    def __init__(self):
        self.test_cases = []
        self.results = []

    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建测试用例"""
        test_cases = [
            {
                "id": "test_001",
                "category": "governance",
                "question": "COBIT 2019框架的核心原则是什么？",
                "expected_answer": "COBIT 2019框架包含5个核心原则",
                "evaluation_criteria": ["准确性", "完整性", "专业性"]
            },
            {
                "id": "test_002",
                "category": "risk_assessment",
                "question": "如何评估IT系统的数据泄露风险？",
                "expected_answer": "需要考虑技术风险、管理风险、人员风险和环境风险",
                "evaluation_criteria": ["准确性", "实用性", "专业性"]
            },
            {
                "id": "test_003",
                "category": "compliance",
                "question": "SOX法案对内部控制的主要要求是什么？",
                "expected_answer": "要求管理层评估和报告内部控制的充分性",
                "evaluation_criteria": ["准确性", "合规性", "专业性"]
            },
            {
                "id": "test_004",
                "category": "security",
                "question": "ISO 27001标准的主要控制措施类别有哪些？",
                "expected_answer": "包含93个控制措施，分为14个主要类别",
                "evaluation_criteria": ["准确性", "完整性", "专业性"]
            },
            {
                "id": "test_005",
                "category": "data_protection",
                "question": "数据安全法对数据分类的要求是什么？",
                "expected_answer": "要求对数据进行分类分级保护",
                "evaluation_criteria": ["准确性", "合规性", "实用性"]
            }
        ]

        return test_cases

    def evaluate_model(self, model, tokenizer, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估模型"""
        try:
            results = []

            for test_case in test_cases:
                # 生成回答
                prompt = f"<|im_start|>user\n{test_case['question']}<|im_end|>\n<|im_start|>assistant\n"

                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split("<|im_start|>assistant\n")[-1]

                # 评估回答
                evaluation = self._evaluate_response(
                    test_case['question'],
                    response,
                    test_case['expected_answer'],
                    test_case['evaluation_criteria']
                )

                results.append({
                    'test_id': test_case['id'],
                    'category': test_case['category'],
                    'question': test_case['question'],
                    'expected_answer': test_case['expected_answer'],
                    'actual_answer': response,
                    'evaluation': evaluation
                })

            # 计算总体指标
            overall_metrics = self._calculate_overall_metrics(results)

            return {
                'results': results,
                'overall_metrics': overall_metrics,
                'evaluation_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return {}

    def _evaluate_response(self, question: str, response: str, expected: str, criteria: List[str]) -> Dict[str, float]:
        """评估单个回答"""
        try:
            evaluation = {}

            # 准确性评估
            if '准确性' in criteria:
                accuracy = self._calculate_accuracy(response, expected)
                evaluation['accuracy'] = accuracy

            # 完整性评估
            if '完整性' in criteria:
                completeness = self._calculate_completeness(response, expected)
                evaluation['completeness'] = completeness

            # 专业性评估
            if '专业性' in criteria:
                professionalism = self._calculate_professionalism(response)
                evaluation['professionalism'] = professionalism

            # 实用性评估
            if '实用性' in criteria:
                practicality = self._calculate_practicality(response)
                evaluation['practicality'] = practicality

            # 合规性评估
            if '合规性' in criteria:
                compliance = self._calculate_compliance(response)
                evaluation['compliance'] = compliance

            return evaluation

        except Exception as e:
            logger.error(f"回答评估失败: {e}")
            return {}

    def _calculate_accuracy(self, response: str, expected: str) -> float:
        """计算准确性"""
        try:
            # 简化的准确性计算
            response_lower = response.lower()
            expected_lower = expected.lower()

            # 计算关键词匹配
            expected_words = set(expected_lower.split())
            response_words = set(response_lower.split())

            if not expected_words:
                return 0.0

            match_ratio = len(expected_words.intersection(response_words)) / len(expected_words)
            return min(match_ratio, 1.0)

        except Exception as e:
            logger.error(f"准确性计算失败: {e}")
            return 0.0

    def _calculate_completeness(self, response: str, expected: str) -> float:
        """计算完整性"""
        try:
            # 基于长度的完整性评估
            response_length = len(response)
            expected_length = len(expected)

            if expected_length == 0:
                return 0.0

            length_ratio = min(response_length / expected_length, 1.0)
            return length_ratio

        except Exception as e:
            logger.error(f"完整性计算失败: {e}")
            return 0.0

    def _calculate_professionalism(self, response: str) -> float:
        """计算专业性"""
        try:
            # 审计专业术语
            professional_terms = [
                '审计', '风险', '合规', '控制', '标准', '流程', '治理',
                '内部控制', '风险评估', '合规性', '审计标准', '审计程序',
                'COBIT', 'ISO', 'SOX', 'GDPR', '数据安全法'
            ]

            response_lower = response.lower()
            term_count = sum(1 for term in professional_terms if term.lower() in response_lower)

            professionalism_score = min(term_count / len(professional_terms), 1.0)
            return professionalism_score

        except Exception as e:
            logger.error(f"专业性计算失败: {e}")
            return 0.0

    def _calculate_practicality(self, response: str) -> float:
        """计算实用性"""
        try:
            # 实用性关键词
            practical_keywords = [
                '建议', '措施', '方法', '步骤', '流程', '实施', '执行',
                '如何', '怎样', '具体', '操作', '实践'
            ]

            response_lower = response.lower()
            keyword_count = sum(1 for keyword in practical_keywords if keyword in response_lower)

            practicality_score = min(keyword_count / len(practical_keywords), 1.0)
            return practicality_score

        except Exception as e:
            logger.error(f"实用性计算失败: {e}")
            return 0.0

    def _calculate_compliance(self, response: str) -> float:
        """计算合规性"""
        try:
            # 合规性关键词
            compliance_keywords = [
                '符合', '合规', '标准', '法规', '要求', '规定', '法律',
                '必须', '应该', '需要', '遵循', '遵守'
            ]

            response_lower = response.lower()
            keyword_count = sum(1 for keyword in compliance_keywords if keyword in response_lower)

            compliance_score = min(keyword_count / len(compliance_keywords), 1.0)
            return compliance_score

        except Exception as e:
            logger.error(f"合规性计算失败: {e}")
            return 0.0

    def _calculate_overall_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算总体指标"""
        try:
            all_scores = []
            category_scores = {}

            for result in results:
                evaluation = result['evaluation']
                category = result['category']

                # 计算平均分数
                scores = list(evaluation.values())
                if scores:
                    avg_score = sum(scores) / len(scores)
                    all_scores.append(avg_score)

                    if category not in category_scores:
                        category_scores[category] = []
                    category_scores[category].append(avg_score)

            # 计算总体指标
            overall_metrics = {
                'overall_score': sum(all_scores) / len(all_scores) if all_scores else 0.0,
                'total_tests': len(results),
                'category_scores': {
                    category: sum(scores) / len(scores)
                    for category, scores in category_scores.items()
                }
            }

            return overall_metrics

        except Exception as e:
            logger.error(f"总体指标计算失败: {e}")
            return {}


# 使用示例
if __name__ == "__main__":
    try:
        # 数据收集
        data_collector = DataCollector()
        training_data = data_collector.collect_all_data()

        # 保存训练数据
        data_collector.save_data(training_data, str(PATHS['training_data'] / 'audit_training_data.json'))

        # SFT训练
        sft_trainer = SFTTrainer()
        sft_trainer.setup_model()

        # 准备数据集
        train_test_split = sft_trainer.prepare_dataset(training_data)

        # 开始训练
        sft_trainer.train(train_test_split['train'], train_test_split['test'])

        # RLHF训练
        rlhf_trainer = RLHFTrainer(str(PATHS['models'] / 'sft_model'))
        rlhf_trainer.setup_reward_model()

        # 准备偏好数据（简化）
        preference_data = [
            {
                'prompt': '什么是COBIT框架？',
                'chosen': 'COBIT是IT治理和管理框架',
                'rejected': 'COBIT是一个软件'
            }
        ]

        # 训练奖励模型
        rlhf_trainer.train_reward_model(preference_data)

        # PPO训练
        rlhf_trainer.train_with_ppo(train_test_split['train'])

        # 基准测试
        evaluator = BenchmarkEvaluator()
        test_cases = evaluator.create_test_cases()

        # 评估模型
        model = AutoModelForCausalLM.from_pretrained(str(PATHS['models'] / 'rlhf_model'))
        tokenizer = AutoTokenizer.from_pretrained(str(PATHS['models'] / 'rlhf_model'))

        evaluation_results = evaluator.evaluate_model(model, tokenizer, test_cases)

        # 保存评估结果
        with open(str(PATHS['models'] / 'evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        print("训练和评估完成！")
        print(f"总体评分: {evaluation_results['overall_metrics']['overall_score']:.2f}")

    except Exception as e:
        logger.error(f"训练管道执行失败: {e}")

