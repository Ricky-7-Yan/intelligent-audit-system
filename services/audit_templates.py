"""Audit engagement templates for industry scenarios."""

from __future__ import annotations

from typing import Any, Dict, List


AUDIT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "template_id": "tpl-itgc-sox",
        "name": "SOX ITGC 财务系统审计",
        "audit_type": "内部控制审计",
        "standard": "SOX",
        "risk_level": "高",
        "scope": ["访问控制", "变更管理", "作业调度", "备份恢复", "接口对账"],
        "evidence": ["用户清单", "权限复核记录", "变更单", "测试报告", "上线审批", "备份日志", "批处理监控记录"],
        "deliverables": ["ITGC 控制矩阵", "抽样底稿", "例外清单", "管理层整改计划", "复核签字页"],
    },
    {
        "template_id": "tpl-erp-access",
        "name": "ERP 权限与职责分离审计",
        "audit_type": "安全审计",
        "standard": "ISO27001",
        "risk_level": "高",
        "scope": ["账号生命周期", "角色权限", "职责分离", "特权账号", "定期复核"],
        "evidence": ["账号导出", "角色矩阵", "授权审批", "冲突权限报表", "复核记录", "离职人员清单"],
        "deliverables": ["权限风险清单", "SoD 冲突清单", "权限整改计划", "复核底稿"],
    },
    {
        "template_id": "tpl-data-security",
        "name": "数据安全与个人信息处理审计",
        "audit_type": "合规审计",
        "standard": "数据安全法",
        "risk_level": "中",
        "scope": ["数据分类分级", "敏感数据访问", "加密脱敏", "共享审批", "日志审计", "应急响应"],
        "evidence": ["数据目录", "分类分级规则", "访问审批", "加密配置", "脱敏规则", "共享台账", "日志样本"],
        "deliverables": ["数据处理活动审计表", "敏感数据风险清单", "合规缺口分析", "整改路线图"],
    },
    {
        "template_id": "tpl-change-release",
        "name": "生产变更与发布管理审计",
        "audit_type": "风险评估",
        "standard": "COBIT",
        "risk_level": "中",
        "scope": ["需求审批", "开发测试", "上线审批", "回退方案", "紧急变更", "上线后复核"],
        "evidence": ["变更单", "需求审批", "测试证据", "上线记录", "回退方案", "紧急变更审批"],
        "deliverables": ["变更样本测试表", "紧急变更清单", "发布风险评估", "流程优化建议"],
    },
    {
        "template_id": "tpl-backup-recovery",
        "name": "备份恢复与业务连续性审计",
        "audit_type": "风险评估",
        "standard": "ISO27001",
        "risk_level": "中",
        "scope": ["备份策略", "备份成功率", "恢复演练", "RPO/RTO", "灾备切换", "问题整改"],
        "evidence": ["备份策略", "备份日志", "恢复演练报告", "RPO/RTO 定义", "灾备预案", "整改记录"],
        "deliverables": ["备份恢复测试底稿", "业务连续性缺口清单", "恢复能力评估", "整改计划"],
    },
]


def list_audit_templates() -> List[Dict[str, Any]]:
    return AUDIT_TEMPLATES
