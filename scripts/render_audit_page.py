# -*- coding: utf-8 -*-
from pathlib import Path


def ent(text: str) -> str:
    return "".join(f"&#{ord(ch)};" if ord(ch) > 127 else ch for ch in text)


def write() -> None:
    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{ent('审计项目工作台')} - {ent('审脉 AuditPilot')}</title>
  <link rel="stylesheet" href="../static/app.css">
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand"><div class="brand-icon">AP</div><span>{ent('审脉 AuditPilot')}</span></div>
      <nav class="nav-list">
        <a class="nav-link" href="/">&#8962; <span>{ent('总览')}</span></a>
        <a class="nav-link" href="/chat">&#9679; <span>{ent('审计对话')}</span></a>
        <a class="nav-link active" href="/audit">&#8981; <span>{ent('审计工作台')}</span></a>
        <a class="nav-link" href="/knowledge">&#9638; <span>{ent('知识库')}</span></a>
        <a class="nav-link" href="/training">&#9633; <span>{ent('评测')}</span></a>
        <a class="nav-link" href="/skills">&#9671; <span>Skills</span></a>
      </nav>
      <div class="sidebar-note">{ent('从审计立项、取证、控制测试、复核、报告到整改关闭，形成可追溯的审计交付链路。')}</div>
    </aside>
    <main class="main">
      <div class="topbar">
        <div class="page-title">
          <h1>{ent('审计项目工作台')}</h1>
          <p>{ent('面向审计行业真实交付：场景模板、Agent 分析、证据请求、控制测试、复核和整改闭环。')}</p>
        </div>
        <div class="status-pill"><span class="status-dot"></span><span id="healthText">{ent('加载状态中')}</span></div>
      </div>

      <section class="command-panel">
        <div class="panel">
          <div class="panel-head">
            <div>
              <h2 class="panel-title">{ent('审计指挥栏')}</h2>
              <p class="muted">{ent('这里发起的是一次审计项目分析：Agent 会基于对象、范围、期间、问题和证据生成风险判断、控制测试、证据请求和交付包。')}</p>
            </div>
            <div class="segmented">
              <button class="seg active" data-view="execute">{ent('执行')}</button>
              <button class="seg" data-view="evidence">{ent('取证')}</button>
              <button class="seg" data-view="testing">{ent('测试')}</button>
              <button class="seg" data-view="close">{ent('复核')}</button>
            </div>
          </div>
          <div class="grid grid-4">
            <div class="field"><label>{ent('审计对象')}</label><input id="auditItem" value="{ent('ERP 系统权限管理')}"></div>
            <div class="field"><label>{ent('审计类型')}</label><select id="auditType"><option>{ent('安全审计')}</option><option>{ent('合规审计')}</option><option>{ent('风险评估')}</option><option>{ent('内部控制审计')}</option></select></div>
            <div class="field"><label>{ent('参考标准')}</label><select id="standardType"><option>ISO27001</option><option>COBIT</option><option>SOX</option><option>{ent('数据安全法')}</option></select></div>
            <div class="field"><label>{ent('关注风险')}</label><select id="riskLevel"><option value="high">{ent('高')}</option><option value="medium">{ent('中')}</option><option value="low">{ent('低')}</option></select></div>
          </div>
          <div class="grid grid-2 mt-16">
            <div class="field"><label>{ent('业务背景')}</label><textarea id="businessContext" rows="3" placeholder="{ent('例如：该 ERP 支撑采购、付款、总账和报表流程，近期进行了组织架构调整。')}"></textarea></div>
            <div class="field"><label>{ent('审计范围')}</label><textarea id="auditScope" rows="3" placeholder="{ent('例如：账号生命周期、角色授权、职责分离、特权账号、定期复核。')}"></textarea></div>
          </div>
          <div class="grid grid-3 mt-16">
            <div class="field"><label>{ent('审计期间')}</label><input id="auditPeriod" value="{ent('2026 Q2')}"></div>
            <div class="field"><label>{ent('重点问题')}</label><input id="keyQuestions" placeholder="{ent('例如：是否存在离职未禁用、SoD 冲突、特权账号无复核。')}"></div>
            <div class="field"><label>{ent('已有证据')}</label><input id="existingEvidence" placeholder="{ent('例如：用户清单、角色矩阵、最近一次权限复核记录。')}"></div>
          </div>
          <div class="toolbar mt-16">
            <button class="btn primary" id="runAudit">{ent('运行 Agent 审计')}</button>
            <button class="btn" id="loadControls">{ent('查看控制库')}</button>
            <button class="btn" id="runResearch">{ent('Deep Research')}</button>
            <a class="btn" id="downloadReport" href="#" target="_blank">{ent('下载报告')}</a>
            <a class="btn" id="downloadDelivery" href="#" target="_blank">{ent('下载交付包')}</a>
          </div>
          <div class="section-title"><div><h2>{ent('高频审计场景')}</h2><p>{ent('模板覆盖 ITGC、ERP 权限、数据安全、变更发布、备份恢复和第三方服务。')}</p></div></div>
          <div class="scenario-grid" id="scenarioGrid"></div>
        </div>
        <aside class="panel">
          <h2 class="panel-title">{ent('执行轨迹')}</h2>
          <div id="tracePanel" class="trace-list"><p class="muted">{ent('运行审计后展示 Agent 每一步状态。')}</p></div>
        </aside>
      </section>

      <section class="kpi-row mt-16">
        <div class="card metric"><div class="metric-value" id="riskLevelCard">-</div><div class="metric-label">{ent('剩余风险')}</div></div>
        <div class="card metric"><div class="metric-value" id="riskScoreCard">-</div><div class="metric-label">{ent('风险评分')}</div></div>
        <div class="card metric"><div class="metric-value" id="complianceCard">-</div><div class="metric-label">{ent('合规评分')}</div></div>
        <div class="card metric"><div class="metric-value" id="qualityCard">-</div><div class="metric-label">{ent('质量门')}</div></div>
        <div class="card metric"><div class="metric-value" id="stageCard">-</div><div class="metric-label">{ent('项目阶段')}</div></div>
      </section>

      <section class="grid layout-2 mt-16">
        <div class="panel"><h2 class="panel-title">{ent('审计结论')}</h2><div id="auditResult" class="prewrap muted">{ent('等待分析结果')}</div></div>
        <div class="panel"><h2 class="panel-title">{ent('质量门')}</h2><div id="qualityPanel" class="muted">{ent('等待结果')}</div></div>
      </section>

      <section class="panel mt-16">
        <h2 class="panel-title">{ent('Deep Research 推理')}</h2>
        <div id="researchPanel" class="list dense"><p class="muted">{ent('点击 Deep Research 后展示查询改写、来源融合、推理轨迹和答案评测。')}</p></div>
      </section>

      <section class="panel mt-16">
        <h2 class="panel-title">{ent('审计交付包预览')}</h2>
        <div id="deliveryPreview" class="delivery-grid"><p class="muted">{ent('选择或运行审计档案后展示底稿、证据、访谈、现场日程和复核轨迹。')}</p></div>
      </section>

      <section class="grid grid-2 mt-16">
        <div class="panel"><h2 class="panel-title">{ent('任务计划')}</h2><div id="taskPlan" class="timeline"><p class="muted">{ent('等待结果')}</p></div></div>
        <div class="panel"><h2 class="panel-title">{ent('证据包')}</h2><div id="evidencePack" class="list dense"><p class="muted">{ent('等待结果')}</p></div></div>
      </section>

      <section class="grid grid-2 mt-16">
        <div class="panel"><h2 class="panel-title">{ent('证据请求中心')}</h2><div id="evidenceRequests" class="list dense"><p class="muted">{ent('等待项目档案')}</p></div></div>
        <div class="panel"><h2 class="panel-title">{ent('控制测试工作台')}</h2><div id="controlTests" class="list dense"><p class="muted">{ent('等待项目档案')}</p></div></div>
      </section>

      <section class="panel mt-16"><h2 class="panel-title">{ent('控制矩阵')}</h2><div id="controlMatrix" class="matrix muted">{ent('等待结果')}</div></section>
      <section class="grid grid-2 mt-16"><div class="panel"><h2 class="panel-title">{ent('审计程序')}</h2><div id="auditProgram" class="list dense"><p class="muted">{ent('等待结果')}</p></div></div><div class="panel"><h2 class="panel-title">{ent('抽样计划')}</h2><div id="samplingPlan" class="list dense"><p class="muted">{ent('等待结果')}</p></div></div></section>
      <section class="panel mt-16"><h2 class="panel-title">{ent('审计发现草稿')}</h2><div id="findings" class="grid grid-2"><p class="muted">{ent('等待结果')}</p></div></section>
      <section class="panel mt-16"><h2 class="panel-title">{ent('整改行动计划')}</h2><div id="recommendations" class="grid grid-3"><p class="muted">{ent('等待结果')}</p></div></section>
      <section class="panel mt-16"><h2 class="panel-title">{ent('整改任务跟踪')}</h2><div id="remediationTasks" class="list dense"><p class="muted">{ent('等待结果')}</p></div></section>

      <section class="grid layout-2 mt-16">
        <div class="panel"><h2 class="panel-title">{ent('审计档案')}</h2><div class="toolbar"><button class="btn" id="refreshRuns">{ent('刷新历史')}</button></div><div id="runHistory" class="list dense mt-16"><p class="muted">{ent('暂无历史')}</p></div></div>
        <div class="panel"><h2 class="panel-title">{ent('人工复核')}</h2><div class="grid grid-2"><div class="field"><label>{ent('复核人')}</label><input id="reviewer" value="{ent('审计经理')}"></div><div class="field"><label>{ent('结论')}</label><select id="reviewDecision"><option value="approve">{ent('通过')}</option><option value="need_evidence">{ent('补充证据')}</option><option value="reject">{ent('退回整改')}</option></select></div></div><div class="field mt-12"><label>{ent('复核意见')}</label><textarea id="reviewComment" rows="4" placeholder="{ent('记录人工判断、证据缺口或整改要求')}"></textarea></div><button class="btn primary mt-16" id="submitReview">{ent('提交复核')}</button><div id="reviewResult" class="muted mt-12"></div></div>
      </section>
    </main>
  </div>
  <script src="../static/app.js"></script>
  <script src="../static/audit.js"></script>
</body>
</html>
"""
    Path("templates/audit.html").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    write()
