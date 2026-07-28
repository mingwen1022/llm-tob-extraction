"""构建域外(OOD)探针集：40 条不属于 cord / duee_fin / ccks_fraud 任何一域的文档。

用途：测量 Orchestrator 在域外输入上的现状行为（实验 A），为「要不要加拒答」
提供基准数据。**不需要任何标注** —— 这个集合不算 F1，只看系统行为。

来源分两类，用 origin 字段如实区分：
  origin=real      —— 真实公开语料，随机采样(seed=42)后按长度过滤
  origin=synthetic —— 找不到合适公开集的类别(中文简历/会议纪要/数字表格)，人工构造

中英文都要有：三个域里两个是中文，若域外全英文，路由可能只是按语言判别，
测出来的就不是「域外行为」而是「语言判别」。

长度对齐域内(中位 116~472 字)，避免长度本身成为混杂变量。

Run:
  python -m shared.build_ood_probe
  python -m shared.build_ood_probe --sample
"""
from __future__ import annotations

import argparse
import json
import os
import random

OUT_DIR = "data/ood_probe"
SEED = 42

# 真实公开语料：(HF数据集, config, 文本列, source标签, 语言, 取几条, 长度下限, 长度上限)
REAL_SOURCES = [
    ("coastalcph/lex_glue", "ledgar", "text", "contract", "en", 6, 200, 1500),
    ("snoop2head/enron_aeslc_emails", None, "text", "email", "en", 6, 200, 1500),
    ("fancyzhx/ag_news", None, "text", "news", "en", 5, 150, 1500),
    ("hfl/cmrc2018", None, "context", "encyclopedia", "zh", 8, 200, 1200),
    ("lansinuote/ChnSentiCorp", None, "text", "review", "zh", 5, 150, 1200),
]

# 构造样本：公开集里找不到合适中文语料的三类。
# 刻意避开三个域的话题面：不含票据金额行项、不含上市公司公告要素、不含诈骗案情。
SYNTHETIC = [
    # ---- 中文简历 ----
    ("resume", "zh", """个人简历
姓名：李明轩    求职意向：后端开发工程师    期望城市：杭州
教育背景
2016.09-2020.06  华中科技大学  计算机科学与技术  本科  GPA 3.6/4.0
工作经历
2020.07-2023.04  某电商平台  后端开发工程师
  负责订单履约链路的服务拆分，将单体应用改造为六个微服务，接口平均响应时间由 480ms 降至 160ms。
  主导缓存层重构，引入多级缓存与热点探测，大促期间数据库 QPS 下降约 40%。
2023.05-至今    某物流科技公司  高级后端开发工程师
  负责运力调度系统的核心排程模块，支撑日均两百万单的分配计算。
专业技能
  语言：Java / Go / Python；框架：Spring Boot、gRPC；中间件：Kafka、Redis、Elasticsearch
  熟悉分布式事务与一致性方案，有大规模并发场景调优经验。
"""),
    ("resume", "zh", """简历
王思远  女  1995年生  硕士  现居北京  联系方式：见附件
求职方向：数据分析师（用户增长方向）
教育经历
  2018-2021  北京师范大学  应用统计学  硕士
  2014-2018  郑州大学  数学与应用数学  学士
实习与工作
  2021.07-2024.02  某在线教育公司  数据分析师
    搭建用户生命周期分析体系，定义新客激活、留存、沉默三段口径，被业务方作为标准指标沿用。
    通过漏斗归因定位到注册流程第三步流失率异常，推动交互改版，次周留存提升 6.2 个百分点。
  2024.03-至今    某内容社区  高级数据分析师
    负责创作者侧激励策略的实验设计与效果评估，累计完成三十余组 A/B 实验。
技能：SQL、Python（pandas/statsmodels）、Tableau；熟悉因果推断与实验设计。
"""),
    ("resume", "zh", """Resume / 个人履历
姓名：陈禹  应聘岗位：产品经理（企业服务方向）
自我评价
  五年 toB 产品经验，熟悉从需求调研、方案设计到上线迭代的完整链路，具备跨部门协作与项目推进能力。
工作经历
  2019.06-2022.08  某 SaaS 服务商  产品经理
    负责权限中心与组织架构模块，设计支持多层级、多租户的角色模型，覆盖客户三百余家。
    推动开放平台接口标准化，将客户平均对接周期从三周压缩到五个工作日。
  2022.09-至今    某工业软件公司  高级产品经理
    负责设备管理平台的告警与工单模块，梳理十二类设备的告警分级规则。
教育背景
  2015-2019  同济大学  工业工程  本科
"""),
    ("resume", "zh", """求职简历
张一鸣  男  28岁  上海  求职意向：算法工程师（自然语言处理）
教育
  2019.09-2022.06  上海交通大学  计算机技术  专业硕士  研究方向：文本分类与序列标注
  2015.09-2019.06  东南大学  自动化  工学学士
科研与项目
  硕士期间参与中文分词与命名实体识别相关课题，在中文信息学报发表论文一篇。
  开源项目：实现了一套轻量级中文文本纠错工具，GitHub 累计 1.2k star。
工作经历
  2022.07-至今  某搜索技术公司  算法工程师
    负责查询理解模块的意图识别模型迭代，线上准确率由 88.4% 提升至 93.1%。
    参与检索排序特征体系重构，主导语义相关性特征的离线评估。
技能：PyTorch、Transformers、Faiss；熟悉模型蒸馏与量化部署。
"""),
    # ---- 中文会议纪要 ----
    ("minutes", "zh", """产品迭代评审会 会议纪要
时间：周三下午 14:00-15:30    地点：三楼会议室 A    主持：产品组 刘工
参会：产品组、前端组、后端组、测试组各一名代表
议题一  下个版本范围确认
  产品组提出本轮迭代聚焦搜索体验优化，包含筛选项重构、结果页排序调整两项。
  前端组反馈筛选项重构涉及组件库改造，工作量评估为五个人日，高于原排期。
  结论：筛选项重构拆为两期，本轮先做交互层，组件库改造下轮进行。
议题二  遗留缺陷处理
  测试组通报当前遗留缺陷 23 个，其中严重级别 2 个，均与导出功能相关。
  后端组确认两个严重缺陷根因相同，为并发导出时的临时文件覆盖问题，本周内修复。
议题三  灰度方案
  同意采用按租户维度灰度，首批放量 5%，观察三天无异常后逐步扩大。
待办
  1. 前端组周五前给出筛选项交互稿  2. 后端组本周修复导出缺陷  3. 测试组补充并发场景用例
"""),
    ("minutes", "zh", """技术方案评审 纪要
会议主题：日志采集链路改造方案评审
参会人员：架构组、运维组、各业务线技术负责人共 9 人
背景
  现有日志采集基于客户端直写，随业务增长出现写入毛刺，且缺乏统一的采样与降级能力。
方案要点
  架构组提出改为「客户端 → 本地代理 → 消息队列 → 存储」四段式链路，在代理层实现采样与背压。
讨论意见
  运维组关注代理层的资源占用，要求给出单节点内存上限与故障时的降级行为。
  业务线甲提出历史日志格式不统一，迁移期间需要双写兼容，建议保留至少一个月过渡期。
  架构组回应：代理层内存上限设为 512MB，超限后按优先级丢弃低级别日志；同意双写过渡一个月。
结论
  方案原则通过，架构组补充容量测算与回滚预案后，下周二再评审一次细节。
"""),
    ("minutes", "zh", """周例会记录
日期：本周一上午    形式：线上会议    记录人：项目助理
一、上周进展同步
  研发侧：完成用户中心模块的接口联调，累计交付接口 34 个，联调通过 31 个。
  设计侧：完成移动端三个主要页面的高保真稿，已交付研发。
  测试侧：完成第一轮冒烟测试，提交问题单 47 条，已关闭 29 条。
二、风险与阻塞
  第三方短信通道的测试账号额度不足，影响验证码相关用例，需采购同事协助补充额度。
  移动端适配机型清单尚未最终确认，测试侧无法排期兼容性测试。
三、本周计划
  研发侧完成剩余接口联调并进入自测；设计侧输出暗色模式规范；测试侧完成第二轮功能测试。
四、需要决策事项
  是否将适配机型范围从二十款收敛到十二款，请项目负责人本周内给出结论。
"""),
    # ---- 纯数字表格（刻意避开票据形态：无金额行项、无合计/找零）----
    ("table", "zh", """服务器资源监控周报（单位：百分比 / GB）
节点        CPU均值  CPU峰值  内存均值  内存峰值  磁盘使用  网络入GB  网络出GB
node-01      34.2     78.5     41.6      69.3      52.1      118.4     96.7
node-02      29.8     71.2     38.9      64.7      48.6      102.3     88.1
node-03      45.1     92.4     57.2      88.5      63.4      156.8     134.2
node-04      31.7     69.8     40.3      66.1      50.9      110.6     92.4
node-05      52.6     95.1     63.8      91.2      71.3      178.2     149.5
节点总数 5    采样间隔 60 秒    统计周期 7 天
"""),
    ("table", "zh", """某班级期末成绩统计表
学号     语文  数学  英语  物理  化学  总分  班级排名
2023011   87    92    79    85    88    431      6
2023012   91    78    88    72    80    409     14
2023013   76    95    83    94    91    439      3
2023014   94    88    92    81    86    441      2
2023015   82    71    75    68    74    370     28
2023016   88    99    86    97    95    465      1
年级平均分 412.6    班级平均分 425.8    参考人数 42
"""),
    ("table", "en", """Quarterly Sensor Calibration Log (readings in mV)
sensor_id   baseline   q1_drift   q2_drift   q3_drift   q4_drift   tolerance
S-1042        512.4       -1.8       -2.3       -3.1       -2.7        +/-5.0
S-1043        509.7       +0.9       +1.4       +2.0       +1.6        +/-5.0
S-1044        515.2       -4.1       -5.6       -6.8       -7.2        +/-5.0
S-1045        511.8       +2.2       +1.9       +2.6       +3.0        +/-5.0
S-1046        508.3       -0.4       -0.7       -1.1       -0.9        +/-5.0
Units under test: 5    Sampling interval: 90 days    Units out of tolerance: 1
"""),
]


def load_real(rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    import warnings
    warnings.filterwarnings("ignore")

    out: list[dict] = []
    for name, cfg, col, source, lang, n, lo, hi in REAL_SOURCES:
        d = load_dataset(name, cfg, split="train[:2000]", verification_mode="no_checks")
        pool = [str(t).strip() for t in d[col] if lo <= len(str(t).strip()) <= hi]
        picked = rng.sample(pool, min(n, len(pool)))
        for t in picked:
            out.append({"source": source, "lang": lang, "origin": "real",
                        "corpus": name, "text": t})
        print(f"  [real] {source:13} <- {name:35} 候选池 {len(pool):4} 取 {len(picked)}")
    return out


def build() -> list[dict]:
    rng = random.Random(SEED)
    records = load_real(rng)
    for source, lang, text in SYNTHETIC:
        records.append({"source": source, "lang": lang, "origin": "synthetic",
                        "corpus": "constructed", "text": text.strip()})
    print(f"  [synthetic] 构造 {len(SYNTHETIC)} 条（简历/会议纪要/数字表格）")

    rng.shuffle(records)
    for i, r in enumerate(records, 1):
        r["id"] = f"ood_{i:03d}"
    return records


def run() -> None:
    records = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "probe.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps({"id": r["id"], "source": r["source"], "lang": r["lang"],
                                "origin": r["origin"], "corpus": r["corpus"],
                                "text": r["text"]}, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"\n[done] {len(records)} 条 -> {path}")
    print("  按来源:", dict(Counter(r["source"] for r in records)))
    print("  按语言:", dict(Counter(r["lang"] for r in records)))
    print("  真实/构造:", dict(Counter(r["origin"] for r in records)))
    L = sorted(len(r["text"]) for r in records)
    print(f"  长度: 中位={L[len(L)//2]}  min={L[0]}  max={L[-1]}")


def sample() -> None:
    for r in build()[:3]:
        print(f"=== {r['id']}  source={r['source']} lang={r['lang']} origin={r['origin']} ===")
        print(r["text"][:300])
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    sample() if args.sample else run()


if __name__ == "__main__":
    main()
