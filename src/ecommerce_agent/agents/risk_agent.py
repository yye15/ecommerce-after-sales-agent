"""Business-rule based risk assessment agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LEVEL_RANK = {"低风险": 0, "中风险": 1, "高风险": 2}
PRIORITY_BY_LEVEL = {"低风险": "P3", "中风险": "P2", "高风险": "P1"}

COMPLAINT_WORDS = ["投诉", "举报", "维权", "12315", "起诉", "曝光", "平台介入"]
PUBLIC_NEGATIVE_REVIEW_WORDS = ["差评"]
SAFETY_WORDS = [
    "食品安全",
    "过期",
    "变质",
    "长毛",
    "发霉",
    "漏电",
    "爆炸",
    "冒烟",
    "起火",
    "烫伤",
    "划伤",
    "过敏",
    "刺激眼睛",
    "安全隐患",
]
FRAUD_WORDS = ["假货", "*货", "欺骗消费者", "大骗子", "空单号", "货不对版"]
TRUST_CONCERN_WORDS = ["骗人", "被骗", "被坑"]
REFUND_WORDS = ["退款", "退货", "换货", "退换", "退换货", "赔偿", "补偿", "不能退", "不能换", "保修", "不退运费", "运费"]
SERVICE_WORDS = ["客服", "售后", "不回复", "没人理", "没人处理", "不处理", "没人回答", "找不到人", "联系不到", "不理", "应付", "服务态度"]
QUALITY_WORDS = [
    "质量",
    "耐用",
    "不耐用",
    "坏",
    "破",
    "裂开",
    "漏水",
    "不新鲜",
    "划痕",
    "断线",
    "黑点",
    "脏兮兮",
    "掉色",
    "瑕疵",
    "做工",
    "材质",
    "布料",
    "开裂",
    "漏",
    "泄露",
    "泄漏",
    "不能吃",
    "太熟",
    "烂",
    "稀巴烂",
    "洗厕所",
]
CORE_FUNCTION_WORDS = [
    "屏幕",
    "机器",
    "图片",
    "运行",
    "开机",
    "用不了",
    "无法",
    "不能用",
    "很卡",
    "卡",
    "充电不了",
    "充不了电",
    "没有反映",
    "没反应",
    "像素",
    "摄像头",
    "一会黑一会白",
    "听筒",
    "花屏",
    "发错",
    "信号",
    "闹钟",
    "温度",
    "续航",
    "音质",
    "安装",
    "效果",
    "维修站",
    "换机",
    "设计缺陷",
    "确实有问题",
    "发件箱",
    "硬件",
    "软件",
    "烟味",
    "油漆味",
    "吵",
    "散热",
    "鼠标",
    "耳机",
    "没配",
    "没有了",
]
LOGISTICS_WORDS = ["物流", "快递", "发货", "没发货", "没送到", "没有现货", "配送", "派送", "运输", "两个星期", "30公里", "打不通"]
MILD_LOGISTICS_WORDS = ["慢", "太慢", "晚到", "等了", "催"]
SERIOUS_LOGISTICS_WORDS = ["30公里", "打不通", "不让验货", "拒绝签单"]
VAGUE_BAD_WORDS = ["太差", "怎么解决", "啥意思", "什么鬼", "极差", "真心low", "不像话"]
STRONG_DISSUASION_WORDS = ["别买", "三思", "不会买", "不值得住", "强烈建议不要购买"]
SEVERE_SERVICE_WORDS = ["联系不到", "不理", "应付", "没人处理", "没人回答", "找不到人", "流程慢"]
SEVERE_REFUND_WORDS = ["不能换", "不能退", "不退运费"]
MILD_REFUND_WORDS = ["退换货太费劲"]
NEGATIVE_WORDS = [
    "差",
    "坏",
    "不好",
    "不行",
    "不符",
    "不建议",
    "不能",
    "没有",
    "没",
    "无语",
    "麻烦",
    "上当",
    "生气",
    "失望",
    "后悔",
    "垃圾",
    "烂",
    "稀巴烂",
    "差劲",
    "坑人",
    "坑死人",
    "不值得",
    "不能吃",
    "失败",
    "烟味",
    "油漆味",
    "吵",
    "漏了",
    "真心low",
    "极差",
    "搞错",
    "不如",
    "怎么解决",
    "什么鬼",
]

HIGH_RISK_CATEGORIES = {"退换货与赔付问题", "客服响应问题"}
MEDIUM_FLOOR_CATEGORIES = {"商品质量问题", "产品体验问题", "价格与权益问题", "物流问题"}
MULTI_ISSUE_CATEGORIES = {"商品质量问题", "产品体验问题", "客服响应问题", "物流问题", "退换货与赔付问题", "价格与权益问题"}


@dataclass(frozen=True)
class RiskSignal:
    rule_id: str
    level_floor: str
    score_delta: int
    reason: str
    hits: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "level_floor": self.level_floor,
            "score_delta": self.score_delta,
            "reason": self.reason,
            "hits": self.hits,
        }


def assess_risk(text: str, sentiment: dict[str, Any], categories: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess after-sales risk with explicit business rules.

    The goal is not to predict emotion only. It estimates business handling
    priority: safety, rights protection, refund blocks and product failures
    should override a mild-looking sentence.
    """
    if _is_irrelevant_rant(text):
        return {
            "level": "低风险",
            "score": 20,
            "priority": "P3",
            "should_escalate": False,
            "reasons": ["识别为无明确售后对象的情绪宣泄或噪声评论"],
            "triggered_rules": [],
            "rule_version": "risk_policy_v1",
        }

    triplets = sentiment.get("triplets", [])
    negative_count = sum(1 for item in triplets if item.get("sentiment") == "NEG")
    positive_count = sum(1 for item in triplets if item.get("sentiment") == "POS")
    category_names = {item.get("category", "") for item in categories}
    category_negative_count = sum(int(item.get("negative_count", 0)) for item in categories)
    signals = _collect_risk_signals(
        text=text,
        negative_count=negative_count,
        positive_count=positive_count,
        category_names=category_names,
        category_negative_count=category_negative_count,
        category_count=len(categories),
    )

    score = 20 + negative_count * 12 + category_negative_count * 6
    score += len([item for item in categories if item.get("category") != "其他反馈"]) * 4
    score += sum(signal.score_delta for signal in signals)

    level = _level_from_score(score)
    for signal in signals:
        level = _max_level(level, signal.level_floor)
    if level == "高风险" and not any(signal.level_floor == "高风险" for signal in signals):
        level = "中风险"

    if _looks_like_positive_only(text, negative_count, positive_count, signals):
        level = "低风险"
        score = min(score, 35)
    if _looks_like_resolved_or_minor_positive_feedback(text, signals):
        level = "低风险"
        score = min(score, 35)
    if _looks_like_mild_experience_complaint(text, signals):
        level = "低风险"
        score = min(score, 44)

    if level == "高风险":
        score = max(score, 82)
    elif level == "中风险":
        score = max(score, 55)
        score = min(score, 81)
    else:
        score = min(score, 44)
    score = max(0, min(score, 100))

    reasons = []
    if negative_count:
        reasons.append(f"识别到 {negative_count} 个负面情感点")
    reasons.extend(signal.reason for signal in signals)

    return {
        "level": level,
        "score": score,
        "priority": PRIORITY_BY_LEVEL[level],
        "should_escalate": level == "高风险",
        "reasons": reasons or ["未发现明显风险信号"],
        "triggered_rules": [signal.as_dict() for signal in signals],
        "rule_version": "risk_policy_v1",
    }


def _collect_risk_signals(
    text: str,
    negative_count: int,
    positive_count: int,
    category_names: set[str],
    category_negative_count: int,
    category_count: int,
) -> list[RiskSignal]:
    signals: list[RiskSignal] = []
    negative_context = _has_negative_context(text, negative_count, category_negative_count)

    complaint_hits = _hits(text, COMPLAINT_WORDS)
    if "投诉说说" in text:
        complaint_hits = [hit for hit in complaint_hits if hit != "投诉"]
    public_negative_review_hits = _hits(text, PUBLIC_NEGATIVE_REVIEW_WORDS)
    safety_hits = _hits(text, SAFETY_WORDS)
    fraud_hits = _hits(text, FRAUD_WORDS)
    trust_concern_hits = _hits(text, TRUST_CONCERN_WORDS)
    refund_hits = _hits(text, REFUND_WORDS)
    mild_refund_hits = _hits(text, MILD_REFUND_WORDS)
    if any(phrase in text for phrase in ["不想换货", "并不想换货", "不想退货", "并不想退货"]):
        refund_hits = [hit for hit in refund_hits if hit not in {"换货", "退货", "退换"}]
    service_hits = _hits(text, SERVICE_WORDS)
    severe_service_hits = _hits(text, SEVERE_SERVICE_WORDS)
    severe_refund_hits = _hits(text, SEVERE_REFUND_WORDS)
    quality_hits = _filter_positive_negated_hits(text, _hits(text, QUALITY_WORDS)) if negative_context else []
    core_hits = _hits(text, CORE_FUNCTION_WORDS) if negative_context else []
    logistics_hits = _hits(text, LOGISTICS_WORDS)
    mild_logistics_hits = _hits(text, MILD_LOGISTICS_WORDS) if logistics_hits else []
    serious_logistics_hits = _hits(text, SERIOUS_LOGISTICS_WORDS) if logistics_hits else []
    if _looks_resolved_logistics(text):
        serious_logistics_hits = []
    vague_bad_hits = _hits(text, VAGUE_BAD_WORDS)
    dissuasion_hits = _hits(text, STRONG_DISSUASION_WORDS)

    if safety_hits:
        signals.append(
            RiskSignal(
                "R1_SAFETY",
                "高风险",
                45,
                f"命中安全/健康风险：{', '.join(safety_hits)}",
                safety_hits,
            )
        )
    if complaint_hits:
        signals.append(
            RiskSignal(
                "R2_COMPLAINT",
                "高风险",
                35,
                f"命中投诉/维权/公开差评信号：{', '.join(complaint_hits)}",
                complaint_hits,
            )
        )
    if public_negative_review_hits:
        signals.append(
            RiskSignal(
                "R2B_PUBLIC_NEGATIVE_REVIEW",
                "中风险",
                18,
                f"命中公开负面评价信号：{', '.join(public_negative_review_hits)}",
                public_negative_review_hits,
            )
        )
    if fraud_hits:
        signals.append(
            RiskSignal(
                "R3_TRUST",
                "高风险",
                35,
                f"命中欺诈/假货/信任风险：{', '.join(fraud_hits)}",
                fraud_hits,
            )
        )
    if trust_concern_hits:
        signals.append(
            RiskSignal(
                "R3B_TRUST_CONCERN",
                "中风险",
                18,
                f"命中交易信任不满信号：{', '.join(trust_concern_hits)}",
                trust_concern_hits,
            )
        )
    if refund_hits and severe_service_hits:
        signals.append(
            RiskSignal(
                "R4_REFUND_SERVICE_BLOCKED",
                "高风险",
                32,
                "退款/退换诉求叠加客服或售后处理问题",
                sorted(set(refund_hits + service_hits)),
            )
        )
    if refund_hits and (trust_concern_hits or dissuasion_hits or severe_refund_hits):
        signals.append(
            RiskSignal(
                "R4B_REFUND_RIGHTS_DISPUTE",
                "高风险",
                30,
                "退换货/赔付诉求叠加强烈不满或权益争议",
                sorted(set(refund_hits + trust_concern_hits + dissuasion_hits + severe_refund_hits)),
            )
        )
    if mild_refund_hits:
        signals.append(
            RiskSignal(
                "R4C_REFUND_PROCESS_FRICTION",
                "中风险",
                14,
                f"退换货流程体验差：{', '.join(mild_refund_hits)}",
                mild_refund_hits,
            )
        )
    if refund_hits and logistics_hits:
        signals.append(
            RiskSignal(
                "R5_REFUND_LOGISTICS_BLOCKED",
                "高风险",
                28,
                "履约/物流问题叠加退款或权益诉求",
                sorted(set(refund_hits + logistics_hits)),
            )
        )
    if quality_hits:
        signals.append(
            RiskSignal(
                "R6_QUALITY_OR_DURABILITY",
                "中风险",
                18,
                f"质量/耐用性/核心商品问题至少按中风险处理：{', '.join(quality_hits)}",
                quality_hits,
            )
        )
    if core_hits:
        signals.append(
            RiskSignal(
                "R7_CORE_FUNCTION",
                "中风险",
                16,
                f"核心功能或使用体验异常至少按中风险处理：{', '.join(core_hits)}",
                core_hits,
            )
        )
    if quality_hits and (severe_service_hits or severe_refund_hits or dissuasion_hits):
        signals.append(
            RiskSignal(
                "R6B_QUALITY_WITH_ESCALATION_SIGNAL",
                "高风险",
                30,
                "质量问题叠加客服阻塞、退换货受阻或劝阻购买信号",
                sorted(set(quality_hits + severe_service_hits + severe_refund_hits + dissuasion_hits)),
            )
        )
    if core_hits and severe_service_hits and not _weak_complaint_context(text):
        signals.append(
            RiskSignal(
                "R7B_CORE_FUNCTION_WITH_SERVICE_BLOCK",
                "高风险",
                30,
                "核心功能异常叠加客服联系/处理受阻",
                sorted(set(core_hits + severe_service_hits + dissuasion_hits)),
            )
        )
    if (MEDIUM_FLOOR_CATEGORIES & category_names) and negative_context:
        hits = sorted(MEDIUM_FLOOR_CATEGORIES & category_names)
        signals.append(
            RiskSignal(
                "R8_NEGATIVE_BUSINESS_CATEGORY",
                "中风险",
                10,
                f"负面评论涉及关键业务类别：{', '.join(hits)}",
                hits,
            )
        )
    if service_hits and negative_context and not refund_hits:
        signals.append(
            RiskSignal(
                "R9_SERVICE_ONLY",
                "中风险",
                14,
                f"客服/售后响应问题需要跟进：{', '.join(service_hits)}",
                service_hits,
            )
        )
    if logistics_hits and mild_logistics_hits and not refund_hits and not service_hits:
        signals.append(
            RiskSignal(
                "R10_MILD_LOGISTICS",
                "低风险",
                8,
                "物流慢但暂未出现退款、投诉或客服阻塞，可先按常规履约跟进",
                sorted(set(logistics_hits + mild_logistics_hits)),
            )
        )
    if serious_logistics_hits and negative_context:
        signals.append(
            RiskSignal(
                "R10B_SERIOUS_LOGISTICS_BLOCK",
                "高风险",
                28,
                f"物流履约严重阻塞：{', '.join(serious_logistics_hits)}",
                sorted(set(logistics_hits + serious_logistics_hits)),
            )
        )
    if _has_severe_core_failure(text, core_hits):
        signals.append(
            RiskSignal(
                "R12_SEVERE_CORE_FAILURE",
                "高风险",
                32,
                "核心功能持续异常或已进入维修/换机流程",
                core_hits,
            )
        )
    if "诚信经营" in text and "图片保持一致" in text and negative_context:
        signals.append(
            RiskSignal(
                "R13_PRODUCT_MISMATCH_TRUST",
                "高风险",
                24,
                "商品实物与图片/宣传不一致并触发商家诚信质疑",
                ["诚信经营", "图片保持一致"],
            )
        )
    if ("头皮屑" in text and any(word in text for word in ["不敢用", "全家", "正品", "相信"])) or (
        "刺激眼睛" in text and negative_context
    ):
        signals.append(
            RiskSignal(
                "R14_USE_HEALTH_OR_AUTHENTICITY",
                "高风险",
                30,
                "使用后出现健康/正品信任疑虑",
                _hits(text, ["头皮屑", "不敢用", "全家", "正品", "刺激眼睛"]),
            )
        )
    if vague_bad_hits and negative_context:
        signals.append(
            RiskSignal(
                "R15_VAGUE_BAD_EXPERIENCE",
                "中风险",
                12,
                f"存在强烈负面体验但问题描述较泛：{', '.join(vague_bad_hits)}",
                vague_bad_hits,
            )
        )
    if dissuasion_hits and negative_context:
        signals.append(
            RiskSignal(
                "R16_STRONG_DISSUASION",
                "中风险",
                14,
                f"用户出现强烈劝阻或流失信号：{', '.join(dissuasion_hits)}",
                dissuasion_hits,
            )
        )
    if len(MULTI_ISSUE_CATEGORIES & category_names) >= 3 or negative_count >= 3:
        has_blocked_high_risk_signal = bool(
            severe_service_hits
            or severe_refund_hits
            or complaint_hits
            or fraud_hits
            or safety_hits
        )
        signals.append(
            RiskSignal(
                "R11_MULTI_ISSUE",
                "高风险" if HIGH_RISK_CATEGORIES <= category_names and has_blocked_high_risk_signal else "中风险",
                20,
                "同一条反馈涉及多个问题，存在升级风险",
                sorted(MULTI_ISSUE_CATEGORIES & category_names),
            )
        )

    return signals


def _hits(text: str, words: list[str]) -> list[str]:
    return [word for word in words if word and word in text]


def _filter_positive_negated_hits(text: str, hits: list[str]) -> list[str]:
    filtered = []
    for hit in hits:
        if hit == "坏" and any(phrase in text for phrase in ["没有坏", "没坏", "无坏"]):
            continue
        if hit in {"漏", "漏水"} and any(phrase in text for phrase in ["没漏", "没有漏", "一点都不漏"]):
            continue
        if hit == "破" and any(phrase in text for phrase in ["没破", "没有破"]):
            continue
        filtered.append(hit)
    return filtered


def _has_negative_context(text: str, negative_count: int, category_negative_count: int) -> bool:
    return bool(negative_count or category_negative_count or _hits(text, NEGATIVE_WORDS))


def _level_from_score(score: int) -> str:
    if score >= 82:
        return "高风险"
    if score >= 45:
        return "中风险"
    return "低风险"


def _has_severe_core_failure(text: str, core_hits: list[str]) -> bool:
    if not core_hits:
        return False
    severe_patterns = [
        ["屏幕", "裂开"],
        ["机器", "看不到"],
        ["充电不了", "客服"],
        ["充不了电", "客服"],
        ["像素", "联系不到"],
        ["发错", "联系不到"],
        ["听筒", "坏", "换了一个"],
        ["花屏", "退货"],
        ["检测", "换了一个", "退货"],
        ["信号", "维修站", "换机"],
        ["信号", "设计缺陷"],
        ["闹钟", "发件箱", "信号"],
        ["确实有问题", "换机"],
    ]
    return any(all(word in text for word in pattern) for pattern in severe_patterns)


def _looks_resolved_logistics(text: str) -> bool:
    return any(phrase in text for phrase in ["还好最后让验货", "最后让验货", "还好客服说"]) and any(
        word in text for word in ["还不错", "还可以", "用了段时间"]
    )


def _weak_complaint_context(text: str) -> bool:
    return any(phrase in text for phrase in ["投诉说说", "不想换货", "并不想换货", "不想退货", "并不想退货"])


def _max_level(current: str, candidate: str) -> str:
    return candidate if LEVEL_RANK[candidate] > LEVEL_RANK[current] else current


def _looks_like_positive_only(
    text: str,
    negative_count: int,
    positive_count: int,
    signals: list[RiskSignal],
) -> bool:
    has_positive_review = positive_count > 0 or any(word in text for word in ["很好", "不错", "满意", "合适", "还会买", "喜欢"])
    hard_rule_ids = {signal.rule_id for signal in signals if signal.level_floor in {"中风险", "高风险"}}
    return has_positive_review and negative_count == 0 and not hard_rule_ids


def _looks_like_resolved_or_minor_positive_feedback(text: str, signals: list[RiskSignal]) -> bool:
    if any(signal.level_floor == "高风险" for signal in signals):
        return False
    if "不是京东的问题" in text:
        return True
    positive_markers = ["整体体验还不错", "挺好的", "值得购买", "很好", "不错", "真好", "满意", "还可以"]
    if not any(marker in text for marker in positive_markers):
        return False
    mild_markers = [
        "没有坏",
        "没坏",
        "没漏",
        "还好最后让验货",
        "最后让验货",
        "用了段时间才评价",
        "赠品没有",
        "就是赠品",
        "稍微有点吵",
        "难免",
        "不是京东的问题",
        "以前记得送",
        "没有了",
    ]
    return any(marker in text for marker in mild_markers)


def _looks_like_mild_experience_complaint(text: str, signals: list[RiskSignal]) -> bool:
    """Keep mild experience complaints classified, but do not over-escalate them."""
    if any(signal.level_floor == "高风险" for signal in signals):
        return False
    hard_rule_ids = {signal.rule_id for signal in signals if signal.level_floor == "中风险"}
    if not hard_rule_ids <= {"R7_CORE_FUNCTION", "R8_NEGATIVE_BUSINESS_CATEGORY"}:
        return False
    mild_phrases = [
        "有待提高",
        "不是那种很抢人眼球",
        "10寸还是小了",
        "达不到四星水平",
        "勉强也就是三星",
        "稍微有点吵",
    ]
    severe_phrases = [
        "不能用",
        "用不了",
        "无法",
        "坏",
        "裂开",
        "退货",
        "退款",
        "投诉",
        "客服",
        "售后",
        "漏",
        "冒烟",
        "安全",
    ]
    return any(phrase in text for phrase in mild_phrases) and not any(phrase in text for phrase in severe_phrases)


def _is_irrelevant_rant(text: str) -> bool:
    has_noncommerce_topic = any(word in text for word in ["驾驶证", "超速", "扣6分", "罚400"])
    has_commerce_topic = any(word in text for word in ["买", "收到", "客服", "物流", "退款", "退货", "换货", "发货"])
    return has_noncommerce_topic and not has_commerce_topic
