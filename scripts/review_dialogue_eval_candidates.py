"""Fill human-reviewed labels for the 50 synthetic dialogue candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "dialogue_eval_candidates_50.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "dialogue_eval_candidates_50_reviewed.csv"

LABELS = {
    "D001": ("其他反馈", "低风险", "P3", "正向使用反馈，无售后纠纷或风险信号"),
    "D002": ("商品质量问题;退换货与赔付问题;价格与权益问题", "高风险", "P1", "用户称产品有问题并涉及退货运费争议，同时出现强烈信任不满"),
    "D003": ("其他反馈", "低风险", "P3", "正向评价，未发现售后风险"),
    "D004": ("价格与权益问题", "中风险", "P2", "多次购买未开发票，属于发票/权益问题，需要客服跟进"),
    "D005": ("其他反馈", "低风险", "P3", "正向评价，明确表示没有坏果和送货快"),
    "D006": ("商品质量问题;退换货与赔付问题", "中风险", "P2", "包装泄漏并已联系换货，属于质量与退换货问题，但暂无投诉升级"),
    "D007": ("客服响应问题", "中风险", "P2", "明确反馈电话客服服务差，需要服务改进"),
    "D008": ("商品质量问题;退换货与赔付问题", "高风险", "P1", "衣服质量差且不能换，用户劝阻他人购买，存在纠纷升级风险"),
    "D009": ("其他反馈", "低风险", "P3", "正向评价，未发现售后风险"),
    "D010": ("商品质量问题;产品体验问题;客服响应问题", "高风险", "P1", "平板充不了电且客服敷衍，核心功能异常叠加客服处理不佳"),
    "D011": ("商品质量问题;产品体验问题;退换货与赔付问题", "高风险", "P1", "短期内两台设备连续故障并退货，属于高优先级质量/退货问题"),
    "D012": ("商品质量问题;产品体验问题;退换货与赔付问题", "高风险", "P1", "连续多次换货仍有划痕/摄像头问题，最终退货，纠纷风险高"),
    "D013": ("物流问题;客服响应问题", "中风险", "P2", "物流慢且客服态度差并出现差评，但暂无退款/投诉升级"),
    "D014": ("商品质量问题", "中风险", "P2", "明确质量差和材质差，需要售后跟进"),
    "D015": ("其他反馈", "低风险", "P3", "正向评价，未发现售后风险"),
    "D016": ("商品质量问题;客服响应问题", "高风险", "P1", "衣服短期破裂开裂且客服联系不理，质量问题叠加客服不处理"),
    "D017": ("其他反馈", "低风险", "P3", "最终用户表达合身、质量不错、满意，按正向反馈处理"),
    "D018": ("其他反馈", "低风险", "P3", "正向评价，包装、售后和苹果质量均被肯定"),
    "D019": ("其他反馈", "低风险", "P3", "正向评价，质量、物流、快递服务均较好"),
    "D020": ("产品体验问题;商品质量问题;客服响应问题", "高风险", "P1", "像素/发错货疑虑叠加联系不到客服，存在升级风险"),
    "D021": ("其他反馈", "低风险", "P3", "正向评价，质量、物流、客服均较好"),
    "D022": ("退换货与赔付问题;价格与权益问题", "中风险", "P2", "用户抱怨电子产品退换货流程费劲，属于权益与退换货体验问题"),
    "D023": ("其他反馈", "低风险", "P3", "正向评价，快递辛苦且水果没坏"),
    "D024": ("商品质量问题;价格与权益问题", "中风险", "P2", "质量不好且认为价格不值，属于质量和价格感知问题"),
    "D025": ("产品体验问题;客服响应问题;退换货与赔付问题", "高风险", "P1", "摄像头问题、换货流程慢、服务态度差叠加，需优先处理"),
    "D026": ("客服响应问题;产品体验问题", "中风险", "P2", "酒店前台服务变差、入住办理慢，属于服务体验问题"),
    "D027": ("其他反馈", "低风险", "P3", "书籍内容不符合个人喜好，暂无售后纠纷"),
    "D028": ("其他反馈", "低风险", "P3", "正向评价，未发现售后风险"),
    "D029": ("产品体验问题", "中风险", "P2", "洗发水效果差并表示不会再买，属于产品体验负面"),
    "D030": ("产品体验问题", "中风险", "P2", "酒店房间烟味油漆味和噪音影响体验，但暂无投诉/退款"),
    "D031": ("其他反馈;价格与权益问题", "低风险", "P3", "宽屏不习惯且赠品变化，用户承认不是京东问题，风险较低"),
    "D032": ("商品质量问题", "中风险", "P2", "水果太熟且不能吃，属于生鲜质量问题"),
    "D033": ("商品质量问题", "中风险", "P2", "衣服布料差且穿一天开裂，属于明显质量问题"),
    "D034": ("产品体验问题", "中风险", "P2", "酒店硬件和软件体验均差，属于产品/服务体验问题"),
    "D035": ("商品质量问题", "中风险", "P2", "水果品质差且用户质疑自营把关水平"),
    "D036": ("其他反馈", "低风险", "P3", "内容更像品牌信任吐槽，没有明确订单或售后诉求"),
    "D037": ("商品质量问题", "中风险", "P2", "洗发水包装泄漏三分之一，属于包装/质量问题"),
    "D038": ("产品体验问题;价格与权益问题", "中风险", "P2", "配件缺失和散热影响使用，属于体验与权益问题"),
    "D039": ("商品质量问题;价格与权益问题", "中风险", "P2", "水果个头小、味道差且性价比差，属于质量和价格问题"),
    "D040": ("其他反馈", "低风险", "P3", "正向评价，未发现售后风险"),
    "D041": ("商品质量问题;产品体验问题", "中风险", "P2", "发错/搞错一件且舒适感差，属于质量和体验问题"),
    "D042": ("价格与权益问题;产品体验问题", "中风险", "P2", "配件/赠品缺失导致失望，属于权益与体验问题"),
    "D043": ("产品体验问题", "低风险", "P3", "酒店地段好但稍吵，属于轻微体验反馈"),
    "D044": ("其他反馈", "低风险", "P3", "中性功能描述，无售后风险"),
    "D045": ("价格与权益问题", "低风险", "P3", "整体评价正向，仅赠品缺失，按低风险权益反馈处理"),
    "D046": ("其他反馈", "低风险", "P3", "正向评价，未发现售后风险"),
    "D047": ("其他反馈", "低风险", "P3", "正向评价，未发现售后风险"),
    "D048": ("其他反馈", "低风险", "P3", "正向书籍评价，无售后风险"),
    "D049": ("其他反馈", "低风险", "P3", "正向评价，到货快且还不错"),
    "D050": ("其他反馈", "低风险", "P3", "正向评价，便宜、包装好、无泄漏且满意"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review synthetic dialogue candidates.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    missing = [row["candidate_id"] for row in rows if row["candidate_id"] not in LABELS]
    if missing:
        raise SystemExit(f"Missing review labels for: {', '.join(missing)}")

    for row in rows:
        categories, risk_level, priority, reason = LABELS[row["candidate_id"]]
        row["human_expected_categories"] = categories
        row["human_expected_risk_level"] = risk_level
        row["human_expected_priority"] = priority
        row["human_rule_reason"] = reason
        row["review_status"] = "reviewed"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"input_file={input_path}")
    print(f"output_file={output_path}")
    print(f"reviewed_count={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
