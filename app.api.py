# -*- coding: utf-8 -*-
"""
AI饮食健康智能推荐系统 · 单文件版
===================================
面向大学生食堂场景：拍照识别食物 → 营养分析 → 预警与建议 → 食堂搭配推荐 → 每周报告。

运行方式：
    pip install -r requirements.txt
    streamlit run diet_ai_app.py
启动后会自动打开浏览器（Streamlit 默认行为）。

功能目录（页面 Tabs）：
    1. 📸 饮食记录：拍照/上传 → AI 识别 + 营养分析 + 预警 + 白话建议，保存到今日记录
    2. 📊 营养统计：今日累计营养 vs 个人目标，各餐分布
    3. 🍱 食堂推荐：按身体画像与目标推荐三餐搭配
    4. 📄 每周报告：近 7 天记录聚合 + 趋势图 + 文字总结

识别 API 配置（二选一）：
    · 豆包视觉大模型（火山方舟）：设置环境变量 VISION_API_KEY / VISION_MODEL，
      或在页面侧边栏直接填入 Key
    · 任意自定义 HTTP 识别服务：设置 FOOD_API_URL，接口返回
      {"foods": [{"name": "米饭", "amount_grams": 200, "confidence": 0.95}]}

⚠️ 营养数据口径：以《中国食物成分表（第6版）》常见数值为参考基准整理，
   食堂成品菜为估算值，正式上线建议对接权威数据库。
"""
import base64
import datetime as dt
import json
import os
from typing import Dict, List

import httpx
import pandas as pd
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# 一、配置
# ══════════════════════════════════════════════════════════════════
RECORD_FILE = "diet_records.json"          # 饮食记录持久化文件（与脚本同目录）
NUTRITION_CACHE_FILE = "nutrition_cache.json"  # API 营养补全缓存（同种食物只调一次 API）
VISION_API_KEY = os.getenv("VISION_API_KEY", "")   # 豆包视觉大模型 Key
VISION_API_BASE = os.getenv("VISION_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
VISION_MODEL = os.getenv("VISION_MODEL", "ep-20261028170459-nb7px")
CUSTOM_FOOD_API_URL = os.getenv("CUSTOM_FOOD_API_URL", "")   # 自定义识别服务


def _is_deep_think_model(model: str) -> bool:
    """判断是否为深度思考模型：seed 系列模型名，或 ep- 开头的推理接入点
    （接入点绑定的模型用户自己知道，默认按深度思考处理，结构化输出时关思考更稳更省）。
    若你的接入点绑定的是非思考模型且调用报"参数不支持"，把下面 return 改为 return "seed" in m。
    """
    m = model.lower()
    return "seed" in m or m.startswith("ep-")

# ══════════════════════════════════════════════════════════════════
# 二、营养数据库（每 100g 可食部）
# ══════════════════════════════════════════════════════════════════
FOOD_NUTRITION: Dict[str, Dict[str, float]] = {
    # 主食
    "米饭":     {"kcal": 116, "protein": 2.6, "fat": 0.3, "carb": 25.9, "fiber": 0.3, "sodium": 2.5},
    "馒头":     {"kcal": 223, "protein": 7.0, "fat": 1.1, "carb": 47.0, "fiber": 1.3, "sodium": 165.0},
    "面条":     {"kcal": 110, "protein": 3.9, "fat": 0.4, "carb": 22.8, "fiber": 0.6, "sodium": 120.0},
    "包子":     {"kcal": 227, "protein": 7.4, "fat": 6.2, "carb": 35.0, "fiber": 1.2, "sodium": 320.0},
    "饺子":     {"kcal": 240, "protein": 8.5, "fat": 9.0, "carb": 30.0, "fiber": 1.5, "sodium": 420.0},
    "粥":       {"kcal": 46,  "protein": 1.1, "fat": 0.3, "carb": 9.9,  "fiber": 0.1, "sodium": 2.0},
    "玉米":     {"kcal": 112, "protein": 4.0, "fat": 1.2, "carb": 22.8, "fiber": 2.9, "sodium": 1.1},
    "红薯":     {"kcal": 86,  "protein": 1.6, "fat": 0.1, "carb": 20.1, "fiber": 1.6, "sodium": 28.0},
    "土豆":     {"kcal": 81,  "protein": 2.6, "fat": 0.2, "carb": 17.2, "fiber": 1.1, "sodium": 2.7},
    # 蛋奶豆
    "鸡蛋":     {"kcal": 144, "protein": 13.1, "fat": 9.5, "carb": 2.1,  "fiber": 0.0, "sodium": 131.0},
    "牛奶":     {"kcal": 54,  "protein": 3.0, "fat": 3.2, "carb": 3.4,  "fiber": 0.0, "sodium": 37.0},
    "酸奶":     {"kcal": 72,  "protein": 2.5, "fat": 2.7, "carb": 9.3,  "fiber": 0.0, "sodium": 39.0},
    "豆腐":     {"kcal": 84,  "protein": 8.1, "fat": 3.7, "carb": 4.2,  "fiber": 0.4, "sodium": 7.0},
    "豆浆":     {"kcal": 31,  "protein": 3.0, "fat": 1.6, "carb": 1.2,  "fiber": 0.1, "sodium": 3.0},
    # 肉禽鱼
    "鸡胸肉":   {"kcal": 118, "protein": 24.6, "fat": 1.9, "carb": 0.6,  "fiber": 0.0, "sodium": 63.0},
    "鸡腿":     {"kcal": 181, "protein": 16.0, "fat": 13.0, "carb": 0.0, "fiber": 0.0, "sodium": 64.0},
    "瘦猪肉":   {"kcal": 143, "protein": 20.3, "fat": 6.2, "carb": 1.5,  "fiber": 0.0, "sodium": 57.0},
    "红烧肉":   {"kcal": 478, "protein": 11.5, "fat": 45.0, "carb": 8.0,  "fiber": 0.0, "sodium": 620.0},
    "瘦牛肉":   {"kcal": 106, "protein": 20.2, "fat": 2.3, "carb": 1.2,  "fiber": 0.0, "sodium": 53.0},
    "鱼肉":     {"kcal": 105, "protein": 18.0, "fat": 3.0, "carb": 0.5,  "fiber": 0.0, "sodium": 60.0},
    "虾":       {"kcal": 93,  "protein": 18.6, "fat": 0.8, "carb": 2.8,  "fiber": 0.0, "sodium": 165.0},
    # 蔬菜
    "西兰花":   {"kcal": 36,  "protein": 4.1, "fat": 0.6, "carb": 4.3,  "fiber": 1.6, "sodium": 18.0},
    "青菜":     {"kcal": 17,  "protein": 1.6, "fat": 0.2, "carb": 2.6,  "fiber": 1.1, "sodium": 73.0},
    "白菜":     {"kcal": 18,  "protein": 1.5, "fat": 0.1, "carb": 3.2,  "fiber": 0.8, "sodium": 57.0},
    "西红柿":   {"kcal": 20,  "protein": 0.9, "fat": 0.2, "carb": 4.0,  "fiber": 0.5, "sodium": 5.0},
    "黄瓜":     {"kcal": 16,  "protein": 0.8, "fat": 0.2, "carb": 2.9,  "fiber": 0.5, "sodium": 4.9},
    "胡萝卜":   {"kcal": 39,  "protein": 1.0, "fat": 0.2, "carb": 8.8,  "fiber": 1.1, "sodium": 71.0},
    "茄子":     {"kcal": 23,  "protein": 1.1, "fat": 0.2, "carb": 4.9,  "fiber": 1.3, "sodium": 5.0},
    "蘑菇":     {"kcal": 24,  "protein": 2.7, "fat": 0.1, "carb": 4.1,  "fiber": 2.1, "sodium": 8.0},
    # 食堂常见成品菜（估算）
    "西红柿炒蛋": {"kcal": 120, "protein": 6.0, "fat": 8.0, "carb": 5.0,  "fiber": 0.8, "sodium": 380.0},
    "青椒肉丝":  {"kcal": 150, "protein": 9.0, "fat": 9.0, "carb": 8.0,  "fiber": 1.2, "sodium": 500.0},
    "宫保鸡丁":  {"kcal": 190, "protein": 12.0, "fat": 11.0, "carb": 12.0, "fiber": 1.0, "sodium": 520.0},
    "鱼香肉丝":  {"kcal": 165, "protein": 9.5, "fat": 10.0, "carb": 10.0, "fiber": 1.2, "sodium": 560.0},
    "麻婆豆腐":  {"kcal": 130, "protein": 8.0, "fat": 9.0, "carb": 6.0,  "fiber": 1.0, "sodium": 700.0},
    "油炸鸡排":  {"kcal": 260, "protein": 15.0, "fat": 18.0, "carb": 12.0, "fiber": 0.5, "sodium": 550.0},
    "炸薯条":    {"kcal": 298, "protein": 3.9, "fat": 15.0, "carb": 38.0, "fiber": 3.0, "sodium": 250.0},
    "汉堡":      {"kcal": 300, "protein": 13.0, "fat": 15.0, "carb": 30.0, "fiber": 1.0, "sodium": 480.0},
    # 水果
    "苹果":     {"kcal": 53,  "protein": 0.4, "fat": 0.2, "carb": 13.7, "fiber": 1.7, "sodium": 1.6},
    "香蕉":     {"kcal": 93,  "protein": 1.4, "fat": 0.2, "carb": 22.0, "fiber": 1.2, "sodium": 0.8},
    "橘子":     {"kcal": 44,  "protein": 0.8, "fat": 0.1, "carb": 10.2, "fiber": 0.5, "sodium": 0.0},
}


def get_nutrition(name: str):
    """按名称查找营养；找不到时模糊匹配（去掉量词/做法词）。"""
    if name in FOOD_NUTRITION:
        return FOOD_NUTRITION[name]
    cleaned = (name.replace("一份", "").replace("一盘", "").replace("一碗", "")
               .replace("大份", "").replace("小份", "")
               .replace("炒", "").replace("蒸", "").replace("煮", "").strip())
    for key, value in FOOD_NUTRITION.items():
        if key in cleaned or cleaned in key:
            return value
    return None


# ══════════════════════════════════════════════════════════════════
# 三、AI 食物识别（外部视觉 API，可插拔）
# ══════════════════════════════════════════════════════════════════
_PROMPT = ("你是食物营养识别专家。请识别图片中的所有食物，并估算每份食物的重量（克）。"
           "要求：1. 只输出 JSON：[{\"name\": \"食物名称\", \"amount_grams\": 250, \"confidence\": 0.95}]"
           "；2. 名称用中文，如\"米饭\"、\"西红柿炒蛋\"；"
           "3. 按餐盘占比估算克数（一碗米饭约200-250g，一份炒菜约150-200g，一个鸡蛋约50g）；"
           "4. confidence 为置信度 0-1；5. 不要输出其他任何文字。")


def recognize_food(image_bytes: bytes, api_key: str) -> List[dict]:
    """调用视觉 API 识别食物。返回 [{"name","amount_grams","confidence"}, ...]"""
    if CUSTOM_FOOD_API_URL:
        resp = httpx.post(CUSTOM_FOOD_API_URL,
                          files={"file": ("food.jpg", image_bytes, "image/jpeg")}, timeout=60)
        resp.raise_for_status()
        return resp.json().get("foods", [])
    if not api_key:
        raise ValueError("未配置识别 API Key：请在侧边栏填写，或设置环境变量 VISION_API_KEY")
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "temperature": 0.2,
    }
    if _is_deep_think_model(VISION_MODEL):
        payload["thinking"] = {"type": "disabled"}   # 深度思考模型：关思考，直出 JSON，快且省 token
    resp = httpx.post(f"{VISION_API_BASE}/chat/completions",
                      headers={"Authorization": f"Bearer {api_key}"},
                      json=payload, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = content.find("["), content.rfind("]") + 1
    data = json.loads(content[start:end])
    return [{"name": str(x.get("name", "")).strip(),
             "amount_grams": float(x.get("amount_grams", 100)),
             "confidence": float(x.get("confidence", 0.0))}
            for x in data if str(x.get("name", "")).strip()]


# ── 未匹配食物营养补全（调用豆包文本模型，按每100g估算营养） ─────
_FILL_PROMPT = """你是中国食物成分领域的营养专家。请为下列每种食物给出营养成分，按【每100克可食部】估算。
只输出一个JSON对象，键为食物原名，值为营养字段，必须包含：
kcal(热量,千卡), protein(蛋白质,克), fat(脂肪,克), carb(碳水化合物,克), fiber(膳食纤维,克), sodium(钠,毫克)
并在能力范围内尽量补充：sugar_g(糖,克), cholesterol_mg(胆固醇,毫克), vitamin_a_ug(维生素A,微克RAE),
vitamin_c_mg(维生素C,毫克), calcium_mg(钙,毫克), iron_mg(铁,毫克)。
数值参考《中国食物成分表》常识口径，必须合理，未知食物按同类食材估算。不要输出除JSON外的任何文字。
食物列表：{names}"""


def _load_json_file(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_json_file(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def complete_nutrition_via_api(names: List[str], api_key: str) -> Dict[str, dict]:
    """对内置库未匹配的食物名，调用豆包模型补全营养（每100g）。返回 {name: {字段: 值}}

    带本地缓存：同一食物只调用一次 API，结果持久化到 nutrition_cache.json，
    避免重复扣费、加快后续分析。
    """
    names = [n for n in names if n]
    if not names:
        return {}
    # 先查缓存
    cache = _load_json_file(NUTRITION_CACHE_FILE)
    result = {n: cache[n] for n in names if n in cache}
    missing = [n for n in names if n not in cache]
    if not missing:
        return result

    if CUSTOM_FOOD_API_URL:
        raise ValueError("自定义识别服务不支持营养补全，请改用豆包模型或配置 VISION_API_KEY")
    if not api_key:
        raise ValueError("未配置 API Key，无法补全营养")
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user",
                      "content": _FILL_PROMPT.replace("{names}", "、".join(missing))}],
        "temperature": 0.1,
    }
    if _is_deep_think_model(VISION_MODEL):
        payload["thinking"] = {"type": "disabled"}   # 深度思考模型：关思考直出 JSON
    resp = httpx.post(f"{VISION_API_BASE}/chat/completions",
                      headers={"Authorization": f"Bearer {api_key}"},
                      json=payload, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = content.find("{"), content.rfind("}") + 1
    data = json.loads(content[start:end])
    for name, nut in data.items():
        if not isinstance(nut, dict):
            continue
        cleaned = {k: float(v) for k, v in nut.items() if isinstance(v, (int, float))}
        if "kcal" in cleaned:   # 至少要有热量才算有效补全
            result[str(name).strip()] = cleaned
            cache[str(name).strip()] = cleaned
    _save_json_file(NUTRITION_CACHE_FILE, cache)
    return result


# ══════════════════════════════════════════════════════════════════
# 四、营养分析引擎
# ══════════════════════════════════════════════════════════════════
# 核心字段（用于预警/目标对比）
CORE_FIELDS = ("kcal", "protein", "fat", "carb", "fiber", "sodium")
# 扩展字段中文名（API 补全时尽可能给出）
FIELD_LABELS = {
    "kcal": "热量(kcal)", "protein": "蛋白质(g)", "fat": "脂肪(g)", "carb": "碳水(g)",
    "fiber": "膳食纤维(g)", "sodium": "钠(mg)", "sugar_g": "糖(g)",
    "cholesterol_mg": "胆固醇(mg)", "vitamin_a_ug": "维生素A(μg)", "vitamin_c_mg": "维生素C(mg)",
    "calcium_mg": "钙(mg)", "iron_mg": "铁(mg)",
}


def analyze_items(items: List[dict], extra_nutrition: dict = None):
    """items: [{"name","amount_grams","confidence"}] → 明细 + 合计 + 预警 + 建议

    extra_nutrition: 内置库未匹配食物由 API 补全的营养表 {name: {字段: 每100g值}}，
    查找优先级：extra 补全表 → 内置库。
    """
    extra_nutrition = extra_nutrition or {}
    totals = {}
    details, unknown = [], []
    for it in items:
        per100 = extra_nutrition.get(it["name"]) or get_nutrition(it["name"])
        if per100 is None:
            unknown.append(it["name"])
            continue
        ratio = it["amount_grams"] / 100.0
        nut = {k: round(v * ratio, 1) for k, v in per100.items() if k in FIELD_LABELS or k in CORE_FIELDS}
        for k in nut:
            totals[k] = totals.get(k, 0.0) + nut[k]
        details.append({"name": it["name"], "amount_grams": it["amount_grams"],
                        "confidence": it.get("confidence", 0.0), **nut})
    for k in CORE_FIELDS:
        totals.setdefault(k, 0.0)
    totals = {k: round(v, 1) for k, v in totals.items()}

    warnings = []
    if totals["kcal"] > 900:
        warnings.append(f"本餐约 {totals['kcal']:.0f} kcal，偏高（参考上限 900），注意控制份量")
    if totals["fat"] > 31:
        warnings.append(f"脂肪约 {totals['fat']:.0f} g，接近一天参考量的 45%，警惕高油（油炸/肥肉/重油炒菜）")
    if totals["sodium"] > 1200:
        warnings.append(f"钠约 {totals['sodium']:.0f} mg，占每日参考上限 60%+，重口/腌制/卤味偏多")
    if len(items) <= 1:
        warnings.append("本餐种类偏单一，建议增加蔬菜和蛋白质来源")

    tips = []
    if totals["protein"] < 20:
        tips.append("蛋白质偏少：加一份鸡蛋/鸡胸肉/豆腐/鱼肉（约一掌大小）更扛饿")
    elif totals["protein"] >= 35:
        tips.append("蛋白质充足，注意搭配粗粮和蔬菜帮助消化")
    if totals["fiber"] < 5:
        tips.append("膳食纤维偏低：这顿缺绿叶菜，下一餐补一份青菜或西兰花")
    if totals["carb"] < 60:
        tips.append("主食偏少，如果下午有课/运动容易饿，可加半份米饭或红薯")
    elif totals["carb"] > 120:
        tips.append("碳水偏多：主食减半，用蔬菜把胃填满")
    if unknown:
        tips.append(f"未匹配营养数据的食物：{'、'.join(unknown[:3])}，可在表格中改为相近食物")
    if not tips:
        tips.append("本餐搭配不错！保持「一主食 + 一荤 + 两素」的结构")
    return details, totals, warnings, tips


# ══════════════════════════════════════════════════════════════════
# 五、能量目标 + 食堂搭配推荐
# ══════════════════════════════════════════════════════════════════
ACTIVITY_FACTOR = {"久坐（少运动）": 1.2, "轻度（每周1-3次运动）": 1.375,
                   "中度（每周3-5次运动）": 1.55, "高强度（每周6-7次运动）": 1.725}
GOAL_ADJUST = {"均衡饮食": 0, "减脂": -350, "增肌": 300}
MEAL_SPLIT = {"早餐": 0.30, "午餐": 0.40, "晚餐": 0.30}

CANTEEN_MENU: List[dict] = [
    # 早餐
    {"name": "鸡蛋+小米粥+包子", "meal": "早餐", "window": "早点窗口",
     "kcal": 420, "protein": 16, "fat": 12, "carb": 60, "tags": "均衡", "price": "约6元"},
    {"name": "豆浆+油条+茶叶蛋", "meal": "早餐", "window": "早点窗口",
     "kcal": 480, "protein": 13, "fat": 20, "carb": 62, "tags": "高油", "price": "约5.5元"},
    {"name": "燕麦粥+鸡蛋+牛奶", "meal": "早餐", "window": "早点窗口",
     "kcal": 360, "protein": 18, "fat": 10, "carb": 50, "tags": "清淡·高蛋白", "price": "约7元"},
    {"name": "酱香饼+豆浆", "meal": "早餐", "window": "早点窗口",
     "kcal": 450, "protein": 10, "fat": 16, "carb": 66, "tags": "高油", "price": "约5元"},
    # 午/晚餐
    {"name": "自选菜：米饭+鸡胸肉+西兰花+西红柿炒蛋", "meal": "午餐", "window": "大众自选窗口",
     "kcal": 620, "protein": 38, "fat": 16, "carb": 78, "tags": "高蛋白·均衡", "price": "约12元"},
    {"name": "盖浇饭：鱼香肉丝盖饭", "meal": "午餐", "window": "盖浇饭窗口",
     "kcal": 780, "protein": 24, "fat": 26, "carb": 110, "tags": "高油·高盐", "price": "约13元"},
    {"name": "麻辣烫（多蔬菜+豆制品，少麻酱）", "meal": "午餐", "window": "麻辣烫窗口",
     "kcal": 520, "protein": 22, "fat": 14, "carb": 60, "tags": "均衡", "price": "约15元"},
    {"name": "牛肉拉面+卤蛋", "meal": "午餐", "window": "面食窗口",
     "kcal": 650, "protein": 26, "fat": 14, "carb": 100, "tags": "均衡", "price": "约11元"},
    {"name": "鸡排饭（油炸鸡排+米饭+配菜）", "meal": "午餐", "window": "风味窗口",
     "kcal": 820, "protein": 30, "fat": 32, "carb": 100, "tags": "高油", "price": "约14元"},
    {"name": "自选菜：米饭+瘦牛肉+青菜+豆腐", "meal": "晚餐", "window": "大众自选窗口",
     "kcal": 560, "protein": 34, "fat": 14, "carb": 72, "tags": "高蛋白·均衡", "price": "约12元"},
    {"name": "玉米+鸡腿+青菜", "meal": "晚餐", "window": "杂粮窗口",
     "kcal": 480, "protein": 26, "fat": 16, "carb": 58, "tags": "均衡·高纤维", "price": "约10元"},
    {"name": "炒面+煎蛋", "meal": "晚餐", "window": "面食窗口",
     "kcal": 720, "protein": 20, "fat": 26, "carb": 102, "tags": "高油", "price": "约9元"},
    {"name": "粥+蒸饺+小菜", "meal": "晚餐", "window": "粥粉窗口",
     "kcal": 430, "protein": 14, "fat": 12, "carb": 66, "tags": "清淡", "price": "约8元"},
]


def calc_target(gender: str, age: int, height_cm: float, weight_kg: float,
                activity: str, goal: str) -> dict:
    """Mifflin-St Jeor 计算 BMR → TDEE → 按目标调整。"""
    bmr = (10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if gender == "男" else -161))
    kcal = max(bmr * ACTIVITY_FACTOR[activity] + GOAL_ADJUST[goal], 1200.0)
    p_ratio, f_ratio = (0.25, 0.25) if goal != "均衡饮食" else (0.20, 0.25)
    return {"kcal": round(kcal), "protein_g": round(kcal * p_ratio / 4),
            "fat_g": round(kcal * f_ratio / 9),
            "carb_g": round(kcal * (1 - p_ratio - f_ratio) / 4), "bmr": round(bmr)}


def recommend_meal(meal: str, target: dict, goal: str) -> List[dict]:
    """从食堂菜品库按「接近本餐热量目标 + 高蛋白」打分推荐。"""
    candidates = [d for d in CANTEEN_MENU if d["meal"] == meal]
    if goal != "增肌":
        candidates = [d for d in candidates if "高油" not in d["tags"]]
    meal_target = target["kcal"] * MEAL_SPLIT[meal]
    candidates.sort(key=lambda d: -abs(d["kcal"] - meal_target) + d["protein"] * 2, reverse=True)
    return candidates[:3]


# ══════════════════════════════════════════════════════════════════
# 六、记录持久化（JSON 文件，供营养统计与周报告使用）
# ══════════════════════════════════════════════════════════════════
def load_records() -> dict:
    if os.path.exists(RECORD_FILE):
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_records(records: dict):
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════
# 七、Streamlit 界面
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="AI饮食健康智能推荐系统", page_icon="🍱", layout="wide")

# 初始化状态
if "all_records" not in st.session_state:
    st.session_state.all_records = load_records()
if "api_key" not in st.session_state:
    st.session_state.api_key = VISION_API_KEY
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── 侧边栏：个人档案 + API 配置 ────────────────────────────────
with st.sidebar:
    st.header("👤 个人档案")
    username = st.text_input("你的名字（用于区分数据）", value="", placeholder="输入名字后回车")
    if not username:
        st.warning("请先输入你的名字，记录才会保存到你的账号下")
    st.divider()
    gender = st.radio("性别", ["男", "女"], horizontal=True)
    age = st.slider("年龄", 10, 60, 20)
    col1, col2 = st.columns(2)
    height_cm = col1.number_input("身高(cm)", 100.0, 250.0, 170.0, step=0.5)
    weight_kg = col2.number_input("体重(kg)", 30.0, 300.0, 60.0, step=0.5)
    activity = st.selectbox("活动量", list(ACTIVITY_FACTOR.keys()))
    goal = st.selectbox("饮食目标", list(GOAL_ADJUST.keys()))

    st.divider()
    st.header("🔑 识别 API")
    api_key = st.text_input("视觉API Key（豆包/火山方舟）", value=st.session_state.api_key,
                            type="password", help="也可用环境变量 VISION_API_KEY 配置")
    st.session_state.api_key = api_key
    st.caption("未配置时无法拍照识别，可改用手动录入。")
    if st.button("🗑 清除我的饮食记录"):
        if username in st.session_state.all_records:
            del st.session_state.all_records[username]
            save_records(st.session_state.all_records)
        st.success("已清除你的记录")

# 当前用户的记录视图
user_records = st.session_state.all_records.get(username, {}) if username else {}

# 个人目标
target = calc_target(gender, age, height_cm, weight_kg, activity, goal)
bmi = weight_kg / (height_cm / 100) ** 2

st.title("🍱 AI饮食健康智能推荐系统")
st.caption(f"BMI {bmi:.1f} ｜ 基础代谢 {target['bmr']} kcal ｜ 每日目标 {target['kcal']} kcal"
           f"（蛋白 {target['protein_g']}g / 脂肪 {target['fat_g']}g / 碳水 {target['carb_g']}g）")

tab1, tab2, tab3, tab4 = st.tabs(["📸 饮食记录", "📊 营养统计", "🍱 食堂推荐", "📄 每周报告"])

# ── Tab1 饮食记录 ───────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        photo = st.camera_input("📷 拍照（对准餐盘，光线充足）")
    with col_b:
        upload = st.file_uploader("或上传食物图片", type=["jpg", "jpeg", "png"])

    image_bytes = None
    if photo is not None:
        image_bytes = photo.getvalue()
        st.image(photo, width=240)
    elif upload is not None:
        image_bytes = upload.getvalue()
        st.image(upload, width=240)

    meal_choice = st.selectbox("本餐属于", ["早餐", "午餐", "晚餐", "加餐"])

    if st.button("🔍 AI 识别并分析", type="primary", disabled=image_bytes is None):
        try:
            with st.spinner("AI 正在识别食物并估算份量…"):
                foods = recognize_food(image_bytes, st.session_state.api_key)
            if not foods:
                st.error("未从图片中识别出食物，请换一张食物居中、光线更好的图片")
            else:
                st.session_state.last_result = foods
                st.success(f"识别到 {len(foods)} 种食物，可在下方表格中修正后保存")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"识别失败：{e}")

    st.divider()
    st.subheader("本餐食物清单（可手动录入或修正 AI 结果）")
    if st.session_state.last_result:
        default_df = pd.DataFrame(st.session_state.last_result)
    else:
        default_df = pd.DataFrame([{"name": "米饭", "amount_grams": 200.0, "confidence": 1.0}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", width="stretch",
        column_config={
            "name": st.column_config.TextColumn("食物名称"),
            "amount_grams": st.column_config.NumberColumn("份量(克)", min_value=1.0, step=10.0),
            "confidence": st.column_config.NumberColumn("置信度", min_value=0.0, max_value=1.0, format="%.2f"),
        },
        key="food_editor")

    if st.button("✅ 计算营养并保存本餐"):
        items = [{"name": r["name"], "amount_grams": float(r["amount_grams"]),
                  "confidence": float(r.get("confidence", 1.0))}
                 for _, r in edited.iterrows() if str(r["name"]).strip()]
        if not items:
            st.warning("请至少填写一种食物")
        else:
            # 内置库未匹配的食物 → 调用 API 补全营养（尽可能全面）
            missing = sorted({it["name"] for it in items if get_nutrition(it["name"]) is None})
            extra_nutrition = {}
            if missing:
                try:
                    with st.spinner(f"正在调用 API 补全 {len(missing)} 种食物的营养成分…"):
                        extra_nutrition = complete_nutrition_via_api(missing, st.session_state.api_key)
                    if extra_nutrition:
                        st.info(f"已由 AI 补全营养：{'、'.join(extra_nutrition.keys())}（按每100g估算，仅供参考）")
                except Exception as e:
                    st.warning(f"AI 营养补全未生效（{e}），未匹配食物将提示补充")

            details, totals, warnings, tips = analyze_items(items, extra_nutrition)
            st.session_state.last_result = None

            today = dt.date.today().isoformat()
            record = {"meal": meal_choice, "time": dt.datetime.now().strftime("%H:%M"),
                      "items": details, "totals": totals}
            # 按用户名隔离保存
            st.session_state.all_records.setdefault(username, {})
            st.session_state.all_records[username].setdefault(today, []).append(record)
            save_records(st.session_state.all_records)

            st.success(f"已保存至 {today} 的 {meal_choice}")

            col1, col2, col3 = st.columns(3)
            col1.metric("热量", f"{totals['kcal']} kcal")
            col2.metric("蛋白质", f"{totals['protein']} g")
            col3.metric("碳水", f"{totals['carb']} g")
            st.markdown("**营养明细（每份）**")
            detail_df = pd.DataFrame(details)
            detail_df = detail_df.rename(columns={
                "name": "食物", "amount_grams": "份量(g)", "confidence": "置信度",
                **{k: v for k, v in FIELD_LABELS.items() if k in detail_df.columns}})
            st.dataframe(detail_df, width="stretch")
            for w in warnings:
                st.warning(w)
            for t in tips:
                st.info(t)

# ── Tab2 营养统计 ───────────────────────────────────────────────
with tab2:
    today = dt.date.today().isoformat()
    today_records = user_records.get(today, [])
    st.subheader(f"今日营养汇总（{today}，已记录 {len(today_records)} 餐）")
    if not today_records:
        st.info("今天还没有记录，去「饮食记录」拍照或手动录一餐吧")
    else:
        day = {}
        meal_rows = []
        for r in today_records:
            for k, v in r["totals"].items():
                day[k] = day.get(k, 0.0) + v
            meal_rows.append({"餐次": r["meal"], "时间": r["time"], **r["totals"]})
        day = {k: round(v, 1) for k, v in day.items()}

        c1, c2, c3 = st.columns(3)
        c1.metric("今日热量", f"{day['kcal']} kcal", delta=f"目标 {target['kcal']} kcal")
        c2.metric("蛋白质", f"{day['protein']} g", delta=f"目标 {target['protein_g']} g")
        c3.metric("脂肪", f"{day['fat']} g", delta=f"目标 {target['fat_g']} g")

        st.markdown("**三大营养素完成度**")
        for label, now, tgt in [("蛋白质", day["protein"], target["protein_g"]),
                                ("脂肪", day["fat"], target["fat_g"]),
                                ("碳水", day["carb"], target["carb_g"])]:
            ratio = min(now / tgt, 1.0) if tgt else 0
            st.progress(ratio, text=f"{label} {now}g / {tgt}g（{ratio*100:.0f}%）")

        st.markdown("**各餐热量分布**")
        st.bar_chart(pd.DataFrame(meal_rows).set_index("餐次")[["kcal"]])
        st.dataframe(pd.DataFrame(meal_rows), width="stretch")

# ── Tab3 食堂推荐 ───────────────────────────────────────────────
with tab3:
    st.subheader(f"基于你的目标（{goal}）推荐食堂搭配")
    for meal in ["早餐", "午餐", "晚餐"]:
        st.markdown(f"#### {meal}（本餐参考 {target['kcal'] * MEAL_SPLIT[meal]:.0f} kcal）")
        plan = recommend_meal(meal, target, goal)
        for d in plan:
            st.markdown(
                f"- **{d['name']}** ｜ {d['window']} ｜ {d['kcal']} kcal"
                f"（蛋白{d['protein']}g/脂肪{d['fat']}g）｜ {d['tags']} ｜ {d['price']}")
    if goal == "减脂":
        st.info("💡 减脂提示：麻辣烫/自选菜记得「多菜少酱、主食减半」；避开油炸窗口。")
    elif goal == "增肌":
        st.info("💡 增肌提示：训练后加一杯牛奶或两个鸡蛋补足蛋白质；每天多吃一份肉蛋类。")
    else:
        st.info("💡 均衡提示：保持「一主食 + 一荤 + 两素」结构，均衡不超量。")

# ── Tab4 每周报告 ───────────────────────────────────────────────
with tab4:
    st.subheader("近 7 天饮食报告")
    if not user_records:
        st.info("暂无记录，先坚持记录几天就能生成周报告了")
    else:
        dates = sorted(user_records.keys(), reverse=True)[:7]
        rows = []
        for d in dates:
            day_totals = {}
            n_meals = 0
            for r in user_records[d]:
                for k, v in r["totals"].items():
                    day_totals[k] = day_totals.get(k, 0.0) + v
                n_meals += 1
            rows.append({"日期": d, "餐数": n_meals, **{k: round(v, 1) for k, v in day_totals.items()}})
        df = pd.DataFrame(rows).sort_values("日期")
        st.line_chart(df.set_index("日期")[["kcal"]])
        st.dataframe(df, width="stretch")

        avg_kcal = df["kcal"].mean()
        avg_protein = df["protein"].mean()
        skip_days = int((df["餐数"] < 3).sum())
        high_meal = int((df["kcal"] > 2500).sum())
        st.markdown("**本周小结**")
        if skip_days:
            st.warning(f"有 {skip_days} 天没吃够三餐，尽量规律进餐，跳餐最容易导致下一顿暴食")
        if high_meal:
            st.warning(f"有 {high_meal} 天总热量偏高，警惕油炸窗口和含糖饮料")
        st.info(f"日均热量 {avg_kcal:.0f} kcal（目标 {target['kcal']}），日均蛋白质 {avg_protein:.0f} g"
                f"（目标 {target['protein_g']} g）。"
                + ("蛋白质基本达标，继续保持" if avg_protein >= target["protein_g"] * 0.8
                   else "蛋白质略不足，每餐加一份蛋/肉/豆制品"))
