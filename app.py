import json
import pandas as pd
import streamlit as st
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

# OpenAI SDK
from openai import OpenAI

# =============================
# App Meta
# =============================
st.set_page_config(
    page_title="デートプラン自動生成 (ChatGPT5対応)",
    page_icon="💑",
    layout="wide",
)

st.title("💑 デートプランを一瞬で生成するアプリ（ChatGPT 5対応）")
st.caption("Python + Streamlit + OpenAI API で動作。個人情報は保存しません。")

# =============================
# Data Models
# =============================
@dataclass
class Budget:
    unit: str
    min: int
    max: int

@dataclass
class TravelLimit:
    type: str
    value: int

@dataclass
class UserInput:
    date_type: str
    budget: Budget
    origin: str
    travel_limit: TravelLimit
    ages_self: int | None = None
    ages_partner: int | None = None
    interests: str | None = None
    dietary: str | None = None
    date_pref: str | None = None
    weather_alt: bool = False
    notes: str | None = None

# =============================
# Helpers
# =============================
def generate_google_maps_search_url(name: str, area: str) -> str:
    from urllib.parse import quote_plus
    q = quote_plus(f"{name} {area}") if area else quote_plus(name)
    return f"https://www.google.com/maps/search/?api=1&query={q}"

def build_system_prompt(weather_alt: bool) -> str:
    extra = "全ての提案に雨天代替案を必ず含めてください。" if weather_alt else "必要に応じて雨天代替案を書いてください。"
    return (
        "あなたは日本のデートプラン専門プランナーです。"
        "カテゴリが偏らないよう多様性のある20案を生成してください。\n"
        "- 各案に category, theme, itinerary, duration_min, move_overview, cost_yen_pair, place_name, place_area, official_url, alt_weather, why_match を含める\n"
        f"- {extra}\n"
        "- 日本語で簡潔に書いてください\n"
    )

def build_user_prompt(ui: UserInput) -> str:
    return json.dumps(asdict(ui), ensure_ascii=False)

def call_openai_to_generate(ui: UserInput) -> Dict[str, Any]:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Secrets に OPENAI_API_KEY を設定してください。")
    model_name = st.secrets.get("OPENAI_MODEL", "gpt-5")

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model_name,
        temperature=0.6,
        max_output_tokens=2500,
        input=[
            {"role": "system", "content": build_system_prompt(ui.weather_alt)},
            {"role": "user", "content": build_user_prompt(ui)},
        ],
    )
    content = resp.output[0].content[0].text
    return json.loads(content)

def normalize_plans(raw: Dict[str, Any], fallback_area: str) -> List[Dict[str, Any]]:
    plans = raw.get("plans", [])
    rows = []
    for p in plans:
        name = p.get("place_name", "")
        area = p.get("place_area") or fallback_area or ""
        url = p.get("official_url") or generate_google_maps_search_url(name, area)
        rows.append({
            "カテゴリ": p.get("category", "その他"),
            "テーマ": p.get("theme", ""),
            "行程": " / ".join([f"{it.get('time','')}: {it.get('activity','')}" for it in p.get("itinerary", [])]),
            "所要(分)": p.get("duration_min", None),
            "移動の目安": p.get("move_overview", ""),
            "概算費用(2人)": p.get("cost_yen_pair", None),
            "スポット名": name,
            "エリア": area,
            "URL": url,
            "雨天代替": p.get("alt_weather", ""),
            "おすすめ理由": p.get("why_match", ""),
        })
    return rows[:20]

# =============================
# UI Inputs
# =============================
col1, col2, col3 = st.columns([1.2, 1, 1])
with col1:
    date_type = st.selectbox("デート種別", ["昼だけ","夜だけ","半日","1日","旅行"])
with col2:
    budget_unit = st.selectbox("予算の単位", ["2人合計", "1人あたり"])
    unit_key = "pair" if budget_unit == "2人合計" else "per_person"
with col3:
    origin = st.text_input("出発地点（エリア/駅）", "錦糸町")

bmin = st.number_input("予算下限", 0, 100000, 3000, 500)
bmax = st.number_input("予算上限", bmin, 200000, 15000, 500)
travel_value = st.number_input("移動上限（分またはkm）", 0, 500, 30, 5)
age_self = st.number_input("自分の年齢（任意）", 0, 120, 33, 1)
age_partner = st.number_input("相手の年齢（任意）", 0, 120, 37, 1)
interests = st.text_area("趣味・興味", "美術館, カフェ, 散歩")
dietary = st.text_input("食の嗜好/NG", "辛いNG, アルコールOK")
date_pref = st.text_input("日付/時間帯", "2025-10-05 17:00-22:00")
notes = st.text_area("その他メモ", "落ち着いた雰囲気が好き")
weather_alt = st.checkbox("雨天代替案を含める", True)

# =============================
# Run
# =============================
if st.button("✨ 生成開始"):
    with st.spinner("ChatGPT 5 がデートプランを生成中..."):
        ui = UserInput(
            date_type=date_type,
            budget=Budget(unit=unit_key, min=int(bmin), max=int(bmax)),
            origin=origin,
            travel_limit=TravelLimit(type="time", value=int(travel_value)),
            ages_self=age_self,
            ages_partner=age_partner,
            interests=interests,
            dietary=dietary,
            date_pref=date_pref,
            weather_alt=weather_alt,
            notes=notes,
        )
        raw = call_openai_to_generate(ui)
        rows = normalize_plans(raw, fallback_area=origin)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ CSV保存", df.to_csv(index=False).encode("utf-8-sig"), "plans.csv")
