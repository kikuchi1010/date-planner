# app.py
import json
import urllib.parse
from typing import List, Dict
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="デートプラン自動生成", page_icon="💑", layout="wide")

# ----------------- Helpers -----------------
def maps_search_url(name: str, area: str = "") -> str:
    q = urllib.parse.quote_plus(f"{name} {area}".strip())
    return f"https://www.google.com/maps/search/?api=1&query={q}"

def plans_to_dataframe(plans: List[Dict]) -> pd.DataFrame:
    rows = []
    for p in plans:
        rows.append({
            "category": p.get("category", ""),
            "theme": p.get("theme", ""),
            "detail": p.get("detail", ""),
            "itinerary": p.get("itinerary", ""),
            "duration_min": p.get("duration_min", ""),
            "move_overview": p.get("move_overview", ""),
            "cost_pair_yen": p.get("cost_pair_yen", ""),
            "url": p.get("url", "")
        })
    return pd.DataFrame(rows)

DEFAULT_SYSTEM = """あなたは日本のデートプラン専門プランナー。入力条件（予算、移動上限、日程、嗜好/NG、デート種別）を厳守し、カテゴリが重複し過ぎないように多様性ある20案を生成すること。各案は必ずURLを1つ以上含める。公式URLが不明なら Google マップ検索URL を作る。出力は次のJSON配列（20要素）に厳密準拠：
[
  {
    "category": "アート|自然|食|体験|季節|癒し|夜景|屋内|屋外|旅行",
    "theme": "短いタイトル",
    "detail": "どんな体験かの要約（200字以内）",
    "itinerary": "時刻付きの簡易タイムライン（例：17:30 美術館→19:30 夜カフェ）",
    "duration_min": 180,
    "move_overview": "移動手段と概算所要（例：電車計40分）",
    "cost_pair_yen": 12000,
    "url": "https://..."
  }
]
予算や移動制約に90%以上の案が収まるよう調整。重複施設名は避ける。"""

def build_user_prompt(values: dict) -> str:
    # 整形してUserメッセージを作成
    lines = ["条件:"]
    lines.append(f"- デート種別: {values['date_type']}")
    b = values["budget"]
    if b["unit"] == "pair":
        lines.append(f"- 予算: 2人で {b['min']}–{b['max']}円")
    else:
        lines.append(f"- 予算: 1人あたり {b['min']}–{b['max']}円")
    lines.append(f"- 出発地点: {values['origin']}")
    tl = values["travel_limit"]
    lines.append(f"- 移動上限: {('電車' if tl['type']=='time' else '距離')}で{tl['value']}{'分' if tl['type']=='time' else 'km'}以内")
    if values.get("ages"):
        lines.append(f"- 年齢: 自分{values['ages'].get('self','')} / 相手{values['ages'].get('partner','')}")
    if values.get("interests"):
        lines.append(f"- 興味: {values['interests']}")
    if values.get("dietary"):
        lines.append(f"- 嗜好/NG: {values['dietary']}")
    if values.get("date_pref"):
        dp = values["date_pref"]
        lines.append(f"- 日付: {dp.get('date','')}")
    lines.append(f"- 雨天代替: {'必要' if values.get('weather_alt') else '不要'}")
    lines.append("出力は20案のJSON配列のみ。解説文は不要。")
    return "\n".join(lines)

# ----------------- UI -----------------
st.title("💑 デートプラン自動生成（GPT-5）")

with st.form("inputs"):
    c1, c2, c3 = st.columns(3)
    date_type = c1.selectbox("デート種別*", ["昼だけ","夜だけ","半日","1日","旅行"], index=1)
    origin = c2.text_input("出発地点（駅/エリア）*", placeholder="例：錦糸町 / 渋谷 / 横浜")
    budget_unit = c3.selectbox("予算単位*", ["2人合計","1人あたり"])
    c4, c5 = st.columns(2)
    budget_min = c4.number_input("予算最小*", min_value=0, value=8000, step=500)
    budget_max = c5.number_input("予算最大*", min_value=0, value=15000, step=500)

    c6, c7 = st.columns(2)
    travel_type = c6.selectbox("移動制約の種類*", ["時間（分）","距離（km）"])
    travel_value = c7.number_input("移動上限値*", min_value=0, value=30, step=5)

    c8, c9 = st.columns(2)
    age_self = c8.text_input("あなたの年齢（任意）", placeholder="例：33")
    age_partner = c9.text_input("相手の年齢（任意）", placeholder="例：37")

    interests = st.text_input("趣味・興味（任意）", placeholder="例：美術館, カフェ, 静かな場所")
    dietary = st.text_input("嗜好/NG（任意）", placeholder="例：辛いNG, アルコールOK")
    date_pref = st.text_input("日付（任意）", placeholder="例：2025-10-05")
    weather_alt = st.checkbox("雨天代替も欲しい", value=True)

    with st.expander("上級者設定（モデル/プロンプト編集）", expanded=False):
        model = st.selectbox("モデル", ["gpt-5","gpt-5-mini","gpt-5-chat-latest"])
        verbosity = st.select_slider("verbosity", options=["low","medium","high"], value="medium")
        reasoning = st.select_slider("reasoning_effort", options=["minimal","medium","high"], value="minimal")
        system_prompt = st.text_area("System Prompt（編集可）", value=DEFAULT_SYSTEM, height=260)
        user_prefix = st.text_area("User Promptプレフィクス（任意）", value="", height=60)
        user_suffix = st.text_area("User Promptサフィクス（任意）", value="", height=60)
        if st.button("初期Systemに戻す"):
            st.session_state["system_prompt"] = DEFAULT_SYSTEM

    submitted = st.form_submit_button("🎯 生成開始")

# ----------------- Run -----------------
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

if submitted:
    # 入力組み立て
    values = {
        "date_type": {"昼だけ":"day","夜だけ":"night","半日":"half","1日":"full","旅行":"trip"}[date_type],
        "budget": {
            "unit": "pair" if budget_unit=="2人合計" else "per_person",
            "min": int(budget_min), "max": int(budget_max)
        },
        "origin": origin.strip(),
        "travel_limit": {
            "type": "time" if travel_type=="時間（分）" else "distance",
            "value": int(travel_value)
        },
        "ages": {"self": age_self, "partner": age_partner},
        "interests": interests.strip(),
        "dietary": dietary.strip(),
        "date_pref": {"date": date_pref.strip()},
        "weather_alt": weather_alt
    }
    user_msg = build_user_prompt(values)
    if user_prefix: user_msg = user_prefix.strip() + "\n\n" + user_msg
    if user_suffix: user_msg = user_msg + "\n\n" + user_suffix.strip()

    with st.spinner("GPT-5が20案を生成中…"):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content": st.session_state.get("system_prompt", system_prompt)},
                {"role":"user","content": user_msg}
            ],
            temperature=0.7,
            extra_body={
                "verbosity": verbosity,
                "reasoning_effort": reasoning
            }
        )
    text = resp.choices[0].message.content.strip()

    # JSONパース & URL埋め
    try:
        plans = json.loads(text)
        # URLがない場合は検索URLを補完
        for p in plans:
            if not p.get("url"):
                name = p.get("theme","デート")
                area = origin
                p["url"] = maps_search_url(name, area)
        df = plans_to_dataframe(plans)
        st.success(f"{len(df)}件のプランを生成しました。")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSVをダウンロード", data=csv, file_name="date_plans.csv", mime="text/csv")
    except Exception as e:
        st.error("出力の解析に失敗しました。プロンプトのJSONスキーマを見直すか、再実行してください。")
        st.code(text)
