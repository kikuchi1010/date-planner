# app.py
import json
import urllib.parse
from typing import List, Dict, Any
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="デートプラン自動生成", page_icon="💑", layout="wide")

# ================= Helpers =================
API_SESSION_KEY = "api_key"
MODEL_DEFAULT = "gpt-5"  # 既定モデル

DEFAULT_SYSTEM = """あなたは日本のデートプラン専門プランナー。入力条件（予算、移動上限、日程、嗜好/NG、デート種別）を厳守し、カテゴリが重複し過ぎないように多様性ある20案を生成すること。各案は必ずURLを1つ以上含める。公式URLが不明なら Google マップ検索URL を作る。
出力は JSON オブジェクトで、必ず次の形式にすること（解説文は出力しない）:
{
  "items": [
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
    // ← 要素は必ず20件
  ]
}
予算や移動制約に90%以上の案が収まるよう調整。重複施設名は避ける。"""

def get_api_key_from_ui() -> str:
    with st.sidebar:
        st.subheader("🔐 API設定")
        st.caption("※キーはセッション内のみ保持。公開URLで第三者に入力させない運用を推奨。")
        key_input = st.text_input("OpenAI API Key", type="password",
                                  placeholder="sk-...", value=st.session_state.get(API_SESSION_KEY, ""))
        c1, c2 = st.columns(2)
        if c1.button("保存/更新", use_container_width=True):
            if key_input and key_input.strip().startswith("sk-"):
                st.session_state[API_SESSION_KEY] = key_input.strip()
                st.success("APIキーを保存しました（セッション内）。")
            else:
                st.warning("キー形式が不正です（例: sk- で開始）。")
        if c2.button("クリア", use_container_width=True):
            st.session_state.pop(API_SESSION_KEY, None)
            st.info("APIキーをクリアしました。")
    return st.session_state.get(API_SESSION_KEY, "")

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

def build_user_prompt(values: dict) -> str:
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
        if dp.get("date"):
            lines.append(f"- 日付: {dp.get('date')}")
    lines.append(f"- 雨天代替: {'必要' if values.get('weather_alt') else '不要'}")
    lines.append("必ず JSON オブジェクト {\"items\":[...20件...]} のみを返すこと。")
    return "\n".join(lines)

def parse_plans(text: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            return data["items"]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

# ================ UI ================
st.title("💑 デートプラン自動生成（GPT-5）")
api_key = get_api_key_from_ui()

# 入力UI（フォームなし）
c1, c2, c3 = st.columns(3)
date_type = c1.selectbox("デート種別*", ["昼だけ","夜だけ","半日","1日","旅行"], index=1)
origin = c2.text_input("出発地点（駅/エリア）*", placeholder="例：錦糸町 / 渋谷 / 横浜")
budget_unit = c3.selectbox("予算単位*", ["2人合計","1人あたり"], index=0)

c4, c5 = st.columns(2)
budget_min = c4.number_input("予算最小*", min_value=0, value=8000, step=500)
budget_max = c5.number_input("予算最大*", min_value=0, value=15000, step=500)

c6, c7 = st.columns(2)
travel_type = c6.selectbox("移動制約の種類*", ["時間（分）","距離（km）"], index=0)
travel_value = c7.number_input("移動上限値*", min_value=1, value=30, step=5)

c8, c9 = st.columns(2)
age_self = c8.text_input("あなたの年齢（任意）", placeholder="例：33")
age_partner = c9.text_input("相手の年齢（任意）", placeholder="例：37")

interests = st.text_input("趣味・興味（任意）", placeholder="例：美術館, カフェ, 静かな場所")
dietary = st.text_input("嗜好/NG（任意）", placeholder="例：辛いNG, アルコールOK")
date_pref = st.text_input("日付（任意）", placeholder="例：2025-10-05")
weather_alt = st.checkbox("雨天代替も欲しい", value=True)

st.divider()
with st.expander("上級者設定（モデル/プロンプト編集・JSONモード）", expanded=False):
    # セッションに保持
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = DEFAULT_SYSTEM
    if "use_json_mode" not in st.session_state:
        st.session_state["use_json_mode"] = True

    model = st.selectbox("モデル", [MODEL_DEFAULT, "gpt-5-mini", "gpt-5-chat-latest"], index=0)
    st.text_area("System Prompt（編集可）", key="system_prompt", height=240)
    colx, coly, _ = st.columns([1,1,1])
    st.session_state["use_json_mode"] = colx.checkbox("厳格JSONモードを使う（推奨）", value=st.session_state["use_json_mode"])
    if coly.button("初期Systemに戻す"):
        st.session_state["system_prompt"] = DEFAULT_SYSTEM
        st.session_state["use_json_mode"] = True
        st.success("初期化しました。")
        st.experimental_rerun()
    user_prefix = st.text_area("User Promptプレフィクス（任意）", value="", height=60)
    user_suffix = st.text_area("User Promptサフィクス（任意）", value="", height=60)

# ✅ 送信ボタン（常時表示・フォームなし）
go = st.button("🎯 生成開始", type="primary", use_container_width=True)

# ================ Run ================
if go:
    # 1) APIキー確認
    if not api_key:
        st.error("APIキーが未設定です。左の「🔐 API設定」から入力してください。")
        st.stop()
    client = OpenAI(api_key=api_key)

    # 2) 入力検証
    errors = []
    if not origin.strip():
        errors.append("出発地点（駅/エリア）は必須です。")
    if int(budget_min) > int(budget_max):
        errors.append("予算の最小値が最大値を上回っています。")
    if int(travel_value) <= 0:
        errors.append("移動上限値は1以上を指定してください。")
    if errors:
        for e in errors: st.error(e)
        st.stop()

    # 3) 値を組み立て
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
        "ages": {"self": age_self.strip(), "partner": age_partner.strip()},
        "interests": interests.strip(),
        "dietary": dietary.strip(),
        "date_pref": {"date": date_pref.strip()},
        "weather_alt": weather_alt
    }
    user_msg = build_user_prompt(values)
    if user_prefix: user_msg = user_prefix.strip() + "\n\n" + user_msg
    if user_suffix: user_msg = user_msg + "\n\n" + user_suffix.strip()

    # 4) 呼び出し（JSONモード優先→失敗時フォールバック）
    messages = [
        {"role": "system", "content": st.session_state["system_prompt"]},
        {"role": "user", "content": user_msg}
    ]

    text = ""
    with st.spinner("GPT
