# app.py — Date Planner (MVP) using Streamlit + OpenAI
# ----------------------------------------------------
# 概要:
#   ・2人の基本情報 + 条件を入力 → OpenAIでカテゴリ別20案のデートプランを生成
#   ・各案にURL(公式 or Googleマップ検索URL)を必ず付与
#   ・CSV/Markdownエクスポート、簡易フィルタ/並び替え
#
# 使い方:
#   1) Streamlit Community Cloudにデプロイ
#   2) [Secrets] に OPENAI_API_KEY を設定
#   3) 「生成開始」を押して結果を閲覧/エクスポート
#
# 注意:
#   ・MVPでは正式な営業時間/経路/料金の正確性は保証しません。必ずリンク先で最新情報をご確認ください。
#   ・Google Places/Directions API連携や厳密な移動時間は将来拡張で対応可能です。

import json
import math
import textwrap
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# --- OpenAI client ---
try:
    from openai import OpenAI
except Exception:
    # 古い openai パッケージ互換 (v0.x 系) を使っている場合のフォールバック
    OpenAI = None

# ---------------------- ユーティリティ ----------------------

def quote_plus(s: str) -> str:
    try:
        from urllib.parse import quote_plus as _qp
        return _qp(s)
    except Exception:
        return s.replace(" ", "+")


def google_maps_search_url(name: str, area: str = "") -> str:
    """施設名 + エリアから Google マップ検索URLを生成。必ずURLを返すフォールバック。
    """
    query = name.strip()
    if area.strip():
        query = f"{query} {area.strip()}"
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"


def to_markdown_table(df: pd.DataFrame) -> str:
    return "\n".join([df.to_markdown(index=False)])


def coalesce_url(item: Dict[str, Any], area_hint: str = "") -> str:
    # official_url があればそれを、無ければ Google Maps 検索URLを返す
    url = (item.get("url") or item.get("official_url") or "").strip()
    if url and url.startswith("http"):
        return url
    # item 内に place_name/spot_name があればそれを使って検索URLに
    name = (
        item.get("place_name")
        or item.get("spot_name")
        or item.get("theme")
        or item.get("category")
        or "デートスポット"
    )
    return google_maps_search_url(str(name), area_hint)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def build_prompt(user_input: Dict[str, Any]) -> str:
    """OpenAI へ渡すプロンプトを生成。JSON で 20 案返すよう強制。"""
    # 役割・制約と出力スキーマのガイド
    system_rules = textwrap.dedent(
        f"""
        あなたは日本在住カップル向けのデートプラン専門プランナーです。以下の厳密な要件に従って、
        入力条件(予算/移動制約/嗜好)を守りつつ、多様性のある**カテゴリ別 20 案**を JSON 配列で出力してください。

        ◆ 厳守事項
        - 20案 必須。カテゴリは最低5カテゴリ以上に分散。
        - 各案は重複を避け、ユニークなテーマ・体験にする。
        - 所要時間、移動の概算、費用は妥当な近似値で明記。
        - URL は必ず1つ以上含める。公式サイトが不明な場合は候補スポット名を返す(クライアント側で地図URL化)。
        - 日本の施設/イベント/飲食に限定し、常識的な営業時間帯を想定。

        ◆ 出力フォーマット(JSON 配列; 要素=各プラン)
        [
          {
            "category": "アート|アクティビティ|食|自然|癒し|季節|イベント|夜景|散歩|学び など",
            "theme": "秋の夜長×美術館→夜カフェ",
            "summary": "プランの要約(120文字以内)",
            "itinerary": [
              {"time": "17:30", "activity": "上野の森美術館"},
              {"time": "19:30", "activity": "不忍池散歩"},
              {"time": "20:15", "activity": "上野 夜カフェ"}
            ],
            "duration_min": 210,
            "move_overview": "電車合計40分程度",
            "cost_yen": {"pair": 12000, "breakdown": "入館料×2+カフェ×2"},
            "place_name": "上野の森美術館",  
            "url": "https://example.com" ,  
            "alt_weather": "屋内中心/雨天でも実施可",
            "why_match": "静かめ志向・アート好きに合致"
          }
        ]

        ◆ スタイル
        - 現実的・実用的。初対面〜安定カップルまで適用可能なトーン。
        - 年齢・嗜好・移動制約・予算を尊重した提案。
        - 旅行モードのときは1日以上の構成とする。
        """
    )

    # 入力情報を JSON で添付
    payload = json.dumps(user_input, ensure_ascii=False)

    user_directive = textwrap.dedent(
        f"""
        # 入力
        {payload}

        # 指示
        - 必ず JSON 配列のみを出力。説明文は不要。
        - 各要素は上記スキーマのキーをすべて含めること(unknownは空文字で可)。
        - 公式URLが断定できない場合は、候補施設名を `place_name` に必ず入れること。
        - 合計20案に満たない場合でも20案に到達するまで生成を続けること。
        """
    )

    prompt = system_rules + "\n\n" + user_directive
    return prompt


def call_openai(prompt: str, model_name: str = "gpt-4o") -> List[Dict[str, Any]]:
    """OpenAI API を呼び出し JSON配列を受け取る。"""
    api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY が設定されていません。[Settings] から入力してください。")
        return []

    # 新/旧クライアントに対応
    if OpenAI is not None:
        client = OpenAI(api_key=api_key)
        try:
            # Responses API 推奨 (利用環境に応じて変更可)
            resp = client.responses.create(
                model=model_name,
                input=prompt,
                temperature=0.6,
                max_output_tokens=6000,
            )
            text = resp.output_text
        except Exception as e:
            st.warning(f"Responses API 呼び出しに失敗しました: {e}\nChat Completions にフォールバックします。")
            try:
                # chat.completions フォールバック
                from openai import ChatCompletion
                # 古いSDKの場合の仮想呼び出し — 実環境に合わせて修正してください
                cc = ChatCompletion()
                r = cc.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=6000,
                )
                text = r["choices"][0]["message"]["content"]
            except Exception as e2:
                st.error(f"OpenAI 呼び出しに失敗しました: {e2}")
                return []
    else:
        # さらに古い openai==0.x を想定
        try:
            import openai
            openai.api_key = api_key
            r = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=6000,
            )
            text = r["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"OpenAI 呼び出しに失敗しました: {e}")
            return []

    # JSON をパース
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        # JSON以外の場合を救う: ```json ... ``` を除去
    except Exception:
        pass

    # コードブロック除去して再トライ
    cleaned = text.strip()
    for fence in ("```json", "```JSON", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        st.error("モデル出力のJSON解析に失敗しました。プロンプト/モデル設定を見直してください。")
        return []


def normalize_items(items: List[Dict[str, Any]], area_hint: str) -> List[Dict[str, Any]]:
    normed = []
    for it in items:
        item = dict(it)
        # URL フォールバック
        item["final_url"] = coalesce_url(item, area_hint)
        # 型の安全化
        item["duration_min"] = safe_int(item.get("duration_min"), 0)
        cost_pair = item.get("cost_yen", {}).get("pair", 0)
        item["cost_pair"] = safe_int(cost_pair, 0)
        item["category"] = (item.get("category") or "その他").strip()
        item["theme"] = (item.get("theme") or "プラン").strip()
        item["summary"] = (item.get("summary") or "").strip()
        item["move_overview"] = (item.get("move_overview") or "").strip()
        item["why_match"] = (item.get("why_match") or "").strip()
        normed.append(item)
    return normed


def group_by_category(items: List[Dict[str, Any]]):
    cats: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        cats.setdefault(it.get("category", "その他"), []).append(it)
    return cats


# --------------------------- Streamlit UI ---------------------------
st.set_page_config(page_title="Date Planner (MVP)", page_icon="💑", layout="wide")

st.title("💑 デートプラン自動生成 (MVP)")
st.caption("Python + Streamlit + OpenAI — 条件に合わせてカテゴリ別20案を提案します。
検索(公式URL候補の推定など)を伴う生成は gpt-5 を既定で使用します。")

with st.expander("Settings / API", expanded=False):
    default_model = st.session_state.get("OPENAI_MODEL", "gpt-5")
    model_name = st.text_input("OpenAI Model", value=default_model, help="例: gpt-5 / gpt-4.1 / gpt-4o (環境により選択可)")
    st.session_state["OPENAI_MODEL"] = model_name
    api_key_input = st.text_input("OPENAI_API_KEY (任意; Secrets未設定の場合のみ)", type="password")
    if api_key_input:
        st.session_state["OPENAI_API_KEY"] = api_key_input
    st.markdown(":information_source: **検索を用いる場合は gpt-5 を使用**します。モデル欄で別モデルを指定している場合でも、下のブースト設定が有効なら gpt-5 に切替わります。")

st.subheader("1) 基本情報")
col_a, col_b, col_c = st.columns([1.2, 1.2, 1])
with col_a:
    date_type = st.selectbox("デート種別", ["昼だけ", "夜だけ", "半日", "1日", "旅行"], index=1)
    origin = st.text_input("出発エリア/駅 (例: 錦糸町, 渋谷, 横浜)")
    travel_type = st.selectbox("移動制約の種類", ["時間(分)", "距離(km)"])
    travel_value = st.number_input("移動上限値", min_value=0, max_value=999, value=30, step=5)
with col_b:
    budget_unit = st.selectbox("予算の単位", ["2人合計", "1人あたり"], index=0)
    budget_min = st.number_input("予算(最小)", min_value=0, max_value=200000, value=8000, step=500)
    budget_max = st.number_input("予算(最大)", min_value=0, max_value=300000, value=15000, step=500)
    weather_alt = st.checkbox("雨天代替案も作る", value=True)
with col_c:
    age_self = st.text_input("自分の年齢 (例: 33)")
    age_partner = st.text_input("相手の年齢 (例: 37)")
    date_hint = st.text_input("日付/曜日/時間帯 (任意)")

st.subheader("2) 嗜好・NG・自由記述 (任意)")
col_d, col_e = st.columns(2)
with col_d:
    interests = st.text_area(
        "趣味・興味の例: 美術館 / カフェ / 散歩 / 温泉 / イルミネーション / ライブ",
        height=100,
    )
with col_e:
    dietary = st.text_area("食の嗜好/NG (例: 辛いNG, 甘党, アルコールOK, 静かめ希望)", height=100)

st.info(
    "入力のヒント：年齢や関係性(初回/記念日/リラックス重視など)も自由欄に書くと、提案の精度が上がります。\n"
    "URLは公式サイトが優先、無い場合はGoogleマップ検索URLが自動で追加されます。"
)

# 検索品質ブースト (gpt-5強制)
search_boost = st.checkbox("検索品質ブースト (公式URL優先・gpt-5使用)", value=True, help="オンの場合、モデル指定に関わらず gpt-5 を使用します。")

# 生成ボタン
btn = st.button("🚀 生成開始", use_container_width=True)

if btn:
    # 入力のバリデーション(必須)
    errors = []
    if not origin.strip():
        errors.append("出発エリア/駅 は必須です。")
    if budget_min > budget_max:
        errors.append("予算の最小値が最大値を超えています。")
    if errors:
        st.error("\n".join(errors))
        st.stop()

    user_input: Dict[str, Any] = {
        "date_type": {"昼だけ":"day","夜だけ":"night","半日":"half","1日":"full","旅行":"trip"}[date_type],
        "budget": {
            "unit": "pair" if budget_unit == "2人合計" else "per_person",
            "min": int(budget_min),
            "max": int(budget_max),
        },
        "origin": origin.strip(),
        "travel_limit": {
            "type": "time" if travel_type == "時間(分)" else "distance",
            "value": int(travel_value),
        },
        "ages": {"self": age_self.strip(), "partner": age_partner.strip()},
        "interests": interests.strip(),
        "dietary": dietary.strip(),
        "date_pref": date_hint.strip(),
        "weather_alt": bool(weather_alt),
    }

    with st.spinner("AIがプランを作成中…"):
        prompt = build_prompt(user_input)
        # モデル選択: 検索を伴う生成は gpt-5 を優先
_model = "gpt-5" if search_boost else (model_name or "gpt-5")
items_raw = call_openai(prompt, model_name=_model)

    if not items_raw:
        st.stop()

    items = normalize_items(items_raw, area_hint=origin.strip())

    # DataFrame 化
    rows = []
    for it in items:
        url_final = it.get("final_url")
        # タイムライン文字列
        timeline = ", ".join([f"{seg.get('time','')}: {seg.get('activity','')}" for seg in it.get("itinerary", [])])
        rows.append({
            "カテゴリ": it.get("category"),
            "テーマ": it.get("theme"),
            "要約": it.get("summary"),
            "所要(分)": it.get("duration_min"),
            "移動": it.get("move_overview"),
            "費用(2人)": it.get("cost_pair"),
            "行程": timeline,
            "URL": url_final,
            "代替(雨)": it.get("alt_weather", ""),
            "合う理由": it.get("why_match", ""),
        })
    df = pd.DataFrame(rows)

    # 並び替え/フィルタ UI
    st.subheader("3) 結果")
    sort_by = st.selectbox("並び替え", ["デフォルト", "費用(2人)", "所要(分)"])
    view_only_under_budget = st.checkbox("予算内のみ表示", value=False)

    df_view = df.copy()
    if view_only_under_budget:
        max_budget = int(budget_max) if user_input["budget"]["unit"] == "pair" else int(budget_max) * 2
        df_view = df_view[df_view["費用(2人)"].apply(lambda x: safe_int(x, 0) <= max_budget)]

    if sort_by == "費用(2人)":
        df_view = df_view.sort_values("費用(2人)", ascending=True, kind="mergesort")
    elif sort_by == "所要(分)":
        df_view = df_view.sort_values("所要(分)", ascending=True, kind="mergesort")

    # カテゴリ別タブ
    cats = ["すべて"] + sorted(df_view["カテゴリ"].dropna().unique().tolist())
    tabs = st.tabs(cats)

    with tabs[0]:
        st.dataframe(df_view, use_container_width=True, height=600)

    cat_to_tab = {c: t for c, t in zip(cats[1:], tabs[1:])}
    for cat, tab in cat_to_tab.items():
        with tab:
            sub = df_view[df_view["カテゴリ"] == cat]
            st.dataframe(sub, use_container_width=True, height=480)

    # エクスポート
    st.subheader("4) エクスポート")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSVをダウンロード", data=csv_bytes, file_name="date_plans.csv", mime="text/csv")

    md_text = to_markdown_table(df)
    st.download_button("Markdownをダウンロード", data=md_text, file_name="date_plans.md", mime="text/markdown")

    st.success(f"{len(df)} 件のプランを生成しました。必要に応じてカテゴリタブや並び替えで絞り込んでください。")

# -------------- フッター/ヘルプ --------------
with st.expander("ヘルプ / 既知の制約", expanded=False):
    st.markdown(
        """
        **既知の制約 (MVP)**
        - 公式URLの有無はモデル依存です。無い場合は施設名からGoogleマップ検索URLを自動生成します。
        - 移動時間は概算です。正確な経路/所要はリンク先でご確認ください。
        - Google Places/Directions API と連携すれば、営業時間・クチコミ・所要の精度を高められます。

        **ロードマップ**
        - 並び替えとフィルタの強化(屋内/屋外、雨OK、深夜対応)
        - ブックマーク/共有リンク、履歴保存(Supabase/SQLite)
        - 季節・開花・イルミ・花火などの時期特化テンプレート
        - レコメンド学習(ユーザーの選好反映)
        """
    )
