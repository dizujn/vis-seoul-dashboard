"""
서울시 자치구별 상권 분석 대시보드
- 컨셉 1. 자치구별 상권 변화 비교 (지도 클릭 선택 + 시계열)
- 컨셉 2. 유동인구와 추정매출의 관계 (횡단면 회귀 + 잔차 + 개별 구 지수 추이)
- 컨셉 3. 시간대·요일별 상권 활동 패턴 비교

실행:  streamlit run app.py
데이터: data.csv (process_csv.py 산출) / 경계: seoul_gu.geojson (process_spatial.py 산출, WGS84)
      → 없으면 shp/seoul_gu.shp 를 GeoPandas로 읽음
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import t as t_dist
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 경로 · 상수
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data.csv"
GEOJSON_PATH = BASE_DIR / "seoul_gu.geojson"          # 처리된 서울 경계 (WGS84, 경량화) — 기본
SHP_PATH = BASE_DIR / "shp" / "seoul_gu.shp"          # 원본 좌표계(EPSG:5179) SHP — GeoJSON 없을 때

# 색상 (밝은 미색 배경 + 따뜻한 톤)
BG = "#FBF6EC"
PANEL = "#FFFDF8"
GRID = "#EBE0CC"
TEXT = "#3B2F2A"
MUTED = "#8C7B6B"
BASE_FILL = "#EFE4D0"       # 지도 미선택 자치구
POS_COLOR = "#D1603D"       # 회귀선 위 / 양의 잔차
NEG_COLOR = "#5B8E7D"       # 회귀선 아래 / 음의 잔차
AVG_COLOR = "#7A6A5C"       # 전체 자치구 평균 (항상 표시)
AVG_NAME = "전체 자치구 평균"
FOOT_COLOR = "#E3A33B"      # 지수 그래프: 유동인구
SALES_COLOR = "#B8432F"     # 지수 그래프: 추정매출액
FONT = "Pretendard, 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"

GU_ORDER = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구",
    "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구",
    "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
]
# 25개 자치구 고유색: 다채롭지만 따뜻한 쪽으로 기운 팔레트 (모든 차트에서 동일하게 사용)
WARM_PALETTE = [
    "#C8553D", "#E9A23B", "#6A994E", "#B56576", "#2F7F7A", "#E07A5F", "#8E4162", "#BC6C25",
    "#5B8E7D", "#D1495B", "#A68A2C", "#6D597A", "#F08A4B", "#7F5539", "#3E7CB1", "#C9A227",
    "#9C2F3E", "#81A35A", "#D98C8C", "#4F6D7A", "#DDA15E", "#606C38", "#E36588", "#9C6644",
    "#355070",
]
GU_COLOR = dict(zip(GU_ORDER, WARM_PALETTE))

TIME_SLOTS = [("00_06", "00–06시"), ("06_11", "06–11시"), ("11_14", "11–14시"),
              ("14_17", "14–17시"), ("17_21", "17–21시"), ("21_24", "21–24시")]
TIME_HOURS = np.array([6, 5, 3, 3, 4, 3])  # 시간대 구간 길이(시간) — 구간마다 달라 1시간당 환산에 사용
DAYS = ["월", "화", "수", "목", "금", "토", "일"]
TIME_UNIT_OPTS = ["1시간당", "구간 합계"]  # 첫 항목이 기본값

MAX_GU_C1 = 5  # 컨셉 1 최대 선택 자치구 수
VAR_C1 = ["점포 수 변화", "개·폐업률 변화", "추정매출액 변화"]

EOK = 1e8   # 억 원
MAN = 1e4   # 만 명


# ─────────────────────────────────────────────────────────────
# 페이지 설정 · 스타일
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="서울시 자치구별 상권 분석 대시보드", page_icon="🏙️", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    f"""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    html, body, [class*="css"], .stApp {{ font-family: {FONT}; }}
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    /* 사이드바 제거 */
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{ display: none; }}
    .stApp {{ overflow-x: hidden; }}
    .block-container, [data-testid="stMainBlockContainer"] {{
        padding-top: 0 !important; padding-left: 3rem !important; padding-right: 3rem !important;
        max-width: 1200px !important; margin: 0 auto;
    }}

    /* ── 히어로: 제목 + 반투명 둥근 호 + 개요 + 큰 질문 ── */
    .hero-wrap {{ min-height: 94vh; display: flex; flex-direction: column; }}
    .hero-band {{
        position: relative; overflow: hidden;
        width: 100vw; margin-left: calc(50% - 50vw);
        padding: 92px 24px 120px; text-align: center;
    }}
    .hero-arcs {{ position: absolute; inset: 0; width: 100%; height: 100%; z-index: 0; }}
    .hero-title {{ position: relative; z-index: 1; }}
    .hero-title {{
        margin: 0; font-size: clamp(2rem, 4.2vw, 3.1rem); font-weight: 800;
        letter-spacing: -0.035em; color: {TEXT}; line-height: 1.2;
    }}
    .hero-body {{
        flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-start;
        padding: 40px 16px 24px; gap: 64px;   /* 호의 가장 낮은 지점 기준 위 여백 ≈ 개요↔질문 간격 */
    }}
    .hero-info {{
        display: flex; flex-wrap: wrap; justify-content: center; gap: 22px 64px; text-align: center;
    }}
    .info-item {{ min-width: 150px; }}
    .info-label {{
        font-size: 0.82rem; font-weight: 700; color: {POS_COLOR}; letter-spacing: 0.08em; margin-bottom: 6px;
    }}
    .info-value {{ font-size: 1.05rem; font-weight: 500; color: {TEXT}; line-height: 1.55; }}
    .hero-question {{ max-width: 860px; text-align: center; }}
    .hero-q-mark {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 34px; height: 34px; border-radius: 50%; margin-bottom: 14px;
        border: 1.5px solid {POS_COLOR}; color: {POS_COLOR}; font-weight: 800; font-size: 0.95rem;
    }}
    .hero-q-text {{
        font-size: clamp(1.15rem, 1.8vw, 1.35rem); font-weight: 500; line-height: 1.8; color: {TEXT};
    }}
    .hero-q-text mark {{
        background: linear-gradient(transparent 58%, rgba(233, 162, 59, 0.35) 58%);
        color: inherit; font-weight: 700; padding: 0 2px;
    }}
    .scroll-cue {{
        text-align: center; color: {MUTED}; font-size: 0.82rem; padding-bottom: 8px;
        animation: cue 1.8s ease-in-out infinite;
    }}
    @keyframes cue {{ 0%, 100% {{ transform: translateY(0); opacity: .55; }}
                      50% {{ transform: translateY(6px); opacity: 1; }} }}

    /* ── 컨셉 선택 안내 ── */
    .nav-prompt {{ text-align: center; margin: 56px 0 16px; }}
    .nav-prompt-title {{ font-size: 1.25rem; font-weight: 750; color: {TEXT}; }}
    .nav-prompt-sub {{ font-size: 0.9rem; color: {MUTED}; }}

    /* ── 컨셉 소제목 ── */
    .section-title {{
        text-align: center; font-size: 1.75rem; font-weight: 750; color: {TEXT};
        letter-spacing: -0.02em; margin: 4.5rem 0 1.5rem;
    }}
    .section-title::after {{
        content: ""; display: block; width: 44px; height: 3px; border-radius: 3px;
        background: {POS_COLOR}; margin: 12px auto 0; opacity: .7;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.02em; }}
    .question {{
        background: {PANEL}; border-left: 5px solid {POS_COLOR}; border-radius: 8px;
        padding: 12px 22px; color: {TEXT}; font-size: 1.02rem;
        width: fit-content; max-width: 100%; margin: 4px auto 22px; text-align: center;
    }}
    .card-title {{ font-weight: 700; font-size: 1.05rem; margin: 6px 0 2px 0; color: {TEXT}; }}
    .card-sub {{ color: {MUTED}; font-size: 0.86rem; margin-bottom: 4px; }}
    .guide {{
        background: #F6EEDF; border-radius: 8px; padding: 10px 14px; font-size: 0.9rem; color: {TEXT};
    }}
    [data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid {GRID}; border-radius: 10px; padding: 10px 14px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────
def yq_label(yq: int) -> str:
    y, q = divmod(int(yq), 10)
    return f"{y} {q}Q"


@st.cache_data
def load_geo():
    """경계 로드: 처리된 GeoJSON 우선, 없으면 SHP(GeoPandas). (geojson, 라벨 위치, 출처) 반환"""
    if GEOJSON_PATH.exists():
        with open(GEOJSON_PATH, encoding="utf-8") as f:
            geojson = json.load(f)
        source = "seoul_gu.geojson (원본 SHP → WGS84)"
    else:
        import geopandas as gpd
        gdf = gpd.read_file(SHP_PATH)
        if "SIG_KOR_NM" in gdf and not set(gdf["SIG_KOR_NM"]) & set(GU_ORDER):
            gdf = gpd.read_file(SHP_PATH, encoding="cp949")
        if "SIG_CD" in gdf:
            gdf = gdf[gdf["SIG_CD"].astype(str).str.startswith("11")]
        if gdf.crs is None:
            gdf = gdf.set_crs(5179)
        gdf = gdf.to_crs(4326)
        gdf["geometry"] = gdf.geometry.simplify(0.00005)
        geojson = json.loads(gdf[["SIG_CD", "SIG_KOR_NM", "geometry"]].to_json())
        source = "shp/seoul_gu.shp"

    # 라벨 위치: 가장 큰 외곽 링의 면적 중심
    centroids = {}
    for feat in geojson["features"]:
        g = feat["geometry"]
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
        best = None
        for ring in rings:
            r = np.asarray(ring)
            x, y = r[:, 0], r[:, 1]
            cross = x[:-1] * y[1:] - x[1:] * y[:-1]
            a = cross.sum() / 2
            if best is None or abs(a) > abs(best[0]):
                cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * a)
                cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * a)
                best = (a, cx, cy)
        centroids[feat["properties"]["SIG_KOR_NM"]] = (best[1], best[2])
    return geojson, centroids, source


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.drop_duplicates(["년분기", "자치구명"])
    df["분기"] = df["년분기"].map(yq_label)
    return df.sort_values(["년분기", "자치구명"]).reset_index(drop=True)


GEOJSON, CENTROIDS, GEO_SOURCE = load_geo()
DF = load_data()
QUARTERS = [int(q) for q in sorted(DF["년분기"].unique())]
QUARTER_LABELS = [yq_label(q) for q in QUARTERS]


# ─────────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────
def style_fig(fig, title=None, height=360, xtitle=None, ytitle=None, legend=True):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=BG, plot_bgcolor=PANEL, height=height,
        font=dict(family=FONT, color=TEXT, size=13),
        title=dict(text=title, font=dict(size=16), x=0, xanchor="left") if title else None,
        margin=dict(l=10, r=10, t=70 if title else 30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12)),
        showlegend=legend,
        hoverlabel=dict(bgcolor=PANEL, bordercolor=GRID, font=dict(family=FONT, color=TEXT)),
    )
    fig.update_xaxes(title=xtitle, gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(title=ytitle, gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def clicked_gu(event):
    """st.plotly_chart 선택 이벤트에서 자치구명을 꺼냄 (customdata → location → text → index 순)"""
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return None
    for p in points:
        for k in ("customdata", "location", "hovertext", "text"):
            v = p.get(k)
            if isinstance(v, (list, tuple)) and v:
                v = v[0]
            if isinstance(v, str):
                v = v.replace("<b>", "").replace("</b>", "")
                if v in GU_ORDER:
                    return v
        idx = p.get("point_index", p.get("point_number"))
        if p.get("curve_number", 0) == 0 and isinstance(idx, int) and idx < len(GU_ORDER):
            return GU_ORDER[idx]
    return None


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def card(title, sub=None):
    st.markdown(f'<div class="card-title">{title}</div>' +
                (f'<div class="card-sub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 컨셉 1. 자치구별 상권 변화 비교
# ─────────────────────────────────────────────────────────────
def seoul_map(selected):
    n = len(GU_ORDER)
    scale = []
    for i, g in enumerate(GU_ORDER):  # 자치구마다 계단식 색 구간 → 선택 시 고유색, 미선택 시 미색
        c = GU_COLOR[g] if g in selected else BASE_FILL
        scale += [[i / n, c], [(i + 1) / n, c]]

    fig = go.Figure(go.Choropleth(
        geojson=GEOJSON, featureidkey="properties.SIG_KOR_NM",
        locations=GU_ORDER, z=[i + 0.5 for i in range(n)], zmin=0, zmax=n,
        colorscale=scale, showscale=False, customdata=GU_ORDER,
        marker_line_color=PANEL, marker_line_width=1.6,
        hovertemplate="<b>%{location}</b><br>클릭해서 선택 / 해제<extra></extra>",
    ))
    names = [g for g in GU_ORDER if g in CENTROIDS]
    fig.add_trace(go.Scattergeo(
        lon=[CENTROIDS[g][0] for g in names], lat=[CENTROIDS[g][1] for g in names],
        text=[f"<b>{g}</b>" if g in selected else g for g in names],
        mode="text", hoverinfo="skip", showlegend=False,
        textfont=dict(family=FONT, size=11, color=[("#FFFFFF" if g in selected else MUTED) for g in names]),
    ))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor=BG, projection_type="mercator")
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=BG,
                      dragmode=False, font=dict(family=FONT))
    return fig


def line_chart(df, gus, col, title, ytitle, unit, div=1.0, per_area=False, fmt=",.0f", height=360):
    fig = go.Figure()
    # 전체 자치구 평균: 분기마다 25개 자치구 값의 단순 평균 (선택과 무관하게 항상 표시)
    v = df[col] / div
    if per_area:
        v = v / df["면적_km2"]
    avg = df.assign(_v=v).groupby("년분기").agg(분기=("분기", "first"), v=("_v", "mean"))
    fig.add_trace(go.Scatter(
        x=avg["분기"], y=avg["v"], name=AVG_NAME, mode="lines",
        line=dict(color=AVG_COLOR, width=3, dash="dash"),
        hovertemplate=f"{AVG_NAME}  <b>%{{y:{fmt}}}</b> {unit}<extra></extra>",
    ))
    for g in gus:
        sub = df[df["자치구명"] == g]
        y = sub[col] / div
        if per_area:
            y = y / sub["면적_km2"]
        fig.add_trace(go.Scatter(
            x=sub["분기"], y=y, name=g, mode="lines+markers",
            line=dict(color=GU_COLOR[g], width=2.6), marker=dict(size=5),
            hovertemplate=f"{g}  <b>%{{y:{fmt}}}</b> {unit}<extra></extra>",
        ))
    style_fig(fig, title, height=height, xtitle="연도·분기", ytitle=f"{ytitle} ({unit})")
    # 마우스 위치의 시점에 세로선 + 그 시점의 선택 자치구 수치를 한 상자에 표시
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(rangemode="tozero")  # Y축은 0에서 시작 (컨셉 3 막대그래프와 동일)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikecolor=MUTED, spikethickness=1, spikedash="dot")
    return fig


def page_concept1():
    section_title("① 자치구별 상권의 변화 비교")
    st.markdown('<div class="question">서울시 자치구별 상권의 규모와 활력은 시간에 따라 어떻게 변화했는가?</div>',
                unsafe_allow_html=True)

    ss = st.session_state
    if "c1_toggle" in ss:  # 직전 지도 클릭 반영
        g = ss.pop("c1_toggle")
        sel = list(ss["c1_gus"])
        if g in sel:
            sel.remove(g)
        elif len(sel) >= MAX_GU_C1:
            ss["c1_msg"] = f"최대 {MAX_GU_C1}개까지 선택할 수 있어요. 다른 자치구를 먼저 해제한 뒤 {g}를 선택하세요."
        else:
            sel.append(g)
        ss["c1_gus"] = sel

    left, right = st.columns([1, 1.45], gap="large")

    # 자치구 선택: 지도 클릭으로만
    with left:
        card("자치구 선택", f"지도에서 자치구를 클릭해 추가·해제하세요 (최대 {MAX_GU_C1}개)")
        event = st.plotly_chart(seoul_map(ss["c1_gus"]), key=f"c1_map_{ss['c1_map_v']}",
                                on_select="rerun", selection_mode="points")
        g = clicked_gu(event)
        if g:
            ss["c1_toggle"] = g
            ss["c1_map_v"] += 1  # 차트 key를 바꿔 선택 상태를 초기화 → 같은 구 재클릭 시 해제 가능
            st.rerun()
        if msg := ss.pop("c1_msg", None):
            st.warning(msg)
        if ss["c1_gus"]:
            st.button("선택 초기화", key="c1_reset", on_click=lambda: ss.update(c1_gus=[]))

    # 변수 선택 + 그래프
    with right:
        card("변수 선택", f"점선은 {AVG_NAME}(분기마다 25개 자치구 값의 단순 평균)으로 항상 표시됩니다")
        var = st.radio("변수", VAR_C1, horizontal=True, key="c1_var", label_visibility="collapsed")
        per_area = False
        if var != "개·폐업률 변화":
            mode = st.radio("표시 기준", ["절대값", "면적 대비 (㎢당)"], horizontal=True, key="c1_mode")
            per_area = mode.startswith("면적")

        gus = ss["c1_gus"]
        if not gus:
            st.caption(f"지금은 {AVG_NAME}만 표시 중입니다. 왼쪽 지도에서 자치구를 클릭해 비교해 보세요.")

        if var == "점포 수 변화":
            if per_area:
                fig = line_chart(DF, gus, "점포수", "1-1. 면적당 점포 수 변화", "면적당 점포 수", "개/㎢",
                                 per_area=True, fmt=",.1f", height=430)
            else:
                fig = line_chart(DF, gus, "점포수", "1-1. 점포 수 변화", "점포 수", "개", height=430)
            st.plotly_chart(fig, key="c1_stores")

        elif var == "개·폐업률 변화":
            # 같은 Y축 범위로 맞춰 두 그래프를 바로 비교할 수 있게 함
            rates = DF[DF["자치구명"].isin(gus)][["개업률", "폐업률"]]
            avg_rates = DF.groupby("년분기")[["개업률", "폐업률"]].mean()
            lo = float(min(rates.min().min(), avg_rates.min().min())) if gus else float(avg_rates.min().min())
            hi = float(max(rates.max().max(), avg_rates.max().max())) if gus else float(avg_rates.max().max())
            pad = (hi - lo) * 0.08 or 0.2
            for col, title, key in [("개업률", "1-2. 개업률 변화", "c1_open"),
                                    ("폐업률", "1-3. 폐업률 변화", "c1_close")]:
                fig = line_chart(DF, gus, col, title, col, "%", fmt=".2f", height=300)
                fig.update_yaxes(range=[0, hi + pad])
                st.plotly_chart(fig, key=key)
            st.caption("두 그래프의 Y축 범위는 같습니다.")

        else:
            if per_area:
                fig = line_chart(DF, gus, "총추정매출액", "1-4. 면적당 추정매출액 변화", "면적당 추정매출액",
                                 "억 원/㎢", div=EOK, per_area=True, fmt=",.1f", height=430)
            else:
                fig = line_chart(DF, gus, "총추정매출액", "1-4. 추정매출액 변화", "추정매출액", "억 원",
                                 div=EOK, height=430)
            st.plotly_chart(fig, key="c1_sales")


# ─────────────────────────────────────────────────────────────
# 컨셉 2. 유동인구와 매출의 관계
# ─────────────────────────────────────────────────────────────
@st.cache_data
def pooled_regression():
    """전체 기간 관측치(자치구 × 분기)를 하나로 묶은 단순 선형회귀 + 자치구별 잔차 요약"""
    d = DF[["년분기", "분기", "자치구명", "총유동인구", "총추정매출액"]].copy()
    d["x"] = d["총유동인구"] / MAN
    d["y"] = d["총추정매출액"] / EOK
    slope, intercept = np.polyfit(d["x"], d["y"], 1)
    d["pred"] = intercept + slope * d["x"]
    d["resid"] = d["y"] - d["pred"]
    r = float(np.corrcoef(d["x"], d["y"])[0, 1])
    r2 = float(1 - (d["resid"] ** 2).sum() / ((d["y"] - d["y"].mean()) ** 2).sum())
    by = (d.groupby("자치구명")["resid"].agg(mean="mean", min="min", max="max")
          .reset_index().sort_values("mean", ascending=False))

    # 기울기 유의확률
    # 같은 자치구가 여러 분기 반복 관측되므로 자치구별 군집 표준오차(CR1, 자유도 = 자치구 수 − 1)를 기본으로 사용
    X = np.column_stack([np.ones(len(d)), d["x"].to_numpy()])
    e = d["resid"].to_numpy()
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    groups = d["자치구명"].to_numpy()
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        sg = X[groups == g].T @ e[groups == g]
        meat += np.outer(sg, sg)
    n_cl = len(np.unique(groups))
    v_cl = xtx_inv @ meat @ xtx_inv * (n_cl / (n_cl - 1)) * ((n - 1) / (n - k))
    se_cl = float(np.sqrt(v_cl[1, 1]))
    p_cl = float(2 * t_dist.sf(abs(slope / se_cl), n_cl - 1))
    # 참고: 관측치 독립을 가정한 일반 OLS
    se_ols = float(np.sqrt((e @ e) / (n - k) * xtx_inv[1, 1]))
    p_ols = float(2 * t_dist.sf(abs(slope / se_ols), n - k))
    test = dict(se_cl=se_cl, p_cl=p_cl, df_cl=n_cl - 1, se_ols=se_ols, p_ols=p_ols)
    return d, by, float(slope), float(intercept), r, r2, test


def page_concept2():
    section_title("② 상권 활동과 매출의 관계")
    st.markdown('<div class="question">유동인구가 많은 자치구일수록 상권의 매출도 높은가?</div>',
                unsafe_allow_html=True)

    ss = st.session_state
    if "c2_click" in ss:  # 산점도/잔차 그래프 클릭 → 강조 토글 + 2-2 자치구 동기화
        g = ss.pop("c2_click")
        ss["c2_focus"] = None if ss["c2_focus"] == g else g
        if ss["c2_focus"]:
            ss["c2_gu"] = g

    d, by, slope, intercept, r, r2, test = pooled_regression()
    focus = ss["c2_focus"]
    n_gu, n_q = d["자치구명"].nunique(), d["년분기"].nunique()

    # 2-1 회귀분석
    card("2-1. 회귀분석: 유동인구 → 추정매출액",
         f"전체 기간({period_text()})의 자치구·분기 관측치를 하나로 묶은 단순 선형회귀")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("상관계수 r", f"{r:.3f}")
    m2.metric("결정계수 R²", f"{r2:.3f}")
    m3.metric("회귀 기울기", f"{slope:,.2f}", help="유동인구 1만 명 증가 시 예측 추정매출 증가분(억 원)")
    p = test["p_cl"]
    m4.metric("기울기 유의확률 p", "< 0.001" if p < 0.001 else f"{p:.3f}",
              delta="5% 수준 유의" if p < 0.05 else "5% 수준 유의하지 않음",
              delta_color="normal" if p < 0.05 else "off",
              help=(f"자치구별 군집 표준오차 기준 (표준오차 {test['se_cl']:.3f}, t분포 자유도 {test['df_cl']}). "
                    f"같은 자치구가 {n_q}개 분기에 반복 관측되므로 관측치를 독립으로 보는 일반 OLS 검정"
                    f"(p = {test['p_ols']:.1e})은 유의확률을 과소 추정합니다."))
    m5.metric("관측치", f"{len(d):,}개", help=f"{n_gu}개 자치구 × {n_q}개 분기")

    left, right = st.columns([1.15, 1], gap="large")
    cd_cols = ["자치구명", "분기", "y", "pred", "resid"]
    hover = ("<b>%{customdata[0]}</b> · %{customdata[1]}<br>유동인구: %{x:,.0f}만 명"
             "<br>실제 매출: %{customdata[2]:,.0f}억 원<br>예측 매출: %{customdata[3]:,.0f}억 원"
             "<br>잔차: %{customdata[4]:+,.0f}억 원<extra></extra>")

    # 2-1-1 산점도 + 회귀선
    with left:
        fig = go.Figure()
        # 자치구별 고유색: 한 자치구의 22개 분기 점이 하나의 클러스터로 보이도록
        for g in by["자치구명"]:
            sub = d[d["자치구명"] == g]
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"], mode="markers", name=g, showlegend=False,
                marker=dict(size=7, color=GU_COLOR[g], opacity=0.75 if focus is None else 0.15,
                            line=dict(color=PANEL, width=0.6)),
                customdata=sub[cd_cols].to_numpy(), hovertemplate=hover,
            ))
        xs = np.linspace(d["x"].min() * 0.95, d["x"].max() * 1.03, 50)
        fig.add_trace(go.Scatter(x=xs, y=intercept + slope * xs, mode="lines", name="회귀선", showlegend=True,
                                 line=dict(color=TEXT, width=2, dash="dash"), hoverinfo="skip"))
        if focus is not None:
            f = d[d["자치구명"] == focus].sort_values("년분기")
            fig.add_trace(go.Scatter(
                x=f["x"], y=f["y"], mode="lines+markers", name=focus,
                line=dict(color=GU_COLOR[focus], width=1.2),
                marker=dict(size=10, color=GU_COLOR[focus], line=dict(color=TEXT, width=1.5)),
                customdata=f[cd_cols].to_numpy(), hovertemplate=hover,
            ))
        style_fig(fig, "2-1-1. 유동인구와 추정매출액", height=470,
                  xtitle="분기 유동인구 (만 명)", ytitle="분기 추정매출액 (억 원)")
        ev = st.plotly_chart(fig, key=f"c2_scatter_{ss['c2_v']}", on_select="rerun", selection_mode="points")
        if (g := clicked_gu(ev)) is not None:
            ss["c2_click"], ss["c2_v"] = g, ss["c2_v"] + 1
            st.rerun()

    # 2-1-2 자치구별 평균 잔차 (가는 선 = 분기별 최소~최대)
    with right:
        colors = [GU_COLOR[g] for g in by["자치구명"]]
        opac = [1.0 if (focus is None or g == focus) else 0.35 for g in by["자치구명"]]
        lines = [2.5 if g == focus else 0 for g in by["자치구명"]]
        fig = go.Figure(go.Bar(
            x=by["자치구명"], y=by["mean"],
            marker=dict(color=colors, opacity=opac, line=dict(color=TEXT, width=lines)),
            error_y=dict(type="data", symmetric=False, array=by["max"] - by["mean"],
                         arrayminus=by["mean"] - by["min"], color=MUTED, thickness=1.1, width=2.5),
            customdata=by[["자치구명", "mean", "min", "max"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>평균 잔차: %{customdata[1]:+,.0f}억 원"
                          "<br>분기별 범위: %{customdata[2]:+,.0f} ~ %{customdata[3]:+,.0f}억 원<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        style_fig(fig, "2-1-2. 자치구별 평균 회귀 잔차", height=470, legend=False,
                  ytitle="평균 잔차 = 실제 − 예측 (억 원)")
        fig.update_xaxes(tickangle=-50, tickfont=dict(size=11))
        ev = st.plotly_chart(fig, key=f"c2_resid_{ss['c2_v']}", on_select="rerun", selection_mode="points")
        if (g := clicked_gu(ev)) is not None:
            ss["c2_click"], ss["c2_v"] = g, ss["c2_v"] + 1
            st.rerun()

    st.markdown(
        f'<div class="guide"><b>해석 가이드</b> &nbsp; '
        f'<b>양의 잔차</b>(0선 위): 실제 매출액이 회귀모형의 예측값보다 높음 &nbsp;·&nbsp; '
        f'<b>음의 잔차</b>(0선 아래): 실제 매출액이 회귀모형의 예측값보다 낮음'
        f'<br><span style="color:{MUTED}">점과 막대의 색은 자치구별 고유색이며, 마우스를 올리면 자치구명이 표시됩니다. 막대는 자치구별 {n_q}개 분기 잔차의 평균, 가는 선은 분기별 최소~최대 범위입니다. '
        f'점이나 막대를 클릭하면 해당 자치구의 {n_q}개 분기가 강조되고 아래 2-2 그래프의 자치구로 설정됩니다.</span></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # 2-2 개별 자치구 지수 추이
    card("2-2. 개별 자치구의 유동인구·추정매출액 변화",
         "기준 분기 = 100으로 환산한 지수 · 두 지표가 기준 대비 몇 % 변했는지를 같은 눈금에서 비교")
    c1, c2, _ = st.columns([1, 1, 2])
    gu = c1.selectbox("자치구", GU_ORDER, key="c2_gu")
    base_q = c2.selectbox("기준 분기", QUARTERS, format_func=yq_label, key="c2_base")
    sub = DF[DF["자치구명"] == gu].set_index("년분기")
    fig = go.Figure()
    for col, name, color in [("총유동인구", "유동인구", FOOT_COLOR), ("총추정매출액", "추정매출액", SALES_COLOR)]:
        idx = sub[col] / sub.loc[base_q, col] * 100
        fig.add_trace(go.Scatter(
            x=sub["분기"], y=idx, name=name, mode="lines+markers",
            line=dict(color=color, width=3), marker=dict(size=6),
            hovertemplate=f"{name}  <b>%{{y:.1f}}</b><extra></extra>",
        ))
    fig.add_hline(y=100, line_color=MUTED, line_dash="dot", line_width=1,
                  annotation_text=f"기준 {yq_label(base_q)} = 100", annotation_font_color=MUTED)
    style_fig(fig, f"{gu} 지수 추이", height=380, xtitle="연도·분기", ytitle="지수 (기준 분기 = 100)")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, key="c2_index")

    with st.expander("분석 시 유의사항"):
        st.markdown(
            f"- 전체 기간의 {len(d):,}개 관측치({n_gu}개 자치구 × {n_q}개 분기)를 하나의 회귀모형으로 분석합니다.\n"
            "- 같은 자치구가 여러 분기에 반복 관측되어 관측치가 서로 독립이 아닙니다. 그래서 기울기 유의확률은 "
            f"자치구별 군집 표준오차(자유도 {test['df_cl']})로 계산합니다. 일반 OLS 검정은 p값을 지나치게 작게 만듭니다.\n"
            "- 회귀분석 결과는 변수 간 **연관성**을 나타내며 인과관계를 의미하지 않습니다.\n"
            "- 자치구별 총량(규모)의 차이가 상관관계에 영향을 줄 수 있습니다.\n"
            "- 회귀 잔차를 상권의 우수성이나 효율성으로 직접 해석하지 않습니다."
        )


# ─────────────────────────────────────────────────────────────
# 컨셉 3. 시간대·요일 패턴
# ─────────────────────────────────────────────────────────────
def pattern_bar(d, gus, cols, labels, title, xtitle, measure, mode, div, unit, hours=None):
    """hours가 주어지면 구간 길이(시간)로 나눠 1시간당 값으로 환산 (시간대 그래프 전용)"""
    fig = go.Figure()
    if hours is None:
        fmt, u = (".1f", "%") if mode == "구성비" else (",.0f", unit)
    else:
        fmt, u = (".2f", "%/시간") if mode == "구성비" else (",.0f", f"{unit}/시간")

    def values(vals):
        v = vals / vals.sum(axis=-1, keepdims=True) * 100 if mode == "구성비" else vals / div
        return v / hours if hours is not None else v

    # 전체 자치구 평균: 자치구별 값(구성비 또는 절대값)을 25개 자치구에 대해 단순 평균 (항상 표시)
    all_vals = d[cols].astype(float).to_numpy()
    fig.add_trace(go.Bar(
        x=labels, y=values(all_vals).mean(axis=0), name=AVG_NAME,
        marker=dict(color="#CDBFB1", pattern=dict(shape="/", fgcolor=AVG_COLOR, size=6, solidity=0.25),
                    line=dict(color=AVG_COLOR, width=1)),
        hovertemplate=f"<b>{AVG_NAME}</b> · %{{x}}<br>{measure}: %{{y:{fmt}}}{u}<extra></extra>",
    ))
    for g in gus:
        y = values(d.loc[d["자치구명"] == g, cols].iloc[0].astype(float).to_numpy())
        fig.add_trace(go.Bar(
            x=labels, y=y, name=g, marker_color=GU_COLOR[g],
            hovertemplate=f"<b>{g}</b> · %{{x}}<br>{measure}: %{{y:{fmt}}}{u}<extra></extra>",
        ))
    if hours is None:
        ytitle = f"{measure} 구성비 (%)" if mode == "구성비" else f"{measure} ({unit})"
    else:
        ytitle = f"1시간당 {measure} 구성비 (%/시간)" if mode == "구성비" else f"1시간당 {measure} ({unit}/시간)"
    style_fig(fig, title, height=360, xtitle=xtitle, ytitle=ytitle)
    if hours is not None and mode == "구성비":  # 하루 24시간에 고르게 분포할 때의 기준선
        fig.add_hline(y=100 / 24, line_color=MUTED, line_dash="dot", line_width=1,
                      annotation_text="균등 분포 4.17%/시간", annotation_font_color=MUTED,
                      annotation_position="top left")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    return fig


def page_concept3():
    section_title("③ 상권 활동의 시간적 패턴 비교")
    st.markdown('<div class="question">자치구마다 상권 활동이 집중되는 시간대와 요일은 어떻게 다른가?</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.3, 1.6, 1])
    q = c1.select_slider("연도·분기", options=QUARTERS, format_func=yq_label, key="c3_q")
    # 선택한 자치구 태그를 자치구 고유색으로 표시
    tag_css = "".join(
        f'.st-key-c3_gus [data-baseweb="tag"]:has([title="{g}"]) {{ background-color: {GU_COLOR[g]} !important; }}'
        for g in GU_ORDER)
    st.markdown(f'<style>{tag_css} .st-key-c3_gus [data-baseweb="tag"] * {{ color: #FFFFFF; }}</style>',
                unsafe_allow_html=True)
    gus = c2.multiselect("자치구 (최대 3개)", GU_ORDER, max_selections=3, key="c3_gus")
    mode = c3.radio("표시 기준", ["구성비", "절대값"], horizontal=True, key="c3_mode")
    if not gus:
        st.caption(f"지금은 {AVG_NAME}만 표시 중입니다. 자치구를 선택해 비교해 보세요.")
    d = DF[DF["년분기"] == q]
    t_labels = [f"{lab}<br>({h}시간)" for (_, lab), h in zip(TIME_SLOTS, TIME_HOURS)]
    d_labels = [f"{x}" for x in DAYS]

    st.subheader("유동인구")
    a, b = st.columns(2, gap="large")
    with a:
        t_unit = st.radio("시간대 기준", TIME_UNIT_OPTS, horizontal=True, key="c3_ft_unit",
                          help="시간대 구간 길이가 6·5·3·3·4·3시간으로 달라, 1시간당으로 환산하면 구간 길이 차이를 없앤 비교가 됩니다.")
        st.plotly_chart(pattern_bar(d, gus, [f"시간대_{s}_유동인구" for s, _ in TIME_SLOTS], t_labels,
                                    "3-1-1. 시간대별 유동인구" + (" (1시간당)" if t_unit == "1시간당" else ""),
                                    "시간대", "유동인구", mode, MAN, "만 명",
                                    hours=TIME_HOURS if t_unit == "1시간당" else None),
                        key="c3_ft")
    with b:
        st.caption("요일은 모두 하루 단위라 환산 없이 비교합니다.")
        st.plotly_chart(pattern_bar(d, gus, [f"{x}요일_유동인구" for x in DAYS], d_labels,
                                    "3-1-2. 요일별 유동인구", "요일", "유동인구", mode, MAN, "만 명"),
                        key="c3_fd")

    st.divider()
    st.subheader("추정매출액")
    a, b = st.columns(2, gap="large")
    with a:
        t_unit = st.radio("시간대 기준", TIME_UNIT_OPTS, horizontal=True, key="c3_st_unit",
                          help="시간대 구간 길이가 6·5·3·3·4·3시간으로 달라, 1시간당으로 환산하면 구간 길이 차이를 없앤 비교가 됩니다.")
        st.plotly_chart(pattern_bar(d, gus, [f"시간대_{s}_추정매출액" for s, _ in TIME_SLOTS], t_labels,
                                    "3-2-1. 시간대별 추정매출액" + (" (1시간당)" if t_unit == "1시간당" else ""),
                                    "시간대", "추정매출액", mode, EOK, "억 원",
                                    hours=TIME_HOURS if t_unit == "1시간당" else None),
                        key="c3_st")
    with b:
        st.caption("요일은 모두 하루 단위라 환산 없이 비교합니다.")
        st.plotly_chart(pattern_bar(d, gus, [f"{x}요일_추정매출액" for x in DAYS], d_labels,
                                    "3-2-2. 요일별 추정매출액", "요일", "추정매출액", mode, EOK, "억 원"),
                        key="c3_sd")

    st.markdown(
        '<div class="guide">구성비는 각 자치구의 해당 분기 총량을 기준으로 시간대별·요일별로 각각 계산합니다. '
        f'빗금 막대인 <b>{AVG_NAME}</b>은 25개 자치구의 구성비(또는 절대값)를 단순 평균한 값입니다. '
        '<b>구성비</b>는 활동이 언제 집중되는지(패턴)를, <b>절대값</b>은 활동 규모를 보여줍니다. '
        '시간대 그래프의 <b>1시간당</b>은 구간 값을 구간 길이(6·5·3·3·4·3시간)로 나눈 값으로, '
        '구간 길이가 다른 시간대끼리 활동 강도를 공정하게 비교할 때 씁니다.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# 사이드바 · 라우팅
# ─────────────────────────────────────────────────────────────
# 위젯 기본값 + 페이지 전환 시에도 선택값 유지 (Streamlit은 화면에 없는 위젯 상태를 지우므로 재할당)
DEFAULTS = {
    "c1_gus": [], "c1_var": VAR_C1[0], "c1_mode": "절대값", "c1_map_v": 0,
    "c2_gu": "강남구", "c2_base": QUARTERS[0], "c2_focus": None, "c2_v": 0,
    "c3_q": QUARTERS[-1], "c3_gus": [], "c3_mode": "구성비",
    "c3_ft_unit": TIME_UNIT_OPTS[0], "c3_st_unit": TIME_UNIT_OPTS[0],
}
for _k, _v in DEFAULTS.items():
    st.session_state[_k] = st.session_state.get(_k, _v)

# 컨셉 전환 버튼: (key, 제목, 부제, 아이콘 SVG, 페이지 함수)
ICON_TREND = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect x='2' y='2' width='60' height='60' rx='16' fill='#FBE1D3'/>"
    "<path d='M13 49H51' stroke='#C9A994' stroke-width='2.5' stroke-linecap='round'/>"
    "<polyline points='13,43 23,39 32,41 41,30 51,25' fill='none' stroke='#E9A23B' stroke-width='4'"
    " stroke-linecap='round' stroke-linejoin='round'/>"
    "<polyline points='13,37 23,30 32,33 41,20 51,14' fill='none' stroke='#C8553D' stroke-width='4'"
    " stroke-linecap='round' stroke-linejoin='round'/>"
    "<circle cx='51' cy='14' r='4.2' fill='#C8553D'/><circle cx='51' cy='25' r='4.2' fill='#E9A23B'/>"
    "</svg>"
)
ICON_RELATION = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect x='2' y='2' width='60' height='60' rx='16' fill='#FDEBCB'/>"
    "<line x1='12' y1='49' x2='52' y2='15' stroke='#8C7B6B' stroke-width='3' stroke-dasharray='5 4'"
    " stroke-linecap='round'/>"
    "<circle cx='17' cy='38' r='4.6' fill='#D1603D'/><circle cx='24' cy='46' r='4.6' fill='#5B8E7D'/>"
    "<circle cx='30' cy='28' r='4.6' fill='#D1603D'/><circle cx='37' cy='38' r='4.6' fill='#5B8E7D'/>"
    "<circle cx='44' cy='19' r='4.6' fill='#D1603D'/><circle cx='49' cy='30' r='4.6' fill='#5B8E7D'/>"
    "</svg>"
)
ICON_PATTERN = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect x='2' y='2' width='60' height='60' rx='16' fill='#E5EEDB'/>"
    "<rect x='12' y='40' width='7' height='10' rx='2' fill='#E9A23B'/>"
    "<rect x='21' y='35' width='7' height='15' rx='2' fill='#6A994E'/>"
    "<rect x='30' y='25' width='7' height='25' rx='2' fill='#C8553D'/>"
    "<rect x='39' y='31' width='7' height='19' rx='2' fill='#E9A23B'/>"
    "<rect x='48' y='21' width='7' height='29' rx='2' fill='#C8553D'/>"
    "<circle cx='20' cy='20' r='9' fill='#FFFDF8' stroke='#5B8E7D' stroke-width='3'/>"
    "<path d='M20 15V20L24 22' stroke='#5B8E7D' stroke-width='2.6' fill='none' stroke-linecap='round'"
    " stroke-linejoin='round'/>"
    "</svg>"
)
NAV = [
    ("c1", "① 자치구별 상권 변화", "점포 수 · 개업률 · 폐업률 · 매출 추이", ICON_TREND, page_concept1),
    ("c2", "② 유동인구 × 매출 관계", "산점도 · 회귀 잔차 · 지수 추이", ICON_RELATION, page_concept2),
    ("c3", "③ 시간대·요일 패턴", "시간대별 · 요일별 구성비 비교", ICON_PATTERN, page_concept3),
]


def nav_css():
    from urllib.parse import quote
    css = []
    for key, _, sub, icon, _ in NAV:
        sel = f".st-key-nav_{key}"
        css.append(f"""
        {sel}, {sel} .stButton {{ width: 100%; }}
        {sel} button {{
            width: 100%; min-height: 66px; border-radius: 12px; padding: 10px 14px 10px 62px;
            justify-content: flex-start; text-align: left;
            background-color: {PANEL}; color: {TEXT}; border: 1.5px solid {GRID};
            background-image: url("data:image/svg+xml,{quote(icon)}");
            background-repeat: no-repeat; background-position: 13px center; background-size: 38px 38px;
            box-shadow: 0 1px 3px rgba(80, 50, 30, 0.06); transition: all .15s ease;
        }}
        {sel} button:hover {{
            border-color: {POS_COLOR}; color: {TEXT}; transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(160, 80, 40, 0.12);
        }}
        {sel} button:focus:not(:active) {{ color: {TEXT}; border-color: {POS_COLOR}; }}
        {sel} button[kind="primary"], {sel} button[data-testid="stBaseButton-primary"] {{
            background-color: #FFF0E3; border: 2px solid {POS_COLOR}; color: {TEXT};
            box-shadow: 0 4px 14px rgba(200, 85, 61, 0.18);
        }}
        {sel} button p {{ font-size: 0.93rem; font-weight: 700; line-height: 1.3; margin: 0; }}
        {sel} button p::after {{
            content: "{sub}"; display: block; margin-top: 3px;
            font-size: 0.75rem; font-weight: 500; color: {MUTED};
        }}""")
    # 스크롤로 화면에 들어올 때 아래에서 떠오르듯 등장 (지원하지 않는 브라우저는 그냥 표시)
    keys = ", ".join(f".st-key-nav_{k}" for k, *_ in NAV)
    css.append(f"""
        @keyframes rise {{ from {{ opacity: 0; transform: translateY(32px); }}
                          to {{ opacity: 1; transform: none; }} }}
        @supports (animation-timeline: view()) {{
            {keys}, .nav-prompt {{
                animation: rise linear both; animation-timeline: view(); animation-range: entry 0% cover 28%;
            }}
        }}""")
    st.markdown("<style>" + "".join(css) + "</style>", unsafe_allow_html=True)


def go_page(key):
    st.session_state["page"] = key


def period_text():
    (y0, q0), (y1, q1) = divmod(QUARTERS[0], 10), divmod(QUARTERS[-1], 10)
    return f"{y0}년 {q0}분기 ~ {y1}년 {q1}분기"


# 반투명 호 여러 겹: 붉은 계열(POS_COLOR 주변)을 옅게 겹쳐 연한 코랄 톤을 만듦
HERO_ARCS = (
    '<svg class="hero-arcs" viewBox="0 0 1440 300" preserveAspectRatio="none" aria-hidden="true">'
    '<path d="M0 0H1440V205Q720 335 0 205Z" fill="#C8553D" fill-opacity="0.10"/>'
    '<path d="M0 0H1440V140Q1010 330 0 245Z" fill="#E07A5F" fill-opacity="0.12"/>'
    '<path d="M0 0H1440V250Q430 300 0 135Z" fill="#D1603D" fill-opacity="0.08"/>'
    '<path d="M0 0H1440V110Q720 230 0 110Z" fill="#F2A58E" fill-opacity="0.20"/>'
    '<path d="M0 232Q720 352 1440 232" fill="none" stroke="#C8553D" stroke-opacity="0.22"'
    ' stroke-width="1.2" vector-effect="non-scaling-stroke"/>'
    '<path d="M0 168Q760 318 1440 182" fill="none" stroke="#C8553D" stroke-opacity="0.14"'
    ' stroke-width="1" vector-effect="non-scaling-stroke"/>'
    '</svg>'
)


def hero():
    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero-band">
            {HERO_ARCS}
            <h1 class="hero-title">서울시 자치구별 상권 분석 대시보드</h1>
          </div>
          <div class="hero-body">
            <div class="hero-info">
              <div class="info-item">
                <div class="info-label">분석 대상</div>
                <div class="info-value">서울시 {DF['자치구명'].nunique()}개 자치구</div>
              </div>
              <div class="info-item">
                <div class="info-label">분석 기간</div>
                <div class="info-value">{period_text()}<br>({len(QUARTERS)}개 분기)</div>
              </div>
              <div class="info-item">
                <div class="info-label">기반 데이터</div>
                <div class="info-value">서울시 상권분석서비스<br>+ 자치구 경계 데이터</div>
              </div>
            </div>
            <div class="hero-question">
              <div class="hero-q-mark">Q</div>
              <div class="hero-q-text">
                서울시 자치구별 상권은 시간에 따라 <mark>어떻게 변해 왔고</mark>,<br>
                유동인구·매출·활동 시간대에서 자치구마다 <mark>어떤 차이</mark>를 보이는가?
              </div>
            </div>
          </div>
          <div class="scroll-cue">아래로 스크롤 ⌄</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.session_state["page"] = st.session_state.get("page", "c1")
nav_css()
hero()
st.markdown('<div class="nav-prompt"><div class="nav-prompt-sub">버튼을 누르면 아래에 해당 분석이 펼쳐집니다</div></div>',
            unsafe_allow_html=True)
_, *nav_cols, _ = st.columns([0.8, 3, 3, 3, 0.8], gap="small")
for col, (key, label, _, _, _) in zip(nav_cols, NAV):
    with col:
        st.button(label, key=f"nav_{key}", on_click=go_page, args=(key,),
                  type="primary" if st.session_state["page"] == key else "secondary")

{k: fn for k, _, _, _, fn in NAV}[st.session_state["page"]]()
