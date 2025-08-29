#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import re
from datetime import date

#######################
# Page configuration
st.set_page_config(
    page_title="DAPA Contract Analysis",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")
alt.themes.enable("default")

#######################
# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem; padding-right: 2rem;
    padding-top: 1rem;  padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stVerticalBlock"] { padding-left: 0rem; padding-right: 0rem; }

/* KPI 카드 흰색 */
[data-testid="stMetric"]{
    background-color:#ffffff; text-align:center; padding:15px 0;
    border:1px solid #eaeaea; border-radius:10px; color:#111111;
}
[data-testid="stMetricLabel"]{ display:flex; justify-content:center; align-items:center; }
[data-testid="stMetricDeltaIcon-Up"],[data-testid="stMetricDeltaIcon-Down"]{
    position:relative; left:38%; transform:translateX(-50%);
}

/* 작은 정보 상자 & 기간 배지 */
.small-metric{
    background:#fff; border:1px solid #eaeaea; border-radius:10px;
    padding:10px 12px; line-height:1.25; font-size:0.9rem; overflow-wrap:anywhere;
}
.small-metric b{ font-size:0.95rem; }
.period-badge{
    display:inline-block; background:#f6f6f6; border:1px solid #eaeaea;
    border-radius:8px; padding:8px 10px; margin-bottom:10px; font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)

#######################
# Load data
df_reshaped = pd.read_csv('data.csv', encoding='cp949')  # 분석 데이터 넣기

# 공통 전처리: 숫자/연도/단위 변환
df_reshaped = df_reshaped.copy()
if "계약금액" in df_reshaped.columns:
    df_reshaped["계약금액"] = pd.to_numeric(df_reshaped["계약금액"], errors="coerce")
else:
    df_reshaped["계약금액"] = pd.NA

if "계약체결일자" in df_reshaped.columns:
    df_reshaped["_date"] = pd.to_datetime(df_reshaped["계약체결일자"], errors="coerce")
    df_reshaped["_year"] = df_reshaped["_date"].dt.year
else:
    df_reshaped["_date"] = pd.NaT
    df_reshaped["_year"] = pd.NA

# 단위 컬럼
df_reshaped["계약금액_억원"] = df_reshaped["계약금액"] / 1e8
df_reshaped["계약금액_조원"] = df_reshaped["계약금액"] / 1e12

#######################
# Sidebar
with st.sidebar:
    st.title("국방 계약 분석 대시보드")
    st.caption("필터와 키워드를 조합해 원하는 계약만 좁혀보세요.")

    df = df_reshaped.copy()

    # ------------ 필터 UI ------------
    years = sorted([int(y) for y in df["_year"].dropna().unique()]) if df["_year"].notna().any() else []
    year_sel = st.selectbox("연도 선택 (Δ는 연도 선택 시 표시)", options=["전체"] + years, index=0)

    type_options = sorted([x for x in df.get("업무구분명", pd.Series(dtype=str)).dropna().unique()])
    type_sel = st.multiselect("업무구분 선택", options=type_options, default=[])

    method_options = sorted([x for x in df.get("계약체결방법명", pd.Series(dtype=str)).dropna().unique()])
    method_sel = st.multiselect("계약방법 선택", options=method_options, default=[])

    # 💰 금액 슬라이더: 억원
    amt_min = float(df["계약금액_억원"].min()) if df["계약금액_억원"].notna().any() else 0.0
    amt_max = float(df["계약금액_억원"].max()) if df["계약금액_억원"].notna().any() else 0.0
    amt_range = st.slider(
        "계약금액 범위 (억원)",
        min_value=float(0 if amt_min < 0 else round(amt_min, 1)),
        max_value=float(round(amt_max, 1)),
        value=(float(0 if amt_min < 0 else round(amt_min, 1)), float(round(amt_max, 1))),
        step=0.1,
        disabled=not df["계약금액_억원"].notna().any(),
    )

    st.markdown("---")

    # ------------ 키워드 검색 ------------
    keyword = st.text_input("🔎 키워드 검색", placeholder="계약명, 수요기관명, 대표업체명, 주소 등")
    match_mode = st.radio("일치 방식", ["AND (모두 포함)", "OR (하나라도 포함)"], horizontal=True)

    search_cols = [c for c in ["계약명", "수요기관명", "대표업체명", "대표업체주소", "업무구분명", "계약체결방법명"] if c in df.columns]
    if search_cols:
        if "search_text" not in df.columns:
            df["search_text"] = (
                df[search_cols].astype(str).fillna("").agg(" ".join, axis=1).str.lower()
            )
    else:
        df["search_text"] = ""

    # ------------ 필터 적용 ------------
    filtered = df.copy()
    if year_sel != "전체":
        filtered = filtered[filtered["_year"] == int(year_sel)]
    if type_sel:
        filtered = filtered[filtered["업무구분명"].isin(type_sel)]
    if method_sel:
        filtered = filtered[filtered["계약체결방법명"].isin(method_sel)]
    if filtered["계약금액_억원"].notna().any():
        filtered = filtered[(filtered["계약금액_억원"] >= amt_range[0]) & (filtered["계약금액_억원"] <= amt_range[1])]

    if keyword.strip():
        tokens = [t for t in keyword.lower().split() if t]
        if tokens:
            if match_mode.startswith("AND"):
                for t in tokens:
                    filtered = filtered[filtered["search_text"].str.contains(re.escape(t), na=False)]
            else:
                pattern = "|".join(map(re.escape, tokens))
                filtered = filtered[filtered["search_text"].str.contains(pattern, na=False)]

    filtered = filtered.drop(columns=["search_text"], errors="ignore")
    st.session_state["filtered_df"] = filtered

    # Δ 계산(연도 선택 시)
    prev_year_sum_jo = None
    if year_sel != "전체":
        y = int(year_sel)
        prev_df = df.copy()
        prev_df = prev_df[prev_df["_year"] == (y - 1)]
        if type_sel:
            prev_df = prev_df[prev_df["업무구분명"].isin(type_sel)]
        if method_sel:
            prev_df = prev_df[prev_df["계약체결방법명"].isin(method_sel)]
        if prev_df["계약금액_억원"].notna().any():
            prev_df = prev_df[(prev_df["계약금액_억원"] >= amt_range[0]) & (prev_df["계약금액_억원"] <= amt_range[1])]
        if keyword.strip():
            if "search_text" not in prev_df.columns:
                prev_df["search_text"] = (
                    prev_df[search_cols].astype(str).fillna("").agg(" ".join, axis=1).str.lower()
                    if search_cols else ""
                )
            tokens = [t for t in keyword.lower().split() if t]
            if tokens:
                if match_mode.startswith("AND"):
                    for t in tokens:
                        prev_df = prev_df[prev_df["search_text"].str.contains(re.escape(t), na=False)]
                else:
                    pattern = "|".join(map(re.escape, tokens))
                    prev_df = prev_df[prev_df["search_text"].str.contains(pattern, na=False)]
            prev_df = prev_df.drop(columns=["search_text"], errors="ignore")
        prev_year_sum_jo = float(prev_df["계약금액_조원"].sum()) if "계약금액_조원" in prev_df.columns else 0.0

    # 사이드바 요약 배지
    if "_date" in filtered.columns and filtered["_date"].notna().any():
        start_dt = pd.to_datetime(filtered["_date"].min()).date()
        end_dt   = pd.to_datetime(filtered["_date"].max()).date()
        total_jo = float(filtered["계약금액_조원"].sum()) if filtered["계약금액_조원"].notna().any() else 0.0
        st.success(f"기간: {start_dt} ~ {end_dt} | 총액: {total_jo:,.3f} 조원 | 건수: {len(filtered):,}건")
    else:
        total_jo = float(filtered["계약금액_조원"].sum()) if "계약금액_조원" in filtered.columns else 0.0
        st.success(f"기간: 데이터 없음 | 총액: {total_jo:,.3f} 조원 | 건수: {len(filtered):,}건")

    with st.expander("🗺️ 지도(geocoding) 설정", expanded=False):
        st.write("lat/lon 칼럼이 있으면 즉시 지도 표시, 없으면 주소 테이블을 표시합니다.")
        geocode_mode = st.radio("모드 선택", ["자동(칼럼 감지)", "사전매핑 사용(csv 불러오기)", "실시간 지오코딩(권장X)"], horizontal=False)
        st.caption("실시간 지오코딩은 속도/쿼터 문제로 권장하지 않습니다. 사전매핑 CSV를 만들어 lat/lon을 병합하세요.")

#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

# =======================
# 컬럼 0: KPI + 기간
# =======================
with col[0]:
    st.markdown("### 📌 핵심 지표")

    data = st.session_state.get("filtered_df", df_reshaped).copy()

    # 기간 배지
    if "_date" in data.columns and data["_date"].notna().any():
        start_dt = pd.to_datetime(data["_date"].min()).date()
        end_dt   = pd.to_datetime(data["_date"].max()).date()
        st.markdown(f"<div class='period-badge'>기간: <b>{start_dt}</b> ~ <b>{end_dt}</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='period-badge'>기간: 데이터 없음</div>", unsafe_allow_html=True)

    # KPI
    total_jo = data["계약금액_조원"].sum(skipna=True)
    avg_eok  = data["계약금액_억원"].mean(skipna=True)
    cnt      = len(data)

    delta_str = None
    if year_sel != "전체" and prev_year_sum_jo is not None:
        prev = prev_year_sum_jo
        if prev and prev > 0:
            delta_pct = (total_jo - prev) / prev * 100.0
            delta_str = f"{delta_pct:+.1f}%"
        else:
            delta_str = "–"
    st.metric("총 계약 금액", f"{total_jo:,.3f} 조원", delta=delta_str if delta_str else None)
    st.metric("평균 계약 금액", f"{avg_eok:,.1f} 억원" if pd.notna(avg_eok) else "-")
    st.metric("계약 건수", f"{cnt:,} 건")

    if "대표업체명" in data.columns and not data.empty:
        top_vendor_name = (
            data.groupby("대표업체명")["계약금액_억원"].sum().sort_values(ascending=False).index[0]
        )
        st.markdown(f"<div class='small-metric'>최대 계약 업체<br><b>{top_vendor_name}</b></div>", unsafe_allow_html=True)

# =======================
# 컬럼 1: 검색/필터 결과 테이블 + 계약방법 Breakdown + 히트맵
# =======================
with col[1]:
    st.markdown("### 🔎 검색/필터 결과 (테이블)")
    base = st.session_state.get("filtered_df", df_reshaped).copy()

    # 미니필터
    with st.expander("테이블 추가 필터", expanded=False):
        q_name   = st.text_input("계약명 검색", placeholder="예: 디스플레이, 연구 용역 등")
        q_vendor = st.text_input("업체 검색", placeholder="예: ㈜○○, 유한회사 ○○ 등")
        q_org    = st.text_input("수요기관 검색", placeholder="예: 해군, 육군본부 등")
        eok_min  = float(base["계약금액_억원"].min()) if base["계약금액_억원"].notna().any() else 0.0
        eok_max  = float(base["계약금액_억원"].max()) if base["계약금액_억원"].notna().any() else 0.0
        eok_range = st.slider(
            "계약금액(억원) 범위(테이블)",
            min_value=float(0 if eok_min < 0 else round(eok_min, 1)),
            max_value=float(round(eok_max, 1)),
            value=(float(0 if eok_min < 0 else round(eok_min, 1)), float(round(eok_max, 1))),
            step=0.1,
            disabled=not base["계약금액_억원"].notna().any(),
        )

    table_df = base.copy()
    if q_name.strip() and "계약명" in table_df.columns:
        table_df = table_df[table_df["계약명"].astype(str).str.contains(q_name, case=False, na=False)]
    if q_vendor.strip() and "대표업체명" in table_df.columns:
        table_df = table_df[table_df["대표업체명"].astype(str).str.contains(q_vendor, case=False, na=False)]
    if q_org.strip() and "수요기관명" in table_df.columns:
        table_df = table_df[table_df["수요기관명"].astype(str).str.contains(q_org, case=False, na=False)]
    if table_df["계약금액_억원"].notna().any():
        table_df = table_df[(table_df["계약금액_억원"] >= eok_range[0]) & (table_df["계약금액_억원"] <= eok_range[1])]

    show_cols = [c for c in ["계약체결일자","계약명","계약금액_억원","대표업체명","수요기관명","업무구분명","계약체결방법명","계약기간"] if c in table_df.columns]
    if "계약금액_억원" in show_cols:
        table_df = table_df.rename(columns={"계약금액_억원":"계약금액(억원)"})
    if show_cols:
        show_cols = [c if c!="계약금액_억원" else "계약금액(억원)" for c in show_cols]
        table_df_display = table_df[show_cols].sort_values(by="계약체결일자", ascending=False)
        st.dataframe(
            table_df_display.head(200),
            use_container_width=True,
            hide_index=True,
            column_config={
                "계약체결일자": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
                "계약금액(억원)": st.column_config.NumberColumn(format="%.1f"),
            }
        )
        csv = table_df_display.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 현재 표 다운로드 (CSV)", data=csv, file_name="filtered_contracts.csv", mime="text/csv")
    else:
        st.info("표시할 컬럼이 없습니다.")

    st.markdown("---")

    # 📊 계약방법 Breakdown
    st.subheader("📊 계약방법 Breakdown (억원)")
    if "계약체결방법명" in base.columns and not base.empty:
        method_summary = base.groupby("계약체결방법명")["계약금액_억원"].sum().reset_index()
        if not method_summary.empty:
            fig_method = px.pie(
                method_summary, names="계약체결방법명", values="계약금액_억원",
                title="계약방법별 계약금액 비율(억원)"
            )
            st.plotly_chart(fig_method, use_container_width=True)
        else:
            st.info("표시할 계약방법 데이터가 없습니다.")
    else:
        st.info("계약방법명 데이터가 없습니다.")

    st.markdown("---")

    # 🎛️ 히트맵
    st.subheader("📊 연도 × 계약방법 히트맵 (억원)")
    if "계약체결방법명" in base.columns and "_year" in base.columns:
        pivot_df = base.groupby(["_year","계약체결방법명"])["계약금액_억원"].sum().reset_index()
        if not pivot_df.empty:
            heatmap = alt.Chart(pivot_df).mark_rect().encode(
                x=alt.X("계약체결방법명:N", title="계약 방법"),
                y=alt.Y("_year:O", title="연도"),
                color=alt.Color("계약금액_억원:Q", title="금액(억원)", scale=alt.Scale(scheme="blues")),
                tooltip=[alt.Tooltip("_year:O", title="연도"),
                         alt.Tooltip("계약체결방법명:N", title="방법"),
                         alt.Tooltip("계약금액_억원:Q", title="금액(억원)", format=".1f")]
            ).properties(width="container", height=360)
            st.altair_chart(heatmap, use_container_width=True)
        else:
            st.info("표시할 히트맵 데이터가 없습니다.")
    else:
        st.info("계약체결방법명 / 연도 데이터가 부족하여 히트맵 표시 불가")

# =======================
# 컬럼 2: 🏆 순위만 표시 (지도 제거)
# =======================
with col[2]:
    st.markdown("### 🏆 순위")
    data = st.session_state.get("filtered_df", df_reshaped).copy()

    # Top 수요기관
    st.subheader("🏢 Top 수요기관 (억원)")
    if "수요기관명" in data.columns and not data.empty:
        top_org = (data.groupby("수요기관명")["계약금액_억원"].sum().reset_index()
                   .sort_values("계약금액_억원", ascending=False).head(10))
        if not top_org.empty:
            fig_org = px.bar(top_org, x="계약금액_억원", y="수요기관명", orientation="h",
                             labels={"계약금액_억원":"계약 금액(억원)","수요기관명":"기관"},
                             title="상위 10개 수요기관")
            fig_org.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_org, use_container_width=True)
        else:
            st.info("표시할 수요기관 순위 데이터가 없습니다.")
    else:
        st.info("수요기관명 데이터가 없습니다.")

    # Top 업체
    st.subheader("🏭 Top 업체 (억원)")
    if "대표업체명" in data.columns and not data.empty:
        top_vendor = (data.groupby("대표업체명")["계약금액_억원"].sum().reset_index()
                      .sort_values("계약금액_억원", ascending=False).head(10))
        if not top_vendor.empty:
            fig_vendor = px.bar(top_vendor, x="계약금액_억원", y="대표업체명", orientation="h",
                                labels={"계약금액_억원":"계약 금액(억원)","대표업체명":"업체"},
                                title="상위 10개 업체")
            fig_vendor.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_vendor, use_container_width=True)
        else:
            st.info("표시할 업체 순위 데이터가 없습니다.")
    else:
        st.info("대표업체명 데이터가 없습니다.")
