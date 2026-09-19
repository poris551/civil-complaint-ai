import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap

from nlp.preprocessing import clean_text
from nlp.keywords import extract_keywords
from embedding.embedder import ComplaintEmbedder
from clustering.hdbscan_cluster import cluster_embeddings
from similarity.duplicate import find_similar
from analytics.trends import detect_surge

st.set_page_config(
    page_title="민원 분석",
    layout="wide"
)

st.title("민원 분석")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["received_at"] = pd.to_datetime(df["received_at"])
    df["clean_text"] = df["complaint_text"].map(clean_text)
    return df

@st.cache_resource
def get_embedder():
    return ComplaintEmbedder()

@st.cache_data(show_spinner=False)
def make_embeddings(texts):
    model = get_embedder()
    return model.encode(texts)

df = load_data("data/test_complaints.csv")

st.sidebar.header("필터")
category = st.sidebar.multiselect(
    "분류",
    sorted(df["category"].unique()),
    default=sorted(df["category"].unique())
)
filtered = df[df["category"].isin(category)].copy()


c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "전체 민원",
    f"{len(filtered):,}건"
)
c2.metric(
    "대분류",
    f"{filtered['category'].nunique():,}개"
)
c3.metric(
    "세부분류",
    f"{filtered['subcategory'].nunique():,}개"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "현황",
    "군집",
    "유사 민원 검색",
    "급증 탐지"
])

with tab1:
    st.subheader("민원 추이")
    daily = (
        filtered.groupby("received_at")
        .size()
        .reset_index(name="count")
    )
    fig_daily = px.line(
        daily,
        x="received_at",
        y="count",
        markers=True,
        labels={
            "received_at": "접수일",
            "count": "발생건수"
        }
    )
    fig_daily.update_xaxes(
        tickformat="%Y년 %m월 %d일"
    )
    fig_daily.update_traces(
        hovertemplate=
        "접수일: %{x|%Y년 %m월 %d일}<br>"
        "발생건수: %{y}건"
        "<extra></extra>"
    )
    fig_daily.update_layout(
        xaxis_title="접수일",
        yaxis_title="발생건수",
        hovermode="x unified"
    )
    st.plotly_chart(
        fig_daily,
        use_container_width=True
    )
    left, right = st.columns(2)
    with left:
        st.subheader("세부분류")
        sub = (
            filtered["subcategory"]
            .value_counts()
            .reset_index()
        )
        sub.columns = [
            "subcategory",
            "count"
        ]
        fig_sub = px.bar(
            sub,
            x="subcategory",
            y="count",
            labels={
                "subcategory": "세부분류",
                "count": "발생건수"
            }
        )
        fig_sub.update_layout(
            xaxis_title="세부분류",
            yaxis_title="발생건수"
        )
        st.plotly_chart(
            fig_sub,
            use_container_width=True
        )
    with right:
        st.subheader("핵심 키워드")
        kws = extract_keywords(
            filtered["clean_text"].tolist(),
            top_n=15
        )
        kwdf = pd.DataFrame(
            kws,
            columns=[
                "keyword",
                "count"
            ]
        )
        fig_keyword = px.bar(
            kwdf,
            x="count",
            y="keyword",
            orientation="h",
            labels={
                "keyword": "키워드",
                "count": "발생건수"
            }
        )
        fig_keyword.update_layout(
            xaxis_title="발생건수",
            yaxis_title="키워드"
        )
        st.plotly_chart(
            fig_keyword,
            use_container_width=True
        )

with tab2:
    st.subheader("민원 군집 분석")
    st.info(
        "최초 실행 시 SentenceTransformer 모델 다운로드를 위해 "
        "인터넷 연결이 필요합니다."
    )

    if len(filtered) < 10:
        st.warning(
            "군집 분석을 진행하기 위한 민원 데이터가 부족합니다."
        )
    else:
        max_sample = min(
            len(filtered),
            1000
        )
        min_sample = min(
            100,
            max_sample
        )
        default_sample = min(
            len(filtered),
            500
        )
        if max_sample >= 100:

            sample_n = st.slider(
                "군집 분석 건수",
                min_value=100,
                max_value=max_sample,
                value=default_sample,
                step=50
            )
        else:
            sample_n = max_sample
        sample_df = (
            filtered
            .tail(sample_n)
            .reset_index(drop=True)
        )
        with st.spinner(
            "군집 분석 중..."
        ):
            emb = make_embeddings(
                sample_df["clean_text"].tolist()
            )
            labels, _ = cluster_embeddings(
                emb,
                min_cluster_size=6,
                min_samples=3
            )
            sample_df["cluster"] = (
                labels.astype(str)
            )
            reducer = umap.UMAP(
                n_neighbors=min(
                    15,
                    max(2, len(sample_df) - 1)
                ),
                min_dist=0.1,
                metric="cosine",
                random_state=42
            )
            coords = reducer.fit_transform(
                emb
            )
            sample_df["x"] = coords[:, 0]
            sample_df["y"] = coords[:, 1]
        fig_cluster = px.scatter(
            sample_df,
            x="x",
            y="y",
            color="cluster",
            labels={
                "cluster": "군집",
                "subcategory": "세부분류",
                "location": "지역",
                "complaint_text": "민원내용",
                "x": "",
                "y": ""
            },
            hover_data={
                "subcategory": True,
                "location": True,
                "complaint_text": True,
                "x": False,
                "y": False
            },
            title="민원 군집 시각화"
        )

        fig_cluster.update_layout(
            xaxis_title="",
            yaxis_title=""
        )

        st.plotly_chart(
            fig_cluster,
            use_container_width=True
        )

        st.subheader("군집별 요약")

        cluster_summary = (
            sample_df
            .groupby("cluster")
            .agg(
                count=(
                    "complaint_id",
                    "count"
                ),
                major_topic=(
                    "subcategory",
                    lambda x:
                    x.value_counts().index[0]
                ),
                example=(
                    "complaint_text",
                    "first"
                )
            )
            .sort_values(
                "count",
                ascending=False
            )
            .reset_index()
        )
        cluster_summary = (
            cluster_summary.rename(
                columns={
                    "cluster": "군집",
                    "count": "발생건수",
                    "major_topic": "주요 세부분류",
                    "example": "대표 민원"
                }
            )
        )
        st.dataframe(
            cluster_summary,
            use_container_width=True,
            hide_index=True
        )

with tab3:
    st.subheader(
        "유사 민원 검색"
    )

    query = st.text_area(
        "민원 내용 입력",
        "밤마다 배달 오토바이 소음 때문에 잠을 잘 수 없습니다."
    )

    top_k = st.slider(
        "검색 건수",
        min_value=3,
        max_value=20,
        value=8
    )

    if st.button(
        "유사 민원 찾기"
    ):
        if not query.strip():
            st.warning(
                "민원 내용을 입력해주세요."
            )
        else:
            with st.spinner(
                "유사 민원을 검색하고 있습니다..."
            ):
                all_emb = make_embeddings(
                    df["clean_text"].tolist()
                )
                q_emb = (
                    get_embedder()
                    .encode([
                        clean_text(query)
                    ])[0]
                )
                result = find_similar(
                    q_emb,
                    all_emb,
                    top_k=top_k,
                    threshold=0.45
                )
            rows = []
            for idx, score in result:
                row = df.iloc[idx]
                rows.append({
                    "유사도": round(
                        score * 100,
                        1
                    ),
                    "접수일":
                        row["received_at"].date(),

                    "지역":
                        row["location"],

                    "세부분류":
                        row["subcategory"],

                    "담당부서":
                        row["department"],

                    "민원내용":
                        row["complaint_text"]
                })
            if rows:
                result_df = pd.DataFrame(
                    rows
                )
                result_df["유사도"] = (
                    result_df["유사도"]
                    .astype(str)
                    + "%"
                )
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    "유사한 민원을 찾지 못했습니다."
                )

with tab4:
    st.subheader(
        "최근 민원 급증 탐지"
    )
    surge = detect_surge(
        df
    )
    surge_display = (
        surge.rename(
            columns={
                "subcategory": "세부분류",
                "recent_daily_avg": "최근 일평균",
                "baseline_daily_avg": "평시 일평균",
                "increase_pct": "증가율(%)"
            }
        )
    )
    if "최근 일평균" in surge_display.columns:
        surge_display["최근 일평균"] = (
            surge_display["최근 일평균"]
            .round(2)
        )

    if "평시 일평균" in surge_display.columns:
        surge_display["평시 일평균"] = (
            surge_display["평시 일평균"]
            .round(2)
        )
    if "증가율(%)" in surge_display.columns:
        surge_display["증가율(%)"] = (
            surge_display["증가율(%)"]
            .round(1)
        )
    st.dataframe(
        surge_display,
        use_container_width=True,
        hide_index=True
    )
    alerts = surge[
        (surge["increase_pct"] >= 100)
        &
        (surge["recent_daily_avg"] >= 2)
    ]
    if len(alerts) == 0:
        st.success(
            "현재 뚜렷한 민원 급증 항목이 없습니다."
        )
    else:
        st.subheader(
            "급증 감지"
        )

        for _, row in alerts.iterrows():
            st.warning(
                f"⚠ {row['subcategory']} 민원이 "
                f"평시 대비 "
                f"{row['increase_pct']:.1f}% 증가했습니다. "
                f"(최근 일평균 "
                f"{row['recent_daily_avg']:.1f}건)"
            )
st.divider()
