from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Karat Violette Dashboard",
    page_icon="🧀",
    layout="wide",
)

# =========================
# Style / palette
# =========================
PALETTE = [
    "#EC4899",  # pink
    "#A78BFA",  # lavender
    "#60A5FA",  # light blue
    "#2563EB",  # blue
]

PLOT_BG = "white"
PAPER_BG = "white"
GRID_CLR = "rgba(0,0,0,0.10)"
FONT_CLR = "#1F2937"


def apply_theme(fig):
    fig.update_layout(
        template="streamlit",
        colorway=PALETTE,
        legend_title_text="",
        margin=dict(l=20, r=40, t=50, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.18)", zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=False, zeroline=False, automargin=True)
    return fig


# =========================
# Helpers
# =========================
def find_data_dir() -> Path:
    candidates = [
        Path("/Users/evaeva.monsher/Desktop/karat_violette/data"),
        Path("data"),
        Path("./data"),
        Path("../data"),
        Path.cwd() / "data",
        Path.cwd().parent / "data",
    ]
    for c in candidates:
        if (c / "task_outputs" / "csv" / "task1_switch_rank_overall.csv").exists():
            return c

    for p in Path(".").rglob("task1_switch_rank_overall.csv"):
        return p.parent.parent.parent

    raise FileNotFoundError("Не удалось найти папку data с task_outputs/csv")


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def fmt_int(x):
    if pd.isna(x):
        return "—"
    return f"{int(round(x)):,}".replace(",", " ")


def fmt_num(x, ndigits=1):
    if pd.isna(x):
        return "—"
    return f"{x:.{ndigits}f}"


def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x * 100:.1f}%"


def infer_channel_label(v):
    if pd.isna(v):
        return "Неизвестно"

    if isinstance(v, str):
        low = v.strip().lower()
        if low in {"true", "1", "marketplace", "mp", "yes"}:
            return "Маркетплейс"
        if low in {"false", "0", "non_marketplace", "offline", "no"}:
            return "Не маркетплейс"

    return "Маркетплейс" if bool(v) else "Не маркетплейс"


def add_brand_shares(df: pd.DataFrame, base_col: str, n_col: str = "n_buyers") -> pd.DataFrame:
    out = df.copy()
    if base_col in out.columns:
        out["share_of_base"] = out[n_col] / out[base_col]
    return out


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def rename_cols(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    use = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=use)


def freq_caption() -> str:
    return (
        "Сегменты частотности: "
        "high_freq — до 14 дней включительно, "
        "monthly_like — 15–30 дней, "
        "low_freq — 31–60 дней, "
        "rare — более 60 дней."
    )


def pretty_seg(s: str) -> str:
    mp = {
        "high_freq": "high_freq (≤14 дней)",
        "monthly_like": "monthly_like (15–30 дней)",
        "low_freq": "low_freq (31–60 дней)",
        "rare": "rare (>60 дней)",
    }
    return mp.get(s, s)


def sort_hbar(fig):
    fig.update_yaxes(categoryorder="total ascending")
    return fig


def top_by_channel(df: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("channel"):
        parts.append(g.sort_values(metric, ascending=False).head(top_n))
    if parts:
        return pd.concat(parts, ignore_index=True)
    return df.head(0).copy()


def show_table(df: pd.DataFrame, rename_map: dict):
    st.dataframe(
        rename_cols(df, rename_map),
        use_container_width=True,
        hide_index=True,
    )


def bar_metric_view(df, metric, x_title, text_template, color_col=None):
    fig = px.bar(
        df.sort_values(metric, ascending=True),
        x=metric,
        y="brand",
        orientation="h",
        text=metric,
        color=color_col,
        color_discrete_sequence=PALETTE,
    )
    xmax = df[metric].max() if metric in df.columns else None
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Бренд",
        xaxis_range=[0, xmax * 1.28] if xmax and xmax > 0 else None,
        margin=dict(l=20, r=90, t=20, b=20),
    )
    fig.update_traces(texttemplate=text_template, textposition="outside", cliponaxis=False)
    sort_hbar(fig)
    apply_theme(fig)
    return fig



def safe_first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]


INVALID_BRANDS = {
    "unknown",
    "no_brand",
    "none",
    "null",
    "nan",
    "other_or_unknown",
    "",
}


def clean_brand_frame(df: pd.DataFrame, brand_col: str = "brand") -> pd.DataFrame:
    if brand_col not in df.columns:
        return df.copy()
    out = df.copy()
    out = out[out[brand_col].notna()].copy()
    out["_brand_norm"] = out[brand_col].astype(str).str.strip().str.lower()
    out = out[~out["_brand_norm"].isin(INVALID_BRANDS)].copy()
    out = out.drop(columns=["_brand_norm"])
    return out


def detect_violette_brand(df: pd.DataFrame, brand_col: str = "brand"):
    if brand_col not in df.columns:
        return None
    vals = df[brand_col].dropna().astype(str).unique().tolist()
    for v in vals:
        if "violette" in v.lower():
            return v
    return None


def ensure_top_with_violette(
    df: pd.DataFrame,
    metric: str,
    top_n: int,
    brand_col: str = "brand",
) -> pd.DataFrame:
    if df.empty or brand_col not in df.columns or metric not in df.columns:
        return df.copy()

    violette_brand = detect_violette_brand(df, brand_col)
    ranked = df.sort_values(metric, ascending=False).copy()

    if violette_brand is None:
        return ranked.head(top_n).copy()

    top = ranked.head(top_n).copy()
    if violette_brand not in top[brand_col].astype(str).tolist():
        v_row = ranked[ranked[brand_col].astype(str) == str(violette_brand)].head(1)
        top = pd.concat([top.head(max(top_n - 1, 0)), v_row], ignore_index=True)

    top["_violette_first"] = (top[brand_col].astype(str) == str(violette_brand)).astype(int)
    top = top.sort_values(["_violette_first", metric], ascending=[False, False]).drop(columns=["_violette_first"])
    return top.copy()


def ensure_top_by_channel_with_violette(
    df: pd.DataFrame,
    metric: str,
    top_n: int,
    channel_col: str = "channel",
    brand_col: str = "brand",
) -> pd.DataFrame:
    if df.empty or channel_col not in df.columns:
        return df.copy()

    parts = []
    for _, g in df.groupby(channel_col):
        parts.append(ensure_top_with_violette(g, metric=metric, top_n=top_n, brand_col=brand_col))

    if parts:
        return pd.concat(parts, ignore_index=True)
    return df.head(0).copy()


# =========================
# Load data
# =========================
DATA_DIR = find_data_dir()
CSV_DIR = DATA_DIR / "task_outputs" / "csv"

FILES = {
    "task0_brand_summary": "task0_brand_summary_overall.csv",
    "task1": "task1_switch_rank_overall.csv",
    "task2": "task2_churn_switch_rank.csv",
    "task3_raw": "task3_copurchase_products.csv",
    "task3_real": "task3_copurchase_products_real_only.csv",
    "task3_grouped": "task3_copurchase_products_grouped.csv",
    "task4_reg": "task4_buyer_regularity.csv",
    "task4_seg": "task4_regularity_segments.csv",
    "task5_switch_marketplace": "task5_switch_rank_by_marketplace.csv",
    "task5_switch_month": "task5_switch_rank_by_month.csv",
    "task5_switch_flavor": "task5_switch_rank_by_flavor.csv",
    "task5_switch_pack": "task5_switch_rank_by_pack.csv",
    "task5_copurchase_marketplace": "task5_copurchase_by_marketplace.csv",
    "task5_copurchase_marketplace_grouped": "task5_copurchase_by_marketplace_grouped.csv",
    "task5_regularity_marketplace": "task5_regularity_by_marketplace.csv",
}

frames = {k: load_csv(CSV_DIR / v) for k, v in FILES.items()}

f0_brand = frames["task0_brand_summary"].copy()
f1 = add_brand_shares(frames["task1"], "violette_buyer_base")
f2 = add_brand_shares(frames["task2"], "churn_buyer_base")
f3_raw = frames["task3_raw"].copy()
f3_real = frames["task3_real"].copy()
f3_group = frames["task3_grouped"].copy()
f4_reg = frames["task4_reg"].copy()
f4_seg = frames["task4_seg"].copy()
f5_sw_mp = frames["task5_switch_marketplace"].copy()
f5_month = frames["task5_switch_month"].copy()
f5_flavor = frames["task5_switch_flavor"].copy()
f5_pack = frames["task5_switch_pack"].copy()
f5_cop_marketplace = frames["task5_copurchase_marketplace"].copy()
f5_cop_group = frames["task5_copurchase_marketplace_grouped"].copy()
f5_reg_mp = frames["task5_regularity_marketplace"].copy()

# Чистим брендовые витрины от unknown / no_brand / мусорных значений
f0_brand = clean_brand_frame(f0_brand) if "f0_brand" in locals() else f0_brand
f1 = clean_brand_frame(f1)
f2 = clean_brand_frame(f2)
f5_sw_mp = clean_brand_frame(f5_sw_mp)
f5_month = clean_brand_frame(f5_month)
f5_flavor = clean_brand_frame(f5_flavor)
f5_pack = clean_brand_frame(f5_pack)

for df in [f5_sw_mp, f5_cop_group, f5_reg_mp]:
    if "is_marketplace" in df.columns:
        df["channel"] = df["is_marketplace"].apply(infer_channel_label)

service_groups = ["Пакет", "Доставка", "Сборка"]
if "product_group" in f5_cop_group.columns:
    f5_cop_group["group_norm"] = f5_cop_group["product_group"].astype(str).str.strip().str.lower()
    f5_cop_group["is_service_group"] = f5_cop_group["group_norm"].isin(["пакет", "доставка", "сборка"])

flavor_name_map = {
    "plain": "Без добавок",
    "other_or_unknown": "Прочее / неизвестно",
    "greens_herbs": "Зелень и травы",
    "cooking": "Для готовки",
    "cucumber": "Огурец",
    "avocado": "Авокадо",
    "light": "Лёгкий",
    "garlic": "Чеснок",
    "mushrooms_truffle": "Грибы / трюфель",
    "ham_bacon": "Ветчина / бекон",
    "mascarpone": "Маскарпоне",
    "spicy": "Острый",
    "fish": "Рыбный",
}

pack_name_map = {
    "small": "Маленькая упаковка",
    "medium": "Средняя упаковка",
    "large": "Большая упаковка",
    "unknown": "Неизвестный размер",
}


# =========================
# Sidebar
# =========================
st.sidebar.title("Karat Violette")
st.sidebar.caption("Дашборд по заданиям 1–5")

page = st.sidebar.radio(
    "Раздел",
    [
        "Обзор",
        "1. Переключение брендов",
        "2. Переключение оттока",
        "3. Анализ корзины",
        "4. Регулярность",
        "5. Разрезы: канал / месяц / вкус / упаковка",
        "Таблицы",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("**Папка данных**")
st.sidebar.code(str(DATA_DIR))


# =========================
# Overview
# =========================
if page == "Обзор":
    st.title("🧀 Дашборд анализа Karat Violette")
    st.markdown(
        "Этот дашборд объединяет результаты по конкурентному окружению бренда, "
        "переключению оттока, анализу корзины, регулярности покупок и дополнительным "
        "разрезам по каналу, месяцам, вкусу и упаковке."
    )

    violette_base = int(f1["violette_buyer_base"].iloc[0])
    churn_base = int(f2["churn_buyer_base"].iloc[0])
    cream_base = int(f3_group["cream_check_base"].iloc[0]) if "cream_check_base" in f3_group.columns else np.nan
    repeat_buyers = len(f4_reg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Покупатели Violette", fmt_int(violette_base))
    c2.metric("База оттока", fmt_int(churn_base))
    c3.metric("Чеки с творожными сырами", fmt_int(cream_base))
    c4.metric("Повторные покупатели Violette", fmt_int(repeat_buyers))

    col1, col2 = st.columns(2)

    with col1:
        section_header(
            "Главные конкуренты Violette",
            "База: покупатели брендов в категории творожных сыров; неизвестные и безбрендовые значения исключены."
        )

        violette_brand = detect_violette_brand(f0_brand)
        if violette_brand is not None:
            rank_df = f0_brand.sort_values("n_buyers", ascending=False).reset_index(drop=True).copy()
            rank_df["rank"] = rank_df.index + 1
            violette_row = rank_df[rank_df["brand"].astype(str) == str(violette_brand)].head(1)
            if not violette_row.empty:
                st.caption(
                    f"Violette в общем рейтинге: #{int(violette_row['rank'].iloc[0])} "
                    f"из {len(rank_df)} брендов; покупателей: {fmt_int(violette_row['n_buyers'].iloc[0])}."
                )

        top = ensure_top_with_violette(f0_brand, metric="n_buyers", top_n=8, brand_col="brand")

        fig = px.bar(
            top.sort_values("n_buyers", ascending=True),
            x="n_buyers",
            y="brand",
            orientation="h",
            text="n_buyers",
            color=top["brand"].astype(str).str.contains("violette", case=False, na=False).map({True: "Violette", False: "Другие бренды"}),
            color_discrete_map={"Violette": "#EC4899", "Другие бренды": "#60A5FA"},
        )
        fig.update_layout(
            height=430,
            xaxis_title="Число покупателей",
            yaxis_title="Бренд",
            xaxis_range=[0, top["n_buyers"].max() * 1.28],
            margin=dict(l=20, r=90, t=30, b=20),
            showlegend=False,
        )
        fig.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
            marker_line_width=0,
            cliponaxis=False,
        )
        sort_hbar(fig)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div style="height:72px;"><h2 style="margin:0;">Сегменты регулярности</h2></div>', unsafe_allow_html=True)
        st.caption(freq_caption())
        seg_col = safe_first_col(f4_seg, ["segment"])
        n_col = safe_first_col(f4_seg, ["n_buyers"])
        plot_seg = f4_seg.copy()
        plot_seg[seg_col] = plot_seg[seg_col].map(pretty_seg)

        fig = px.bar(
            plot_seg,
            x=seg_col,
            y=n_col,
            text=n_col,
            color=seg_col,
            color_discrete_map={
                "high_freq (≤14 дней)": "#EC4899",
                "monthly_like (15–30 дней)": "#A78BFA",
                "low_freq (31–60 дней)": "#60A5FA",
                "rare (>60 дней)": "#2563EB",
            },
        )
        fig.update_layout(
            height=430,
            xaxis_title="Сегмент частотности",
            yaxis_title="Число покупателей",
            showlegend=False,
            margin=dict(l=20, r=40, t=30, b=20),
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    section_header("Короткие выводы")
    st.markdown(
        "- **Hochland** — главный конкурент по пересечению аудитории.\n"
        "- **Переключение оттока** перераспределяется прежде всего в сторону Hochland, Sveza и Almette.\n"
        "- **Анализ корзины** показывает склонность покупать творожный сыр для завтрака и свежих перекусов.\n"
        "- На маркетплейсах корзину сильно искажают сервисные строки: доставка, пакеты, сборка.\n"
        "- У Violette есть и частотное ядро, и большая доля единичных покупателей."
    )


# =========================
# Task 1
# =========================
elif page == "1. Переключение брендов":
    st.title("1. На какие бренды переключается аудитория Violette")
    st.caption(freq_caption())

    base = int(f1["violette_buyer_base"].iloc[0])
    top_n = st.slider("Сколько брендов показать", 5, 20, 10)
    view = st.radio(
        "Метрика",
        ["Покупатели", "Доля от базы", "Чеки", "Сумма", "Количество"],
        horizontal=True,
    )

    df = f1.sort_values("n_buyers", ascending=False).head(top_n).copy()

    if view == "Покупатели":
        fig = bar_metric_view(df, "n_buyers", "Число покупателей", "%{text:,.0f}")
    elif view == "Доля от базы":
        fig = bar_metric_view(df, "share_of_base", "Доля от базы покупателей Violette", "%{text:.1%}")
    elif view == "Чеки":
        fig = bar_metric_view(df, "n_checks", "Число чеков", "%{text:,.0f}")
    elif view == "Сумма":
        fig = bar_metric_view(df, "total_amount", "Сумма продаж", "%{text:,.0f}")
    else:
        fig = bar_metric_view(df, "total_qty", "Количество", "%{text:,.0f}")

    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("База покупателей Violette", fmt_int(base))
    c2.metric("Лидер", df.iloc[0]["brand"])

    show_table(
        df[[c for c in ["brand", "n_buyers", "share_of_base", "n_checks", "total_amount", "total_qty"] if c in df.columns]],
        {
            "brand": "Бренд (brand)",
            "n_buyers": "Число покупателей (n_buyers)",
            "share_of_base": "Доля от базы (share_of_base)",
            "n_checks": "Число чеков (n_checks)",
            "total_amount": "Сумма продаж (total_amount)",
            "total_qty": "Количество (total_qty)",
        },
    )


# =========================
# Task 2
# =========================
elif page == "2. Переключение оттока":
    st.title("2. Куда уходят покупатели, переставшие покупать Violette")
    st.caption(freq_caption())

    base = int(f2["churn_buyer_base"].iloc[0])
    top_n = st.slider("Сколько брендов показать", 5, 20, 10, key="t2")
    view = st.radio(
        "Метрика",
        ["Покупатели", "Доля от базы", "Чеки", "Сумма", "Количество"],
        horizontal=True,
        key="t2view",
    )

    df = f2.sort_values("n_buyers", ascending=False).head(top_n).copy()

    if view == "Покупатели":
        fig = bar_metric_view(df, "n_buyers", "Число покупателей из базы оттока", "%{text:,.0f}")
    elif view == "Доля от базы":
        fig = bar_metric_view(df, "share_of_base", "Доля от базы оттока", "%{text:.1%}")
    elif view == "Чеки":
        fig = bar_metric_view(df, "n_checks", "Число чеков", "%{text:,.0f}")
    elif view == "Сумма":
        fig = bar_metric_view(df, "total_amount", "Сумма продаж", "%{text:,.0f}")
    else:
        fig = bar_metric_view(df, "total_qty", "Количество", "%{text:,.0f}")

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("База оттока", fmt_int(base))
    c2.metric("Лидер оттока", df.iloc[0]["brand"])
    c3.metric("Доля лидера", fmt_pct(df.iloc[0]["share_of_base"]))

    show_table(
        df[[c for c in ["brand", "n_buyers", "share_of_base", "n_checks", "total_amount", "total_qty"] if c in df.columns]],
        {
            "brand": "Бренд (brand)",
            "n_buyers": "Число покупателей (n_buyers)",
            "share_of_base": "Доля от базы (share_of_base)",
            "n_checks": "Число чеков (n_checks)",
            "total_amount": "Сумма продаж (total_amount)",
            "total_qty": "Количество (total_qty)",
        },
    )


# =========================
# Task 3
# =========================
elif page == "3. Анализ корзины":
    st.title("3. Какие товары попадают в чек вместе с творожными сырами")
    st.caption(freq_caption())

    mode = st.radio(
        "Срез",
        ["Без фильтрации", "Только реальные товары", "Укрупнённые товарные группы"],
        horizontal=True,
    )
    top_n = st.slider("Сколько позиций показать", 5, 20, 12, key="t3")

    if mode == "Без фильтрации":
        df = f3_raw.sort_values("n_checks", ascending=False).head(top_n).copy()
        name_col = safe_first_col(df, ["product_name_clean"])
        title = "Top SKU без фильтрации"
    elif mode == "Только реальные товары":
        df = f3_real.sort_values("n_checks", ascending=False).head(top_n).copy()
        name_col = safe_first_col(df, ["product_name_clean"])
        title = "Top реальных товаров"
    else:
        df = f3_group.sort_values("n_checks", ascending=False).head(top_n).copy()
        name_col = safe_first_col(df, ["product_group"])
        title = "Top товарных групп"

    fig = px.bar(
        df.sort_values("n_checks", ascending=True),
        x="n_checks",
        y=name_col,
        orientation="h",
        text="n_checks",
        title=title,
        color_discrete_sequence=[PALETTE[3]],
    )
    fig.update_layout(xaxis_title="Число чеков", yaxis_title="")
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    sort_hbar(fig)
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    if "cream_check_base" in df.columns:
        st.metric("База чеков с творожными сырами", fmt_int(int(df["cream_check_base"].iloc[0])))

    label = "Товарная группа (product_group)" if name_col == "product_group" else f"Товар ({name_col})"
    show_table(
        df[[c for c in [name_col, "n_checks", "n_buyers", "check_share_of_cream_base"] if c in df.columns]],
        {
            name_col: label,
            "n_checks": "Число чеков (n_checks)",
            "n_buyers": "Число покупателей (n_buyers)",
            "check_share_of_cream_base": "Доля от базы чеков с творожными сырами (check_share_of_cream_base)",
        },
    )


# =========================
# Task 4
# =========================
elif page == "4. Регулярность":
    st.title("4. Регулярность")
    st.caption(freq_caption())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Повторные покупатели", fmt_int(len(f4_reg)))
    c2.metric(
        "Медиана медианных интервалов",
        f"{f4_reg['median_gap_days'].median():.1f} дн." if "median_gap_days" in f4_reg.columns else "—",
    )
    c3.metric(
        "25-й перцентиль медианных интервалов",
        f"{f4_reg['median_gap_days'].quantile(0.25):.1f} дн." if "median_gap_days" in f4_reg.columns else "—",
    )
    c4.metric(
        "75-й перцентиль медианных интервалов",
        f"{f4_reg['median_gap_days'].quantile(0.75):.1f} дн." if "median_gap_days" in f4_reg.columns else "—",
    )

    col1, col2 = st.columns(2)

    with col1:
        seg_col = safe_first_col(f4_seg, ["segment"])
        n_col = safe_first_col(f4_seg, ["n_buyers"])
        plot_seg = f4_seg.copy()
        plot_seg[seg_col] = plot_seg[seg_col].map(pretty_seg)

        fig = px.bar(
            plot_seg,
            x=seg_col,
            y=n_col,
            text=n_col,
            color=seg_col,
            color_discrete_map={
                "high_freq (≤14 дней)": "#EC4899",
                "monthly_like (15–30 дней)": "#A78BFA",
                "low_freq (31–60 дней)": "#60A5FA",
                "rare (>60 дней)": "#2563EB",
            },
        )
        fig.update_layout(
            xaxis_title="Сегмент частотности",
            yaxis_title="Число покупателей",
            height=420,
            showlegend=False,
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "median_gap_days" in f4_reg.columns:
            dist_mode = st.radio(
                "Вид распределения",
                ["Boxplot", "Violin"],
                horizontal=True,
                key="reg_dist_mode",
            )

            if dist_mode == "Boxplot":
                fig = px.box(
                    f4_reg,
                    y="median_gap_days",
                    points="outliers",
                    color_discrete_sequence=["#60A5FA"],
                )
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Медианный интервал между покупками, дни (median_gap_days)",
                    height=420,
                )
            else:
                fig = px.violin(
                    f4_reg,
                    y="median_gap_days",
                    box=True,
                    points="outliers",
                    color_discrete_sequence=["#60A5FA"],
                )
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Медианный интервал между покупками, дни (median_gap_days)",
                    height=420,
                )

            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    seg_col = safe_first_col(f4_seg, ["segment"])
    n_col = safe_first_col(f4_seg, ["n_buyers"])
    table_seg = f4_seg.copy()
    table_seg[seg_col] = table_seg[seg_col].map(pretty_seg)

    show_table(
        table_seg,
        {
            seg_col: "Сегмент (segment)",
            n_col: "Число покупателей (n_buyers)",
        },
    )


# =========================
# Task 5
# =========================
elif page == "5. Разрезы: канал / месяц / вкус / упаковка":
    st.title("5. Дополнительные разрезы")

    tab_channel, tab_month, tab_flavor, tab_pack, tab_reg = st.tabs(
        ["Канал", "Месяцы", "Вкусы", "Упаковки", "Регулярность по каналам"]
    )

    with tab_channel:
        section_header(
            "Переключение по каналам",
            "База: покупатели брендов внутри категории творожных сыров по каждому каналу; Violette принудительно добавляется для сравнения, даже если не входит в top."
        )

        top_n = st.slider("Сколько брендов показать", 5, 15, 8, key="t5ch")
        channels = f5_sw_mp["channel"].dropna().unique().tolist() if "channel" in f5_sw_mp.columns else []
        selected = st.multiselect("Каналы", channels, default=channels)

        df = f5_sw_mp[f5_sw_mp["channel"].isin(selected)].copy()
        if not df.empty:
            plot_df = ensure_top_by_channel_with_violette(df, metric="n_buyers", top_n=top_n, channel_col="channel", brand_col="brand")

            fig = px.bar(
                plot_df,
                x="n_buyers",
                y="brand",
                color="channel",
                facet_col="channel",
                facet_col_wrap=2,
                orientation="h",
                color_discrete_map={
                    "Маркетплейс": "#EC4899",
                    "Не маркетплейс": "#60A5FA",
                },
            )
            fig.update_layout(
                height=540,
                xaxis_title="Число покупателей",
                yaxis_title="Бренд",
                margin=dict(l=30, r=80, t=50, b=30),
            )
            sort_hbar(fig)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            show_table(
                plot_df[[c for c in ["channel", "brand", "n_buyers", "n_checks"] if c in plot_df.columns]],
                {
                    "channel": "Канал (channel)",
                    "brand": "Бренд (brand)",
                    "n_buyers": "Число покупателей (n_buyers)",
                    "n_checks": "Число чеков (n_checks)",
                },
            )

        section_header("Анализ корзины по каналам")

        if not f5_cop_group.empty:
            include_service = st.toggle("Показывать сервисные строки", value=True, key="srv")
            temp = f5_cop_group.copy()

            if "is_service_group" in temp.columns and not include_service:
                temp = temp[~temp["is_service_group"]]

            plot_df = top_by_channel(temp, "n_checks", 10)

            fig = px.bar(
                plot_df,
                x="n_checks",
                y="product_group",
                color="channel",
                facet_col="channel",
                facet_col_wrap=2,
                orientation="h",
                color_discrete_map={
                    "Маркетплейс": "#EC4899",
                    "Не маркетплейс": "#60A5FA",
                },
            )
            fig.update_layout(
                height=620,
                xaxis_title="Число чеков",
                yaxis_title="Товарная группа",
                margin=dict(l=30, r=80, t=50, b=30),
            )
            sort_hbar(fig)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            show_table(
                plot_df[[c for c in ["channel", "product_group", "n_checks"] if c in plot_df.columns]],
                {
                    "channel": "Канал (channel)",
                    "product_group": "Товарная группа (product_group)",
                    "n_checks": "Число чеков (n_checks)",
                },
            )

    with tab_month:
        section_header(
            "Месячная динамика конкурентного окружения",
            "База: покупатели брендов внутри категории творожных сыров по месяцам; показаны top-10 брендов по суммарному числу покупателей, при этом Violette добавляется отдельно для сравнения, если присутствует в витрине."
        )

        if not f5_month.empty:
            month_df = f5_month.copy()

            brand_col = safe_first_col(month_df, ["brand"])
            month_col = safe_first_col(month_df, ["year_month"])
            buyers_col = safe_first_col(month_df, ["n_buyers"])

            # чистим мусорные бренды
            month_df = clean_brand_frame(month_df, brand_col=brand_col)

            # ищем Violette по подстроке
            brand_values = month_df[brand_col].dropna().astype(str).unique().tolist()
            violette_candidates = [b for b in brand_values if "violette" in b.lower()]
            violette_brand = violette_candidates[0] if violette_candidates else None

            month_totals = (
                month_df.groupby(brand_col, as_index=False)[buyers_col]
                .sum()
                .sort_values(buyers_col, ascending=False)
            )

            top_brands = month_totals.head(10)[brand_col].tolist()

            if violette_brand is not None and violette_brand not in top_brands:
                top_brands = top_brands[:9] + [violette_brand]

            month_df = month_df[month_df[brand_col].isin(top_brands)].copy()

            pivot = month_df.pivot_table(
                index=brand_col,
                columns=month_col,
                values=buyers_col,
                fill_value=0,
            )

            ordered_brands = (
                month_df.groupby(brand_col, as_index=False)[buyers_col]
                .sum()
                .sort_values(buyers_col, ascending=False)[brand_col]
                .tolist()
            )

            if violette_brand is not None and violette_brand in ordered_brands:
                ordered_brands = [violette_brand] + [b for b in ordered_brands if b != violette_brand]

            pivot = pivot.loc[[b for b in ordered_brands if b in pivot.index]]

            fig = px.imshow(
                pivot,
                aspect="auto",
                labels=dict(x="Месяц", y="Бренд", color="Число покупателей"),
                color_continuous_scale="Turbo",
            )
            fig.update_layout(
                height=560,
                margin=dict(l=30, r=30, t=40, b=30),
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            if violette_brand is not None:
                st.caption(
                    f"Violette показан первой строкой для удобства сравнения (в витрине бренд найден как: {violette_brand}). "
                    "Если 2023-11 и 2023-12 отсутствуют, это нужно трактовать аккуратно как особенность агрегированной "
                    "выгрузки или особенность изначальных данных, касающихся этого периода."
                )
            else:
                st.caption(
                    "Violette не найден в текущей месячной витрине под названием бренда, поэтому на heatmap показаны только top-10 брендов. "
                    "Если 2023-11 и 2023-12 отсутствуют, это нужно трактовать аккуратно как особенность агрегированной "
                    "выгрузки или особенность изначальных данных, касающихся этого периода."
                )

    with tab_flavor:
        section_header("Лидеры по вкусам", "База: покупатели брендов внутри каждой вкусовой группы; неизвестные и безбрендовые значения исключены.")

        if not f5_flavor.empty:
            flavor_df = f5_flavor.copy()
            flavor_col = safe_first_col(flavor_df, ["flavor_group"])
            brand_col = safe_first_col(flavor_df, ["brand"])
            buyers_col = safe_first_col(flavor_df, ["n_buyers"])

            flavor_df = (
                flavor_df
                .sort_values([flavor_col, buyers_col], ascending=[True, False])
                .groupby(flavor_col, as_index=False)
                .first()
            )

            flavor_df["flavor_label"] = flavor_df[flavor_col].map(lambda x: flavor_name_map.get(x, x))
            flavor_df["label"] = flavor_df[brand_col].astype(str) + " — " + flavor_df[buyers_col].astype(int).astype(str)

            fig = px.bar(
                flavor_df.sort_values(buyers_col, ascending=True),
                x=buyers_col,
                y="flavor_label",
                orientation="h",
                text="label",
                color=brand_col,
                color_discrete_sequence=PALETTE,
            )
            fig.update_layout(
                height=680,
                xaxis_title="Число покупателей",
                yaxis_title="Вкусовая группа",
                showlegend=False,
                xaxis_range=[0, flavor_df[buyers_col].max() * 1.32],
                margin=dict(l=40, r=130, t=50, b=30),
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            sort_hbar(fig)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.caption("На графике показан только бренд-лидер внутри каждой вкусовой группы.")

            show_table(
                flavor_df[[flavor_col, brand_col, buyers_col]],
                {
                    flavor_col: "Вкусовая группа (flavor_group)",
                    brand_col: "Бренд-лидер (brand)",
                    buyers_col: "Число покупателей (n_buyers)",
                },
            )

    with tab_pack:
        section_header("Лидеры по упаковкам", "База: покупатели брендов внутри каждой группы размера упаковки; неизвестные и безбрендовые значения исключены.")

        if not f5_pack.empty:
            pack_df = f5_pack.copy()
            pack_col = safe_first_col(pack_df, ["pack_bucket"])
            brand_col = safe_first_col(pack_df, ["brand"])
            buyers_col = safe_first_col(pack_df, ["n_buyers"])

            pack_df = (
                pack_df
                .sort_values([pack_col, buyers_col], ascending=[True, False])
                .groupby(pack_col, as_index=False)
                .first()
            )

            pack_df["pack_label"] = pack_df[pack_col].map(lambda x: pack_name_map.get(x, x))
            pack_df["label"] = pack_df[brand_col].astype(str) + " — " + pack_df[buyers_col].astype(int).astype(str)

            fig = px.bar(
                pack_df.sort_values(buyers_col, ascending=True),
                x=buyers_col,
                y="pack_label",
                orientation="h",
                text="label",
                color=brand_col,
                color_discrete_sequence=PALETTE,
            )
            fig.update_layout(
                height=450,
                xaxis_title="Число покупателей",
                yaxis_title="Размер упаковки",
                showlegend=False,
                xaxis_range=[0, pack_df[buyers_col].max() * 1.32],
                margin=dict(l=40, r=130, t=50, b=30),
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            sort_hbar(fig)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.caption("На графике показан только бренд-лидер внутри каждой группы размера упаковки.")

            show_table(
                pack_df[[pack_col, brand_col, buyers_col]],
                {
                    pack_col: "Размер упаковки (pack_bucket)",
                    brand_col: "Бренд-лидер (brand)",
                    buyers_col: "Число покупателей (n_buyers)",
                },
            )

    with tab_reg:
        section_header("Регулярность по каналам", freq_caption())

        if not f5_reg_mp.empty:
            channel_col = safe_first_col(f5_reg_mp, ["channel"])
            median_col = safe_first_col(f5_reg_mp, ["median_gap_days"])
            mean_col = safe_first_col(f5_reg_mp, ["mean_gap_days"])

            # KPI
            cols = st.columns(len(f5_reg_mp))
            for i, (_, row) in enumerate(f5_reg_mp.iterrows()):
                cols[i].metric(
                    f"{row[channel_col]}",
                    f"{fmt_num(row[median_col])} дн.",
                    f"mean: {fmt_num(row[mean_col])} дн.",
                )

            # График сравнения median vs mean
            plot_df = f5_reg_mp[[channel_col, median_col, mean_col]].copy()
            plot_df = plot_df.rename(
                columns={
                    channel_col: "Канал",
                    median_col: "Медианный интервал",
                    mean_col: "Средний интервал",
                }
            )
            melt_df = plot_df.melt(id_vars="Канал", var_name="Метрика", value_name="Дни")

            fig = px.bar(
                melt_df,
                x="Канал",
                y="Дни",
                color="Метрика",
                barmode="group",
                text="Дни",
                color_discrete_sequence=[PALETTE[0], PALETTE[3]],
            )
            fig.update_layout(
                height=420,
                xaxis_title="Канал",
                yaxis_title="Интервал между покупками, дни",
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Здесь одновременно показаны медианный и средний интервалы, чтобы было видно типичный ритм покупки и влияние длинного хвоста редких возвратов."
            )

            # Таблица без is_marketplace и без лишних галочек
            table_df = f5_reg_mp.copy()
            cols_to_drop = [c for c in ["is_marketplace"] if c in table_df.columns]
            if cols_to_drop:
                table_df = table_df.drop(columns=cols_to_drop)

            show_table(
                table_df,
                {
                    channel_col: "Канал (channel)",
                    median_col: "Медианный интервал, дни (median_gap_days)",
                    mean_col: "Средний интервал, дни (mean_gap_days)",
                    "n_gaps": "Число интервалов (n_gaps)",
                },
            )


# =========================
# Raw tables
# =========================
elif page == "Таблицы":
    st.title("Таблицы и выгрузки")

    table_name = st.selectbox("Выберите таблицу", list(frames.keys()))
    df = frames[table_name]

    st.write(f"Строк: {len(df):,}".replace(",", " "))
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Скачать CSV",
        data=csv,
        file_name=f"{table_name}.csv",
        mime="text/csv",
    )