import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Konfigurasi halaman & gaya visual
# ------------------------------------------------------------------
st.set_page_config(page_title="Dashboard E-Commerce Brasil", page_icon="🛒", layout="wide")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRAY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
})

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_data.csv")


# ------------------------------------------------------------------
# Memuat data
# ------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        df[col] = pd.to_datetime(df[col])
    df["is_late"] = df["is_late"].astype(str).str.lower() == "true"
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    return df


# ------------------------------------------------------------------
# Fungsi helper untuk menyiapkan DataFrame
# ------------------------------------------------------------------
def create_orders_df(df):
    """Satu baris per pesanan (untuk metrik pengiriman & ulasan)."""
    return (df.groupby("order_id")
            .agg(customer_unique_id=("customer_unique_id", "first"),
                 customer_state=("customer_state", "first"),
                 order_purchase_timestamp=("order_purchase_timestamp", "first"),
                 order_month=("order_month", "first"),
                 order_value=("price", "sum"),
                 review_score=("review_score", "first"),
                 delivery_days=("delivery_days", "first"),
                 delay_days=("delay_days", "first"),
                 is_late=("is_late", "first"),
                 lat=("geolocation_lat", "first"),
                 lon=("geolocation_lng", "first"))
            .reset_index())


def create_monthly_df(df):
    return (df.groupby("order_month")
            .agg(order_count=("order_id", "nunique"), revenue=("price", "sum"))
            .reset_index())


def create_category_df(df):
    cat = (df.groupby("product_category_name_english")
           .agg(revenue=("price", "sum"), order_count=("order_id", "nunique"))
           .sort_values("revenue", ascending=False)
           .reset_index())
    cat["share"] = cat["revenue"] / cat["revenue"].sum() * 100
    return cat


def create_delay_review_df(orders):
    bins = [-np.inf, -15, -8, -1, 0, 3, 7, 14, np.inf]
    labels = [">14 hr lebih cepat", "8-14 hr lebih cepat", "1-7 hr lebih cepat", "Tepat di hari estimasi",
              "1-3 hr terlambat", "4-7 hr terlambat", "8-14 hr terlambat", ">14 hr terlambat"]
    tmp = orders.copy()
    tmp["delay_group"] = pd.cut(tmp["delay_days"], bins=bins, labels=labels)
    return (tmp.groupby("delay_group", observed=False)
            .agg(order_count=("order_id", "count"), avg_review=("review_score", "mean"))
            .reset_index())


def create_state_df(orders):
    state = (orders.groupby("customer_state")
             .agg(order_count=("order_id", "count"), late_rate=("is_late", "mean"),
                  revenue=("order_value", "sum"), avg_review=("review_score", "mean"))
             .reset_index())
    state["late_rate"] = state["late_rate"] * 100
    return state


def create_rfm_df(df):
    ref_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (df.groupby("customer_unique_id")
           .agg(last_purchase=("order_purchase_timestamp", "max"),
                frequency=("order_id", "nunique"),
                monetary=("price", "sum"))
           .reset_index())
    rfm["recency"] = (ref_date - rfm["last_purchase"]).dt.days
    return rfm.drop(columns="last_purchase")


def segment_rfm(rfm):
    rfm = rfm.copy()
    # rank() mencegah error qcut ketika banyak nilai yang sama
    rfm["R_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    conditions = [
        rfm["frequency"] >= 2,
        (rfm["R_score"] >= 4) & (rfm["M_score"] >= 4),
        rfm["R_score"] >= 4,
        rfm["R_score"] == 3,
        rfm["M_score"] >= 4,
    ]
    choices = ["Pelanggan Loyal", "Baru & Bernilai Tinggi", "Baru Potensial",
               "Perlu Perhatian", "Berisiko Hilang (Nilai Tinggi)"]
    rfm["segment"] = np.select(conditions, choices, default="Hibernasi")
    return rfm


def create_value_bin_df(orders):
    bins = [0, 50, 100, 200, 500, np.inf]
    labels = ["< 50", "50–100", "100–200", "200–500", "> 500"]
    tmp = orders.copy()
    tmp["value_group"] = pd.cut(tmp["order_value"], bins=bins, labels=labels, right=False)
    vb = (tmp.groupby("value_group", observed=False)
          .agg(orders=("order_id", "count"), revenue=("order_value", "sum")))
    vb["order_share"] = vb["orders"] / max(vb["orders"].sum(), 1) * 100
    vb["revenue_share"] = vb["revenue"] / max(vb["revenue"].sum(), 1) * 100
    return vb


def fmt_brl(x):
    if x >= 1e6:
        return f"BRL {x / 1e6:,.2f} jt"
    if x >= 1e3:
        return f"BRL {x / 1e3:,.1f} rb"
    return f"BRL {x:,.0f}"


# ------------------------------------------------------------------
# Sidebar: filter interaktif
# ------------------------------------------------------------------
all_df = load_data(DATA_PATH)
min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()
all_states = sorted(all_df["customer_state"].unique())
all_categories = sorted(all_df["product_category_name_english"].unique())

with st.sidebar:
    st.title("🛒 Olist E-Commerce")
    st.caption("Filter berlaku untuk seluruh visualisasi di dashboard.")

    date_range = st.date_input("Rentang tanggal pembelian", value=[min_date, max_date],
                               min_value=min_date, max_value=max_date)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:  # pengguna baru memilih satu tanggal
        start_date = end_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range

    selected_states = st.multiselect("Negara bagian pelanggan", options=all_states,
                                     placeholder="Semua negara bagian")
    selected_categories = st.multiselect("Kategori produk", options=all_categories,
                                         placeholder="Semua kategori")
    top_n = st.slider("Jumlah kategori ditampilkan", min_value=5, max_value=20, value=10)

mask = (all_df["order_purchase_timestamp"].dt.date >= start_date) & (all_df["order_purchase_timestamp"].dt.date <= end_date)
if selected_states:
    mask &= all_df["customer_state"].isin(selected_states)
if selected_categories:
    mask &= all_df["product_category_name_english"].isin(selected_categories)
main_df = all_df[mask]

st.title("Dashboard Analisis E-Commerce Brasil (Olist)")
st.caption(f"Pesanan berstatus *delivered* • {start_date:%d %b %Y} – {end_date:%d %b %Y} • "
           "Pendapatan = jumlah harga barang (tanpa ongkos kirim)")

if main_df.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini. Silakan ubah rentang tanggal, negara bagian, atau kategori.")
    st.stop()

orders_df = create_orders_df(main_df)
monthly_df = create_monthly_df(main_df)
category_df = create_category_df(main_df)
state_df = create_state_df(orders_df)

# ------------------------------------------------------------------
# KPI utama
# ------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total pesanan", f"{orders_df['order_id'].nunique():,}")
k2.metric("Total pendapatan", fmt_brl(main_df["price"].sum()))
k3.metric("Rata-rata skor ulasan", f"{orders_df['review_score'].mean():.2f} / 5")
k4.metric("Pesanan terlambat", f"{orders_df['is_late'].mean() * 100:.1f}%")

st.divider()

# ------------------------------------------------------------------
# Pertanyaan 1: tren bulanan
# ------------------------------------------------------------------
st.subheader("1. Tren pesanan & pendapatan bulanan")
tab_rev, tab_ord = st.tabs(["Pendapatan", "Jumlah pesanan"])


def line_chart(data, col, ylabel, scale=1.0, unit=""):
    fig, ax = plt.subplots(figsize=(12, 4))
    y = data[col] / scale
    ax.plot(data["order_month"], y, color=BLUE, linewidth=2, marker="o", markersize=5)
    if len(data) > 1:
        i = y.idxmax()
        ax.scatter(data.loc[i, "order_month"], y[i], s=90, color=BLUE, edgecolor="white", linewidth=2, zorder=5)
        ax.annotate(f"Puncak: {data.loc[i, 'order_month']:%b %Y} ({y[i]:,.0f}{unit})",
                    xy=(data.loc[i, "order_month"], y[i]), xytext=(-40, -38), textcoords="offset points",
                    ha="right", fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK_2))
    ax.set_ylim(0, y.max() * 1.15)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))
    return fig


with tab_rev:
    st.pyplot(line_chart(monthly_df, "revenue", "Pendapatan (ribu BRL)", scale=1e3, unit=" rb BRL"))
with tab_ord:
    st.pyplot(line_chart(monthly_df, "order_count", "Jumlah pesanan", unit=" pesanan"))
st.caption("Puncak tertinggi pada data lengkap terjadi pada November 2017 (Black Friday). "
           "Jan–Ags 2018 tumbuh ±140% dibanding Jan–Ags 2017.")

# ------------------------------------------------------------------
# Pertanyaan 2: kategori produk
# ------------------------------------------------------------------
st.subheader("2. Kategori produk dengan pendapatan tertinggi & terendah")
n = min(top_n, len(category_df))
c1, c2 = st.columns(2)

with c1:
    top = category_df.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * n + 1.2))
    ax.barh(top["product_category_name_english"], top["revenue"] / 1e3,
            color=[GRAY] * (len(top) - min(3, len(top))) + [BLUE] * min(3, len(top)), height=0.7)
    for yy, v in enumerate(top["revenue"] / 1e3):
        ax.text(v, yy, f" {v:,.0f}", va="center", fontsize=8, color=INK_2)
    ax.set_title(f"{n} kategori pendapatan TERTINGGI")
    ax.set_xlabel("Pendapatan (ribu BRL)")
    ax.set_xlim(0, top["revenue"].max() / 1e3 * 1.18)
    ax.grid(axis="y", visible=False)
    st.pyplot(fig)

with c2:
    bottom = category_df.tail(n)
    fig, ax = plt.subplots(figsize=(7, 0.42 * n + 1.2))
    ax.barh(bottom["product_category_name_english"], bottom["revenue"] / 1e3,
            color=[GRAY] * (len(bottom) - 1) + [ORANGE], height=0.7)
    for yy, v in enumerate(bottom["revenue"] / 1e3):
        ax.text(v, yy, f" {v:,.1f}", va="center", fontsize=8, color=INK_2)
    ax.set_title(f"{n} kategori pendapatan TERENDAH")
    ax.set_xlabel("Pendapatan (ribu BRL)")
    ax.set_xlim(0, bottom["revenue"].max() / 1e3 * 1.25)
    ax.grid(axis="y", visible=False)
    st.pyplot(fig)

st.caption(f"{n} kategori teratas menyumbang **{category_df.head(n)['share'].sum():.1f}%** dari total pendapatan "
           "pada filter terpilih. Skala sumbu-x kedua grafik berbeda.")

# ------------------------------------------------------------------
# Pertanyaan 3: keterlambatan vs ulasan
# ------------------------------------------------------------------
st.subheader("3. Keterlambatan pengiriman & skor ulasan")
c1, c2 = st.columns([1, 1])

with c1:
    delay_df = create_delay_review_df(orders_df)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = [ORANGE if "terlambat" in str(g) else BLUE for g in delay_df["delay_group"]]
    ax.bar(delay_df["delay_group"].astype(str), delay_df["avg_review"].fillna(0), color=colors, width=0.7)
    for i, v in enumerate(delay_df["avg_review"]):
        if pd.notna(v):
            ax.text(i, v + 0.07, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Rata-rata skor ulasan (1–5)")
    ax.set_title("Skor ulasan vs selisih tanggal tiba terhadap estimasi")
    ax.tick_params(axis="x", rotation=35)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.grid(axis="x", visible=False)
    st.pyplot(fig)

    on_time = orders_df.loc[~orders_df["is_late"], "review_score"].mean()
    late = orders_df.loc[orders_df["is_late"], "review_score"].mean()
    m1, m2 = st.columns(2)
    m1.metric("Skor rata-rata: tepat waktu", f"{on_time:.2f}" if pd.notna(on_time) else "–")
    m2.metric("Skor rata-rata: terlambat", f"{late:.2f}" if pd.notna(late) else "–",
              delta=f"{late - on_time:.2f}" if pd.notna(late) and pd.notna(on_time) else None)

with c2:
    min_orders = st.slider("Minimal jumlah pesanan per negara bagian", 0, 500, 100, step=50)
    sp = state_df[state_df["order_count"] >= min_orders].sort_values("late_rate").tail(15)
    if sp.empty:
        st.info("Tidak ada negara bagian yang memenuhi batas minimal pesanan.")
    else:
        top5 = sp.tail(5)["customer_state"].tolist()
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(sp) + 1.3))
        ax.barh(sp["customer_state"], sp["late_rate"],
                color=[ORANGE if s in top5 else GRAY for s in sp["customer_state"]], height=0.7)
        for yy, (v, cnt) in enumerate(zip(sp["late_rate"], sp["order_count"])):
            ax.text(v, yy, f" {v:.1f}% (n={cnt:,})", va="center", fontsize=8, color=INK_2,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
        nat = orders_df["is_late"].mean() * 100
        ax.axvline(nat, color=INK, linestyle="--", linewidth=1, zorder=0)
        ax.set_title(f"Persentase terlambat per negara bagian (rata-rata {nat:.1f}%)")
        ax.set_xlabel("Pesanan terlambat (%)")
        ax.set_xlim(0, max(sp["late_rate"].max(), nat) * 1.35 + 0.1)
        ax.grid(axis="y", visible=False)
        st.pyplot(fig)

# ------------------------------------------------------------------
# Analisis lanjutan: RFM
# ------------------------------------------------------------------
st.divider()
st.subheader("4. Analisis lanjutan: RFM & segmentasi pelanggan")
rfm_df = create_rfm_df(main_df)

r1, r2, r3 = st.columns(3)
r1.metric("Rata-rata recency (hari)", f"{rfm_df['recency'].mean():.1f}")
r2.metric("Rata-rata frequency", f"{rfm_df['frequency'].mean():.2f}")
r3.metric("Rata-rata monetary", fmt_brl(rfm_df["monetary"].mean()))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
rfm_df["short_id"] = rfm_df["customer_unique_id"].str[:8]
for ax, (col, asc, title) in zip(axes, [("recency", True, "Recency terbaik (hari)"),
                                        ("frequency", False, "Frequency tertinggi"),
                                        ("monetary", False, "Monetary tertinggi (BRL)")]):
    t = rfm_df.sort_values(col, ascending=asc).head(5).iloc[::-1]
    ax.barh(t["short_id"], t[col], color=[GRAY] * (len(t) - 1) + [BLUE], height=0.6)
    for yy, v in enumerate(t[col]):
        ax.text(v, yy, f" {v:,.0f}", va="center", fontsize=8, color=INK_2)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, max(t[col].max(), 1) * 1.3)
    ax.grid(axis="y", visible=False)
plt.tight_layout()
st.pyplot(fig)

if len(rfm_df) >= 5:
    seg_df = segment_rfm(rfm_df)
    order = ["Pelanggan Loyal", "Baru & Bernilai Tinggi", "Baru Potensial",
             "Perlu Perhatian", "Berisiko Hilang (Nilai Tinggi)", "Hibernasi"]
    seg = (seg_df.groupby("segment").agg(customers=("customer_unique_id", "count"), revenue=("monetary", "sum"))
           .reindex(order).fillna(0))
    seg["Porsi pelanggan (%)"] = seg["customers"] / seg["customers"].sum() * 100
    seg["Porsi pendapatan (%)"] = seg["revenue"] / max(seg["revenue"].sum(), 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    s = seg.iloc[::-1]
    col = [BLUE if i in ("Berisiko Hilang (Nilai Tinggi)", "Pelanggan Loyal") else GRAY for i in s.index]
    for ax, c in zip(axes, ["Porsi pelanggan (%)", "Porsi pendapatan (%)"]):
        ax.barh(s.index, s[c], color=col, height=0.65)
        for yy, v in enumerate(s[c]):
            ax.text(v, yy, f" {v:.1f}%", va="center", fontsize=8, color=INK_2)
        ax.set_title(c, fontsize=11)
        ax.set_xlim(0, max(seg[["Porsi pelanggan (%)", "Porsi pendapatan (%)"]].max().max() * 1.25, 1))
        ax.grid(axis="y", visible=False)
    plt.tight_layout()
    st.pyplot(fig)
    with st.expander("Aturan segmentasi (manual grouping)"):
        st.markdown(
            "- **R-score** & **M-score**: kuintil 1–5 dari recency (makin baru makin tinggi) dan monetary.\n"
            "- **Pelanggan Loyal**: ≥ 2 pesanan.\n"
            "- **Baru & Bernilai Tinggi**: R ≥ 4 dan M ≥ 4 • **Baru Potensial**: R ≥ 4 dan M < 4.\n"
            "- **Perlu Perhatian**: R = 3.\n"
            "- **Berisiko Hilang (Nilai Tinggi)**: R ≤ 2 dan M ≥ 4 • **Hibernasi**: R ≤ 2 dan M < 4."
        )

# ------------------------------------------------------------------
# Analisis lanjutan: geospasial & binning
# ------------------------------------------------------------------
st.divider()
st.subheader("5. Analisis lanjutan: persebaran geografis & kelompok nilai pesanan")
c1, c2 = st.columns([1.2, 1])

with c1:
    st.markdown("**Lokasi pelanggan** (sampel maks. 10.000 pesanan)")
    geo = orders_df.dropna(subset=["lat", "lon"])[["lat", "lon"]]
    if len(geo) > 10000:
        geo = geo.sample(10000, random_state=42)
    st.map(geo, size=20, color="#2a78d6")

with c2:
    vb = create_value_bin_df(orders_df)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(vb))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, vb["order_share"], width=w, color=GRAY, label="Porsi pesanan")
    ax.bar(x + w / 2 + 0.01, vb["revenue_share"], width=w, color=BLUE, label="Porsi pendapatan")
    for i, (a, b) in enumerate(zip(vb["order_share"], vb["revenue_share"])):
        ax.text(i - w / 2, a + 0.6, f"{a:.0f}%", ha="center", fontsize=8, color=INK_2)
        ax.text(i + w / 2, b + 0.6, f"{b:.0f}%", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, [f"{l} BRL" for l in vb.index.astype(str)])
    ax.set_ylabel("Persentase (%)")
    ax.set_ylim(0, max(vb["order_share"].max(), vb["revenue_share"].max()) * 1.2 + 1)
    ax.set_title("Binning nilai pesanan: porsi pesanan vs pendapatan")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    st.pyplot(fig)

top_states = state_df.sort_values("revenue", ascending=False).head(5)
st.caption("5 negara bagian dengan pendapatan terbesar: " +
           ", ".join(f"{r.customer_state} ({fmt_brl(r.revenue)})" for r in top_states.itertuples()))

st.divider()
st.caption("© 2026 Arby Ali Amaludin • Proyek Analisis Data Dicoding • Sumber data: Olist Brazilian E-Commerce Public Dataset")
