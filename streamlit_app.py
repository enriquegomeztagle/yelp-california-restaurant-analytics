import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from data_preparation import ALLOWED_CATEGORIES

COLOR_SEQ_BLUES = px.colors.sequential.Blues
COLOR_DIV_RED_BLUE = ["#B2182B", "#4393C3", "#053061"]
COLOR_CATEGORICAL = px.colors.qualitative.Set2
COLOR_HIGHLIGHT = "#F39C12"
COLOR_ALERT = "#E74C3C"
COLOR_BLUE_ACCENT = "#1B4F72"

st.set_page_config(
    page_title="Análisis EDA – Restaurantes en California",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 360px;
        max-width: 360px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("filtered_yelp_CA.csv")
    return df


def get_category_columns(df: pd.DataFrame, exclude_cols: list) -> list:
    category_columns = [
        col
        for col in df.columns
        if df[col].dropna().isin([0, 1]).all() 
        and col not in exclude_cols
        and col in ALLOWED_CATEGORIES
    ]
    return category_columns


def prepare_data(df: pd.DataFrame) -> dict:
    df["high_rating"] = (df["stars"] >= 4).astype(int)

    exclude_cols = [
        "is_open",
        "Restaurants",
        "high_rating",
        "is_open_binary",
        "popular",
        "is_open_target",
    ]

    category_columns = get_category_columns(df, exclude_cols)

    df["num_categories"] = df[category_columns].sum(axis=1)

    category_counts = df[category_columns].sum().sort_values(ascending=False).head(10)

    top_cities = df["city"].value_counts().head(10)

    avg_rating_by_city = (
        df[df["city"].isin(top_cities.index)]
        .groupby("city")["stars"]
        .mean()
        .sort_values(ascending=False)
    )

    proportion_high_rating_by_city = (
        df[df["city"].isin(top_cities.index)]
        .groupby("city")["high_rating"]
        .mean()
        .sort_values(ascending=False)
    )

    return {
        "df": df,
        "category_columns": category_columns,
        "category_counts": category_counts,
        "avg_rating_by_city": avg_rating_by_city,
        "proportion_high_rating_by_city": proportion_high_rating_by_city,
    }


@st.cache_resource
def train_popularity_model(dataframe: pd.DataFrame):
    df_copy = dataframe.copy()

    median_reviews = df_copy["review_count"].median()
    df_copy["popular"] = (df_copy["review_count"] >= median_reviews).astype(int)

    if "is_open" in df_copy.columns:
        df_copy["is_open_binary"] = df_copy["is_open"].astype(int)
    else:
        df_copy["is_open_binary"] = 1

    exclude_cols = [
        "is_open",
        "Restaurants",
        "high_rating",
        "is_open_binary",
        "popular",
        "is_open_target",
    ]
    category_columns = get_category_columns(df_copy, exclude_cols)

    if category_columns:
        df_copy["num_categories"] = df_copy[category_columns].sum(axis=1)
    else:
        df_copy["num_categories"] = 0

    feature_cols = [
        "stars",
        "num_categories",
        "latitude",
        "longitude",
        "is_open_binary",
        "city",
    ] + category_columns

    X = df_copy[feature_cols]
    y = df_copy["popular"]

    categorical_cols = ["city"]
    numeric_cols = ["stars", "num_categories", "latitude", "longitude"]
    binary_cols = ["is_open_binary"] + category_columns

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("bin", "passthrough", binary_cols),
        ]
    )

    classifier = GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        random_state=42,
    )

    pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])

    pipeline.fit(X, y)

    return pipeline, category_columns


def show_eda_statistics(df: pd.DataFrame):
    st.subheader("Análisis Estadístico")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Información del Dataset:**")
        st.write(f"- Total de registros: {len(df)}")
        st.write(f"- Total de columnas: {len(df.columns)}")
        missing = df.isna().sum().sum()
        st.write(f"- Valores faltantes totales: {missing}")

    with col2:
        st.markdown("**Tipos de Datos:**")
        dtypes_df = df.dtypes.value_counts().reset_index()
        dtypes_df.columns = ["Tipo", "Cantidad"]
        st.dataframe(dtypes_df, hide_index=True)

    st.markdown("**Estadísticas Descriptivas:**")
    numeric_cols = ["stars", "review_count", "latitude", "longitude"]
    available_numeric = [col for col in numeric_cols if col in df.columns]
    if available_numeric:
        stats_df = df[available_numeric].describe().round(2)
        st.dataframe(stats_df)


def show_correlation_matrix(df: pd.DataFrame):
    st.subheader("Matriz de Correlación")
    st.markdown(
        """
    **Correlación:** R ∈ [-1, 1]
    - Positiva (azul): X aumenta → Y aumenta
    - Negativa (rojo): X aumenta → Y disminuye
    """
    )

    numeric_cols = ["stars", "review_count", "num_categories"]
    if "latitude" in df.columns:
        numeric_cols.append("latitude")
    if "longitude" in df.columns:
        numeric_cols.append("longitude")

    available_cols = [col for col in numeric_cols if col in df.columns]

    if len(available_cols) >= 2:
        corr = df[available_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_masked = corr.mask(mask)

        fig = px.imshow(
            corr_masked,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Matriz de Correlación",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay suficientes columnas numéricas para correlación.")


def show_distribution_stars(df: pd.DataFrame, selected_city: str):
    st.subheader("Distribución de Calificaciones")

    if len(df) == 0:
        st.warning("No hay datos disponibles.")
        return

    stars_count = df["stars"].value_counts().sort_index().reset_index()
    stars_count.columns = ["stars", "count"]
    stars_count["y_pos"] = 1

    title_suffix = f" - {selected_city}" if selected_city != "Todo el estado" else ""

    fig = px.scatter(
        stars_count,
        x="stars",
        y="y_pos",
        size="count",
        color="count",
        title=f"Distribución de calificaciones{title_suffix}",
        labels={"stars": "Calificación (Estrellas)"},
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=50,
    )
    fig.update_traces(marker={"symbol": "star", "line": {"width": 0}})
    fig.update_yaxes(visible=False, range=[0.5, 1.5])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            df,
            x="stars",
            nbins=10,
            title="Histograma de Calificaciones",
            labels={"stars": "Estrellas", "count": "Conteo"},
            color_discrete_sequence=[COLOR_BLUE_ACCENT],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_box = px.box(
            df,
            x="stars",
            title="Diagrama de Caja",
            labels={"stars": "Estrellas"},
            color_discrete_sequence=[COLOR_HIGHLIGHT],
        )
        st.plotly_chart(fig_box, use_container_width=True)


def show_distribution_reviews(df: pd.DataFrame, selected_city: str):
    st.subheader("Distribución del Número de Reseñas")

    if len(df) == 0:
        st.warning("No hay datos disponibles.")
        return

    title_suffix = f" - {selected_city}" if selected_city != "Todo el estado" else ""

    counts, bins = np.histogram(df["review_count"], bins=30)
    total = len(df)
    percentages = (counts / total) * 100

    bin_labels = [
        f"{int(left)}–{int(right)}" for left, right in zip(bins[:-1], bins[1:])
    ]

    hist_df = pd.DataFrame(
        {
            "review_range": bin_labels,
            "count": counts,
            "percentage": percentages,
        }
    )

    fig = px.bar(
        hist_df,
        x="review_range",
        y="count",
        color="count",
        title=f"Distribución de reseñas{title_suffix}",
        labels={"review_range": "Rango de reseñas", "count": "Cantidad"},
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    fig.update_layout(height=500, xaxis={"tickangle": -30})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mediana", f"{df['review_count'].median():.0f}")
    with col2:
        st.metric("Promedio", f"{df['review_count'].mean():.1f}")
    with col3:
        iqr = df["review_count"].quantile(0.75) - df["review_count"].quantile(0.25)
        st.metric("IQR", f"{iqr:.0f}")


def show_reviews_vs_stars(df: pd.DataFrame, selected_city: str):
    st.subheader("Relación Reseñas vs Calificación")

    if len(df) == 0:
        st.warning("No hay datos disponibles.")
        return

    title_suffix = f" - {selected_city}" if selected_city != "Todo el estado" else ""

    fig = px.scatter(
        df,
        x="review_count",
        y="stars",
        hover_data=["name", "city"],
        opacity=0.5,
        title=f"Relación reseñas vs calificación{title_suffix}",
        labels={"review_count": "Número de reseñas", "stars": "Estrellas"},
        color_discrete_sequence=[COLOR_HIGHLIGHT],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Análisis de Tendencia (línea de regresión):**")
    fig_trend = px.scatter(
        df,
        x="review_count",
        y="stars",
        trendline="ols",
        opacity=0.3,
        labels={"review_count": "Número de reseñas", "stars": "Estrellas"},
        color_discrete_sequence=[COLOR_HIGHLIGHT],
    )

    for trace in fig_trend.data:
        if trace.mode == "lines":
            trace.line.color = COLOR_ALERT
    st.plotly_chart(fig_trend, use_container_width=True)


def show_high_rating_proportion(
    df: pd.DataFrame, proportion_high_rating_by_city: pd.Series, selected_city: str
):
    st.subheader("Proporción con Alta Calificación")

    if selected_city == "Todo el estado":
        if len(proportion_high_rating_by_city) > 0:
            fig = px.pie(
                names=proportion_high_rating_by_city.index,
                values=proportion_high_rating_by_city.values,
                title="Restaurantes con ≥ 4 estrellas (Top 10 ciudades)",
                color_discrete_sequence=COLOR_CATEGORICAL,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos disponibles.")
    else:
        if len(df) > 0:
            high_count = df["high_rating"].sum()
            total = len(df)
            proportion = high_count / total if total > 0 else 0

            fig = px.pie(
                names=["≥ 4 estrellas", "< 4 estrellas"],
                values=[high_count, total - high_count],
                title=f"Proporción en {selected_city}",
                color_discrete_sequence=COLOR_DIV_RED_BLUE,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(
                f"En {selected_city}: {high_count}/{total} ({proportion*100:.1f}%) tienen ≥ 4 estrellas"
            )
        else:
            st.warning("No hay datos disponibles.")


def show_city_status(df: pd.DataFrame, selected_city: str):
    st.subheader("Estado y Reseñas por Ciudad")

    if len(df) == 0:
        st.warning("No hay datos disponibles.")
        return

    if selected_city == "Todo el estado":
        top_cities = df["city"].value_counts().head(15).index.tolist()
        df_cities = df[df["city"].isin(top_cities)].copy()

        city_status_data = []
        for city in top_cities:
            city_df = df_cities[df_cities["city"] == city]

            abiertos = city_df[city_df["is_open"] == 1]
            cerrados = city_df[city_df["is_open"] == 0]

            city_status_data.append(
                {
                    "Ciudad": city,
                    "Estado": "Abiertos",
                    "Cantidad": len(abiertos),
                    "Prom_reseñas": (
                        abiertos["review_count"].mean() if len(abiertos) > 0 else 0
                    ),
                }
            )
            city_status_data.append(
                {
                    "Ciudad": city,
                    "Estado": "Cerrados",
                    "Cantidad": len(cerrados),
                    "Prom_reseñas": (
                        cerrados["review_count"].mean() if len(cerrados) > 0 else 0
                    ),
                }
            )

        df_status = pd.DataFrame(city_status_data)

        fig = px.bar(
            df_status,
            x="Ciudad",
            y="Cantidad",
            color="Estado",
            title="Abiertos vs Cerrados (Top 15 ciudades)",
            barmode="group",
            color_discrete_map={
                "Abiertos": COLOR_BLUE_ACCENT,
                "Cerrados": COLOR_HIGHLIGHT,
            },
        )
        fig.update_layout(xaxis={"tickangle": -90}, height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detalles por Ciudad")
        summary = []
        for city in top_cities:
            city_df = df_cities[df_cities["city"] == city]
            abiertos = city_df[city_df["is_open"] == 1]
            cerrados = city_df[city_df["is_open"] == 0]
            summary.append(
                {
                    "Ciudad": city,
                    "Total": len(city_df),
                    "Abiertos": len(abiertos),
                    "Cerrados": len(cerrados),
                    "% Abiertos": f"{len(abiertos)/len(city_df)*100:.1f}%",
                }
            )
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    else:
        abiertos = df[df["is_open"] == 1]
        cerrados = df[df["is_open"] == 0]

        status_data = pd.DataFrame(
            {
                "Estado": ["Abiertos", "Cerrados"],
                "Cantidad": [len(abiertos), len(cerrados)],
                "Prom_reseñas": [
                    abiertos["review_count"].mean() if len(abiertos) > 0 else 0,
                    cerrados["review_count"].mean() if len(cerrados) > 0 else 0,
                ],
            }
        )

        fig = px.bar(
            status_data,
            x="Estado",
            y="Cantidad",
            color="Estado",
            title=f"Abiertos vs Cerrados en {selected_city}",
            color_discrete_map={
                "Abiertos": COLOR_BLUE_ACCENT,
                "Cerrados": COLOR_HIGHLIGHT,
            },
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        total = len(df)
        st.info(
            f"**{selected_city}:** {len(abiertos)} abiertos ({len(abiertos)/total*100:.1f}%), {len(cerrados)} cerrados ({len(cerrados)/total*100:.1f}%)"
        )


def show_prediction_interface(
    model, category_columns_pred: list, df_full: pd.DataFrame
):
    st.subheader("Predicción de Popularidad")
    st.markdown(
        "Modelo: GradientBoostingClassifier - Predice si un restaurante será popular (reseñas >= mediana)"
    )

    col1, col2 = st.columns(2)

    with col1:
        city_selected = st.selectbox(
            "Ciudad", sorted(df_full["city"].dropna().unique())
        )
        stars_input = st.slider(
            "Calificación (estrellas)",
            min_value=float(df_full["stars"].min()),
            max_value=float(df_full["stars"].max()),
            value=4.0,
            step=0.1,
        )
        is_open_choice = st.radio("¿Está abierto?", ("Sí", "No"))

    with col2:
        categories_selected = st.multiselect(
            "Categorías", sorted(category_columns_pred)
        )
        latitude = st.number_input("Latitud", value=float(df_full["latitude"].median()))
        longitude = st.number_input(
            "Longitud", value=float(df_full["longitude"].median())
        )

    input_data = {
        "stars": [stars_input],
        "num_categories": [len(categories_selected)],
        "latitude": [latitude],
        "longitude": [longitude],
        "is_open_binary": [1 if is_open_choice == "Sí" else 0],
        "city": [city_selected],
    }
    for col in category_columns_pred:
        input_data[col] = [1 if col in categories_selected else 0]

    input_df = pd.DataFrame(input_data)

    if st.button("Predecir", type="primary"):
        prob = model.predict_proba(input_df)[0][1]
        threshold = 0.53
        result = "Popular" if prob >= threshold else "No Popular"

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilidad", f"{prob:.2%}")
        with col2:
            if result == "Popular":
                st.success(f"**{result}** ✅")
            else:
                st.warning(f"**{result}** ⚠️")


def main():
    st.title("Dashboard EDA - Restaurantes California")

    data_load_state = st.text("Cargando datos...")
    df_full = load_data()
    data_load_state.text("Datos cargados")

    st.sidebar.header("Filtros")
    city_options = ["Todo el estado"] + sorted(
        df_full["city"].dropna().unique().tolist()
    )
    selected_city = st.sidebar.selectbox("Ciudad", city_options, index=0)

    if selected_city == "Todo el estado":
        df = df_full.copy()
    else:
        df = df_full[df_full["city"] == selected_city].copy()

    processed = prepare_data(df)
    df = processed["df"]
    proportion_high_rating_by_city = processed["proportion_high_rating_by_city"]
    model, category_columns_pred = train_popularity_model(df_full.copy())

    tabs = st.tabs(
        [
            "Estadísticas",
            "Correlaciones",
            "Calificaciones",
            "Reseñas",
            "Reseñas vs Estrellas",
            "Alta Calificación",
            "Por Ciudad",
            "Predicción ML",
        ]
    )

    with tabs[0]:
        show_eda_statistics(df)
    with tabs[1]:
        show_correlation_matrix(df)
    with tabs[2]:
        show_distribution_stars(df, selected_city)
    with tabs[3]:
        show_distribution_reviews(df, selected_city)
    with tabs[4]:
        show_reviews_vs_stars(df, selected_city)
    with tabs[5]:
        show_high_rating_proportion(df, proportion_high_rating_by_city, selected_city)
    with tabs[6]:
        show_city_status(df, selected_city)
    with tabs[7]:
        show_prediction_interface(model, category_columns_pred, df_full)


if __name__ == "__main__":
    main()
