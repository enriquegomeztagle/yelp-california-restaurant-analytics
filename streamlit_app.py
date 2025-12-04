import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

COLOR_SEQ_BLUES = px.colors.sequential.Blues
COLOR_DIV_RED_BLUE = ["#B2182B", "#F7F7F7", "#2166AC"]
COLOR_CATEGORICAL = px.colors.qualitative.Set2
COLOR_HIGHLIGHT = "#F39C12"
COLOR_ALERT = "#E74C3C"
COLOR_BLUE_ACCENT = "#1B4F72"

st.set_page_config(
    page_title="Análisis EDA – Restaurantes en California",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("filtered_yelp_CA.csv")
    return df


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
    category_columns = [
        col
        for col in df_copy.columns
        if df_copy[col].dropna().isin([0, 1]).all() and col not in exclude_cols
    ]

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
    category_columns = [
        col
        for col in df.columns
        if df[col].dropna().isin([0, 1]).all() and col not in exclude_cols
    ]

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
        "category_counts": category_counts,
        "avg_rating_by_city": avg_rating_by_city,
        "proportion_high_rating_by_city": proportion_high_rating_by_city,
    }


def main():
    st.title("Dashboard de Exploración de Datos (EDA)")
    st.markdown(
        """
    Este panel interactivo muestra varias visualizaciones para analizar los datos de Yelp de restaurantes en California.
    Utilice las pestañas para explorar diferentes aspectos, como la distribución de calificaciones, número de reseñas,
    categorías más comunes y tendencias por ciudad.
    """
    )

    data_load_state = st.text("Cargando datos...")
    df_full = load_data()
    data_load_state.text("Datos cargados correctamente ✅")

    st.sidebar.header("Filtros")
    city_options = ["Todo el estado"] + sorted(
        df_full["city"].dropna().unique().tolist()
    )
    selected_city = st.sidebar.selectbox(
        "Filtrar por ciudad", options=city_options, index=0
    )

    if selected_city == "Todo el estado":
        df = df_full.copy()
    else:
        df = df_full[df_full["city"] == selected_city].copy()

    processed = prepare_data(df)
    df = processed["df"]
    category_counts = processed["category_counts"]
    avg_rating_by_city = processed["avg_rating_by_city"]
    proportion_high_rating_by_city = processed["proportion_high_rating_by_city"]

    model, category_columns_pred = train_popularity_model(df_full.copy())

    tabs = st.tabs(
        [
            "Distribución de calificaciones",
            "Distribución de reseñas",
            "Relación reseñas vs calificaciones",
            "Proporción de altas calificaciones por ciudad",
            "Estado y reseñas por ciudad",
            "Predicción de popularidad",
        ]
    )

    with tabs[0]:
        st.subheader("Distribución de calificaciones")
        if len(df) > 0:
            stars_count = df["stars"].value_counts().sort_index().reset_index()
            stars_count.columns = ["stars", "count"]
            stars_count["y_pos"] = 1
            title_suffix = (
                f" - {selected_city}" if selected_city != "Todo el estado" else ""
            )
            fig = px.scatter(
                stars_count,
                x="stars",
                y="y_pos",
                size="count",
                color="count",
                title=f"Distribución de calificaciones de restaurantes{title_suffix}",
                labels={"stars": "Calificación (Estrellas)"},
                color_continuous_scale=COLOR_SEQ_BLUES,
                size_max=50,
            )
            fig.update_traces(
                marker=dict(symbol="star", line=dict(width=0)),
                hovertemplate="<b>Calificación: %{x} estrellas</b><br>"
                + "Cantidad de restaurantes: %{marker.size}<br>"
                + "<extra></extra>",
                text=None,
                showlegend=False,
            )
            fig.update_yaxes(visible=False, showticklabels=False, range=[0.5, 1.5])
            fig.update_xaxes(
                range=[df["stars"].min() - 0.3, df["stars"].max() + 0.3],
                tickangle=-90,
                tickfont=dict(size=12),
            )
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para la ciudad seleccionada.")

    with tabs[1]:
        st.subheader("Distribución del número de reseñas")
        if len(df) > 0:
            title_suffix = (
                f" - {selected_city}" if selected_city != "Todo el estado" else ""
            )
            counts, bins = np.histogram(df["review_count"], bins=30)
            total = len(df)
            percentages = (counts / total) * 100

            bin_labels = []
            for left, right in zip(bins[:-1], bins[1:]):
                left_int = int(left)
                right_int = int(right)
                bin_labels.append(f"{left_int}–{right_int}")

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
                title=f"Distribución del número de reseñas{title_suffix}",
                labels={
                    "review_range": "Rango de número de reseñas",
                    "count": "Cantidad de restaurantes",
                },
                color_continuous_scale=COLOR_SEQ_BLUES,
            )

            fig.update_traces(
                customdata=hist_df["percentage"],
                hovertemplate="<b>Rango: %{x}</b><br>"
                + "Cantidad de restaurantes: %{y}<br>"
                + "Porcentaje: %{customdata:.1f}%<extra></extra>",
                text=None,
            )

            fig.update_layout(
                yaxis_title="Cantidad de restaurantes",
                hovermode="x unified",
                showlegend=False,
                height=700,
                xaxis=dict(tickangle=-30, tickfont=dict(size=12)),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para la ciudad seleccionada.")

    with tabs[2]:
        st.subheader("Relación entre número de reseñas y calificación")
        if len(df) > 0:
            title_suffix = (
                f" - {selected_city}" if selected_city != "Todo el estado" else ""
            )
            fig = px.scatter(
                df,
                x="review_count",
                y="stars",
                hover_data=["name", "city"],
                opacity=0.5,
                title=f"Relación entre número de reseñas y calificación{title_suffix}",
                labels={
                    "review_count": "Número de reseñas",
                    "stars": "Calificación (Estrellas)",
                },
                color_discrete_sequence=[COLOR_HIGHLIGHT],
            )
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>"
                + "Ciudad: %{customdata[1]}<br>"
                + "Reseñas: %{x}<br>"
                + "Calificación: %{y} estrellas<extra></extra>",
                customdata=df[["name", "city"]].values,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para la ciudad seleccionada.")

    with tabs[3]:
        st.subheader(
            "Proporción de restaurantes con alta calificación por ciudad (Top 10)"
        )
        if selected_city == "Todo el estado":
            if len(proportion_high_rating_by_city) > 0:
                fig = px.pie(
                    names=proportion_high_rating_by_city.index,
                    values=proportion_high_rating_by_city.values,
                    title="Proporción de restaurantes con calificación ≥ 4 estrellas en las 10 principales ciudades",
                    color_discrete_sequence=COLOR_DIV_RED_BLUE,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos disponibles.")
        else:
            if len(df) > 0:
                high_rating_count = df["high_rating"].sum()
                total_count = len(df)
                proportion = high_rating_count / total_count if total_count > 0 else 0
                fig = px.pie(
                    names=["Calificación ≥ 4", "Calificación < 4"],
                    values=[high_rating_count, total_count - high_rating_count],
                    title=f"Proporción de restaurantes con calificación ≥ 4 estrellas en {selected_city}",
                    color_discrete_sequence=COLOR_DIV_RED_BLUE,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info(
                    f"En {selected_city}: {high_rating_count} de {total_count} restaurantes ({proportion*100:.1f}%) tienen calificación ≥ 4 estrellas"
                )
            else:
                st.warning("No hay datos disponibles para la ciudad seleccionada.")

    with tabs[4]:
        st.subheader("Estado y reseñas por ciudad")
        if len(df) > 0:
            if selected_city == "Todo el estado":
                top_cities = df["city"].value_counts().head(15).index.tolist()
                df_cities = df[df["city"].isin(top_cities)].copy()

                city_status_data = []
                for city in top_cities:
                    city_df = df_cities[df_cities["city"] == city]

                    abiertos = city_df[city_df["is_open"] == 1]
                    abiertos_count = len(abiertos)
                    abiertos_avg_reviews = (
                        abiertos["review_count"].mean() if abiertos_count > 0 else 0
                    )

                    cerrados = city_df[city_df["is_open"] == 0]
                    cerrados_count = len(cerrados)
                    cerrados_avg_reviews = (
                        cerrados["review_count"].mean() if cerrados_count > 0 else 0
                    )

                    city_status_data.append(
                        {
                            "Ciudad": city,
                            "Estado": "Abiertos",
                            "Cantidad": abiertos_count,
                            "Promedio reseñas": abiertos_avg_reviews,
                        }
                    )
                    city_status_data.append(
                        {
                            "Ciudad": city,
                            "Estado": "Cerrados",
                            "Cantidad": cerrados_count,
                            "Promedio reseñas": cerrados_avg_reviews,
                        }
                    )

                df_status = pd.DataFrame(city_status_data)

                fig = px.bar(
                    df_status,
                    x="Ciudad",
                    y="Cantidad",
                    color="Estado",
                    title="Distribución de lugares abiertos y cerrados por ciudad (Top 15)",
                    labels={"Cantidad": "Número de lugares", "Ciudad": "Ciudad"},
                    barmode="group",
                    color_discrete_map={
                        "Abiertos": COLOR_BLUE_ACCENT,
                        "Cerrados": COLOR_HIGHLIGHT,
                    },
                    hover_data=["Promedio reseñas"],
                )

                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>"
                    + "Estado: %{fullData.name}<br>"
                    + "Cantidad: %{y}<br>"
                    + "Promedio reseñas: %{customdata[0]:.1f}<extra></extra>",
                    customdata=df_status[["Promedio reseñas"]].values,
                )

                fig.update_layout(
                    xaxis=dict(tickangle=-90, tickfont=dict(size=12)),
                    height=600,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )

                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Detalles por ciudad")
                summary_table = []
                for city in top_cities:
                    city_df = df_cities[df_cities["city"] == city]
                    abiertos = city_df[city_df["is_open"] == 1]
                    cerrados = city_df[city_df["is_open"] == 0]

                    summary_table.append(
                        {
                            "Ciudad": city,
                            "Total": len(city_df),
                            "Abiertos": len(abiertos),
                            "Cerrados": len(cerrados),
                            "% Abiertos": f"{(len(abiertos)/len(city_df)*100):.1f}%",
                            "Promedio reseñas (Abiertos)": (
                                f"{abiertos['review_count'].mean():.1f}"
                                if len(abiertos) > 0
                                else "N/A"
                            ),
                            "Promedio reseñas (Cerrados)": (
                                f"{cerrados['review_count'].mean():.1f}"
                                if len(cerrados) > 0
                                else "N/A"
                            ),
                        }
                    )

                summary_df = pd.DataFrame(summary_table)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                city_df = df.copy()

                abiertos = city_df[city_df["is_open"] == 1]
                abiertos_count = len(abiertos)
                abiertos_avg_reviews = (
                    abiertos["review_count"].mean() if abiertos_count > 0 else 0
                )

                cerrados = city_df[city_df["is_open"] == 0]
                cerrados_count = len(cerrados)
                cerrados_avg_reviews = (
                    cerrados["review_count"].mean() if cerrados_count > 0 else 0
                )

                status_data = pd.DataFrame(
                    {
                        "Estado": ["Abiertos", "Cerrados"],
                        "Cantidad": [abiertos_count, cerrados_count],
                        "Promedio reseñas": [
                            abiertos_avg_reviews,
                            cerrados_avg_reviews,
                        ],
                    }
                )

                fig = px.bar(
                    status_data,
                    x="Estado",
                    y="Cantidad",
                    title=f"Distribución de lugares abiertos y cerrados en {selected_city}",
                    labels={"Cantidad": "Número de lugares"},
                    color="Estado",
                    color_discrete_map={
                        "Abiertos": COLOR_BLUE_ACCENT,
                        "Cerrados": COLOR_HIGHLIGHT,
                    },
                    hover_data=["Promedio reseñas"],
                )

                fig.update_traces(
                    texttemplate="%{y}<br>Promedio reseñas: %{customdata[0]:.1f}",
                    textposition="outside",
                    customdata=status_data[["Promedio reseñas"]].values,
                )

                fig.update_layout(showlegend=False, height=500)

                st.plotly_chart(fig, use_container_width=True)

                total = len(city_df)
                st.info(
                    f"**{selected_city}:**\n"
                    f"- Total de lugares: {total}\n"
                    f"- Abiertos: {abiertos_count} ({abiertos_count/total*100:.1f}%) - Promedio reseñas: {abiertos_avg_reviews:.1f}\n"
                    f"- Cerrados: {cerrados_count} ({cerrados_count/total*100:.1f}%) - Promedio reseñas: {cerrados_avg_reviews:.1f}"
                )
        else:
            st.warning("No hay datos disponibles para la ciudad seleccionada.")

    with tabs[5]:
        st.subheader("Predicción de popularidad de un restaurante")
        st.markdown(
            """
            Introduzca las características básicas del restaurante para estimar la probabilidad de que sea popular (es decir,
            que reciba un número de reseñas por encima de la mediana del conjunto de datos).
            """
        )

        city_selected = st.selectbox(
            "Ciudad", sorted(df_full["city"].dropna().unique())
        )

        min_star = float(df_full["stars"].min())
        max_star = float(df_full["stars"].max())
        stars_input = st.slider(
            "Calificación del restaurante (estrellas)",
            min_value=min_star,
            max_value=max_star,
            value=4.0,
            step=0.1,
        )

        is_open_choice = st.radio("¿Está abierto actualmente?", ("Sí", "No"))

        categories_selected = st.multiselect(
            "Seleccione categorías (puede elegir múltiples)",
            options=sorted(category_columns_pred),
        )

        latitude = st.number_input("Latitud", value=float(df_full["latitude"].median()))
        longitude = st.number_input(
            "Longitud", value=float(df_full["longitude"].median())
        )

        num_categories = len(categories_selected)

        input_data = {
            "stars": [stars_input],
            "num_categories": [num_categories],
            "latitude": [latitude],
            "longitude": [longitude],
            "is_open_binary": [1 if is_open_choice == "Sí" else 0],
            "city": [city_selected],
        }

        for col in category_columns_pred:
            input_data[col] = [1 if col in categories_selected else 0]

        input_df = pd.DataFrame(input_data)

        if st.button("Predecir popularidad"):
            prob = model.predict_proba(input_df)[0][1]
            threshold = 0.53
            result = "Popular" if prob >= threshold else "No Popular"
            st.write(f"**Probabilidad de popularidad:** {prob:.2f}")
            st.write(f"**Resultado:** {result}")


if __name__ == "__main__":
    main()
