import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

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
            "Top categorías",
            "Promedio de calificaciones por ciudad",
            "Proporción de altas calificaciones por ciudad",
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
                color_continuous_scale="Reds",
                size_max=50,
            )
            fig.update_traces(marker=dict(symbol="star", line=dict(width=0)))
            fig.update_yaxes(visible=False, showticklabels=False, range=[0.5, 1.5])
            fig.update_xaxes(range=[df["stars"].min() - 0.3, df["stars"].max() + 0.3])
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

            bin_centers = (bins[:-1] + bins[1:]) / 2
            hist_df = pd.DataFrame(
                {
                    "review_count": bin_centers,
                    "count": counts,
                    "percentage": percentages,
                }
            )

            fig = px.bar(
                hist_df,
                x="review_count",
                y="count",
                title=f"Distribución del número de reseñas{title_suffix}",
                labels={"review_count": "Número de reseñas", "count": "Frecuencia"},
                color_discrete_sequence=["#4ECDC4"],
            )

            fig.update_traces(
                texttemplate="%{y}<br>(%{customdata:.1f}%)",
                textposition="outside",
                customdata=hist_df["percentage"],
            )

            fig.update_layout(
                yaxis_title="Frecuencia",
                hovermode="x unified",
                showlegend=False,
                height=500,
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
                color_discrete_sequence=["#9B59B6"],
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
        st.subheader("Número de restaurantes distribuidos por ciudad")
        if len(df) > 0:
            if selected_city == "Todo el estado":
                restaurants_by_city = (
                    df["city"].value_counts().sort_values(ascending=False)
                )
                total_restaurants = len(df)
                percentages = (restaurants_by_city.values / total_restaurants) * 100

                fig = px.bar(
                    x=restaurants_by_city.index,
                    y=restaurants_by_city.values,
                    labels={"x": "Ciudad", "y": "Número de restaurantes"},
                    title="Distribución de restaurantes por ciudad",
                    color_discrete_sequence=["#FFA500"],
                )
                fig.update_traces(
                    texttemplate="%{y}<br>(%{customdata:.1f}%)",
                    textposition="outside",
                    customdata=percentages,
                )

                fig.update_layout(xaxis_tickangle=-45, showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                total_restaurants = len(df)
                city_name = selected_city

                fig = px.bar(
                    x=[city_name],
                    y=[total_restaurants],
                    labels={"x": "Ciudad", "y": "Número de restaurantes"},
                    title=f"Número de restaurantes en {city_name}",
                    color_discrete_sequence=["#FFA500"],
                )

                fig.update_traces(
                    texttemplate="%{y} restaurantes", textposition="outside"
                )

                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para la ciudad seleccionada.")

    with tabs[4]:
        st.subheader("Promedio de calificaciones por ciudad (Top 10)")
        if selected_city == "Todo el estado":
            if len(avg_rating_by_city) > 0:
                percentages = (avg_rating_by_city.values / 5.0) * 100

                fig = px.bar(
                    x=avg_rating_by_city.values,
                    y=avg_rating_by_city.index,
                    orientation="h",
                    labels={"x": "Promedio de calificaciones", "y": "Ciudad"},
                    title="Promedio de calificaciones en las 10 ciudades con más restaurantes",
                    color_discrete_sequence=["#AA96DA"],
                )

                fig.update_traces(
                    texttemplate="%{x:.2f}<br>(%{customdata:.1f}%)",
                    textposition="outside",
                    customdata=percentages,
                )

                fig.update_layout(showlegend=False, height=650)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos disponibles.")
        else:
            if len(df) > 0:
                avg_rating = df["stars"].mean()
                percentage = (avg_rating / 5.0) * 100

                fig = px.bar(
                    x=[avg_rating],
                    y=[selected_city],
                    orientation="h",
                    labels={"x": "Promedio de calificaciones", "y": "Ciudad"},
                    title=f"Promedio de calificaciones en {selected_city}",
                    color_discrete_sequence=["#AA96DA"],
                )
                fig.update_traces(
                    texttemplate="%{x:.2f}<br>(%{customdata:.1f}%)",
                    textposition="outside",
                    customdata=[percentage],
                )

                fig.update_layout(showlegend=False, height=650)
                st.plotly_chart(fig, use_container_width=True)
                st.info(
                    f"Promedio de calificaciones en {selected_city}: {avg_rating:.2f} estrellas ({percentage:.1f}% del máximo)"
                )
            else:
                st.warning("No hay datos disponibles para la ciudad seleccionada.")

    with tabs[5]:
        st.subheader(
            "Proporción de restaurantes con alta calificación por ciudad (Top 10)"
        )
        if selected_city == "Todo el estado":
            if len(proportion_high_rating_by_city) > 0:
                fig = px.pie(
                    names=proportion_high_rating_by_city.index,
                    values=proportion_high_rating_by_city.values,
                    title="Proporción de restaurantes con calificación ≥ 4 estrellas en las 10 principales ciudades",
                    color_discrete_sequence=px.colors.sequential.YlOrRd,
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
                    color_discrete_sequence=px.colors.sequential.YlOrRd,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info(
                    f"En {selected_city}: {high_rating_count} de {total_count} restaurantes ({proportion*100:.1f}%) tienen calificación ≥ 4 estrellas"
                )
            else:
                st.warning("No hay datos disponibles para la ciudad seleccionada.")

    with tabs[6]:
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
