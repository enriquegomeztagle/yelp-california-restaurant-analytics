import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(
    page_title="Análisis EDA – Restaurantes en California",
    page_icon="📊",
    layout="wide"
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
    df['high_rating'] = (df['stars'] >= 4).astype(int)

    exclude_cols = ['is_open', 'Restaurants', 'high_rating', 'is_open_binary', 'popular', 'is_open_target']
    category_columns = [col for col in df.columns if df[col].dropna().isin([0, 1]).all() and col not in exclude_cols]

    df['num_categories'] = df[category_columns].sum(axis=1)

    category_counts = df[category_columns].sum().sort_values(ascending=False).head(10)

    top_cities = df['city'].value_counts().head(10)

    avg_rating_by_city = (
        df[df['city'].isin(top_cities.index)]
        .groupby('city')['stars']
        .mean()
        .sort_values(ascending=False)
    )

    proportion_high_rating_by_city = (
        df[df['city'].isin(top_cities.index)]
        .groupby('city')['high_rating']
        .mean()
        .sort_values(ascending=False)
    )

    return {
        'df': df,
        'category_counts': category_counts,
        'avg_rating_by_city': avg_rating_by_city,
        'proportion_high_rating_by_city': proportion_high_rating_by_city
    }

def main():
    st.title("Dashboard de Exploración de Datos (EDA)")
    st.markdown("""
    Este panel interactivo muestra varias visualizaciones para analizar los datos de Yelp de restaurantes en California.
    Utilice las pestañas para explorar diferentes aspectos, como la distribución de calificaciones, número de reseñas,
    categorías más comunes y tendencias por ciudad.
    """)

    data_load_state = st.text("Cargando datos...")
    df = load_data()
    processed = prepare_data(df)
    data_load_state.text("Datos cargados correctamente ✅")

    df = processed['df']
    category_counts = processed['category_counts']
    avg_rating_by_city = processed['avg_rating_by_city']
    proportion_high_rating_by_city = processed['proportion_high_rating_by_city']

    model, category_columns_pred = train_popularity_model(df.copy())

    tabs = st.tabs([
        "Distribución de calificaciones",
        "Distribución de reseñas",
        "Relación reseñas vs calificaciones",
        "Top categorías",
        "Promedio de calificaciones por ciudad",
        "Proporción de altas calificaciones por ciudad",
        "Predicción de popularidad"
    ])

    with tabs[0]:
        st.subheader("Distribución de calificaciones")
        stars_count = df['stars'].value_counts().sort_index().reset_index()
        stars_count.columns = ['stars', 'count']
        stars_count['y_pos'] = 1
        fig = px.scatter(
            stars_count,
            x='stars',
            y='y_pos',
            size='count',
            color='count',
            title="Distribución de calificaciones de restaurantes",
            labels={'stars': 'Calificación (Estrellas)'},
            color_continuous_scale='Reds',
            size_max=50
        )
        fig.update_traces(marker=dict(symbol='star', line=dict(width=0)))
        fig.update_yaxes(visible=False, showticklabels=False, range=[0.5, 1.5])
        fig.update_xaxes(range=[df['stars'].min() - 0.3, df['stars'].max() + 0.3])
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Distribución del número de reseñas")
        fig = px.histogram(
            df,
            x='review_count',
            nbins=30,
            title="Distribución del número de reseñas",
            labels={'review_count': 'Número de reseñas'},
            color_discrete_sequence=['#4ECDC4']
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Relación entre número de reseñas y calificación")
        fig = px.scatter(
            df,
            x='review_count',
            y='stars',
            opacity=0.5,
            title="Relación entre número de reseñas y calificación",
            labels={'review_count': 'Número de reseñas', 'stars': 'Calificación (Estrellas)'},
            color_discrete_sequence=['#9B59B6']
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Top 10 categorías de restaurantes")
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            labels={'x': 'Categoría', 'y': 'Número de restaurantes'},
            title="Top 10 categorías por número de restaurantes",
            color_discrete_sequence=['#FFA500']
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Promedio de calificaciones por ciudad (Top 10)")
        fig = px.bar(
            x=avg_rating_by_city.values,
            y=avg_rating_by_city.index,
            orientation='h',
            labels={'x': 'Promedio de calificaciones', 'y': 'Ciudad'},
            title="Promedio de calificaciones en las 10 ciudades con más restaurantes",
            color_discrete_sequence=['#AA96DA']
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.subheader("Proporción de restaurantes con alta calificación por ciudad (Top 10)")
        fig = px.pie(
            names=proportion_high_rating_by_city.index,
            values=proportion_high_rating_by_city.values,
            title="Proporción de restaurantes con calificación ≥ 4 estrellas en las 10 principales ciudades",
            color_discrete_sequence=px.colors.sequential.YlOrRd
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        st.subheader("Predicción de popularidad de un restaurante")
        st.markdown(
            """
            Introduzca las características básicas del restaurante para estimar la probabilidad de que sea popular (es decir,
            que reciba un número de reseñas por encima de la mediana del conjunto de datos).
            """
        )

        city_selected = st.selectbox(
            "Ciudad",
            sorted(df["city"].dropna().unique())
        )

        min_star = float(df["stars"].min())
        max_star = float(df["stars"].max())
        stars_input = st.slider(
            "Calificación del restaurante (estrellas)",
            min_value=min_star,
            max_value=max_star,
            value=4.0,
            step=0.1
        )

        is_open_choice = st.radio(
            "¿Está abierto actualmente?",
            ("Sí", "No")
        )

        categories_selected = st.multiselect(
            "Seleccione categorías (puede elegir múltiples)",
            options=sorted(category_columns_pred),
        )

        latitude = st.number_input(
            "Latitud",
            value=float(df["latitude"].median())
        )
        longitude = st.number_input(
            "Longitud",
            value=float(df["longitude"].median())
        )

        num_categories = len(categories_selected)

        input_data = {
            'stars': [stars_input],
            'num_categories': [num_categories],
            'latitude': [latitude],
            'longitude': [longitude],
            'is_open_binary': [1 if is_open_choice == "Sí" else 0],
            'city': [city_selected],
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
