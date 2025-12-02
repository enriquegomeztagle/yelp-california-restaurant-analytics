import streamlit as st
import pandas as pd
import plotly.express as px

# Importar módulos de scikit-learn para entrenar el modelo dentro de la aplicación
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# Streamlit configuration
st.set_page_config(
    page_title="Análisis EDA – Restaurantes en California",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data() -> pd.DataFrame:
    """Carga el dataset de Yelp filtrado para California."""
    # Ruta relativa al archivo CSV en el mismo directorio que este script
    df = pd.read_csv("filtered_yelp_CA.csv")
    return df

@st.cache_resource
def train_popularity_model(dataframe: pd.DataFrame):
    """
    Entrena un modelo de GradientBoostingClassifier para predecir la popularidad
    basándose en los datos de Yelp. Devuelve el modelo entrenado y la lista de
    columnas de categorías (variables dummy) utilizadas como características.

    La popularidad se define como que el número de reseñas esté por encima de
    la mediana. Se excluye `review_count` como feature para evitar fuga de
    información.
    """
    df_copy = dataframe.copy()

    # Variable objetivo: popular si review_count >= mediana
    median_reviews = df_copy["review_count"].median()
    df_copy["popular"] = (df_copy["review_count"] >= median_reviews).astype(int)

    # Variable binaria is_open
    if "is_open" in df_copy.columns:
        df_copy["is_open_binary"] = df_copy["is_open"].astype(int)
    else:
        df_copy["is_open_binary"] = 1

    # Identificar columnas de categorías (dummies 0/1)
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

    # Calcular número de categorías por negocio
    if category_columns:
        df_copy["num_categories"] = df_copy[category_columns].sum(axis=1)
    else:
        df_copy["num_categories"] = 0

    # Definir columnas de características
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

    # Clasificar columnas según tipo
    categorical_cols = ["city"]
    numeric_cols = ["stars", "num_categories", "latitude", "longitude"]
    binary_cols = ["is_open_binary"] + category_columns

    # Preprocesamiento: escalar numéricas, codificar categóricas y pasar binarios
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("bin", "passthrough", binary_cols),
        ]
    )

    # Modelo de Gradient Boosting con hiperparámetros fijos
    classifier = GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        random_state=42,
    )

    pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])

    # Entrenar modelo en todos los datos
    pipeline.fit(X, y)

    return pipeline, category_columns

def prepare_data(df: pd.DataFrame) -> dict:
    """
    Prepara diversas métricas y subconjuntos del dataframe para las visualizaciones.

    Devuelve un diccionario con:
    - df: el dataframe original con columna de alta valoración
    - category_counts: conteo de las 10 categorías más frecuentes
    - avg_rating_by_city: promedio de estrellas para las 10 ciudades con más negocios
    - proportion_high_rating_by_city: proporción de negocios de alta valoración en las top ciudades
    """
    # Crea columna de alta valoración
    df['high_rating'] = (df['stars'] >= 4).astype(int)

    # Identificar columnas dummy de categorías (0/1) excluyendo columnas conocidas
    exclude_cols = ['is_open', 'Restaurants', 'high_rating', 'is_open_binary', 'popular', 'is_open_target']
    category_columns = [col for col in df.columns if df[col].dropna().isin([0, 1]).all() and col not in exclude_cols]

    # Crear columna con el número de categorías asociadas
    df['num_categories'] = df[category_columns].sum(axis=1)

    # Top 10 categorías por número de restaurantes
    category_counts = df[category_columns].sum().sort_values(ascending=False).head(10)

    # Top 10 ciudades por cantidad de negocios
    top_cities = df['city'].value_counts().head(10)

    # Promedio de estrellas por ciudad (solo top ciudades)
    avg_rating_by_city = (
        df[df['city'].isin(top_cities.index)]
        .groupby('city')['stars']
        .mean()
        .sort_values(ascending=False)
    )

    # Proporción de negocios con alta valoración en las top ciudades
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

    # Cargar y preparar datos
    data_load_state = st.text("Cargando datos...")
    df = load_data()
    processed = prepare_data(df)
    data_load_state.text("Datos cargados correctamente ✅")

    # Recuperar objetos preparados
    df = processed['df']
    category_counts = processed['category_counts']
    avg_rating_by_city = processed['avg_rating_by_city']
    proportion_high_rating_by_city = processed['proportion_high_rating_by_city']

    # Entrenar modelo de popularidad y obtener la lista de columnas de categorías
    # Se utiliza una copia del DataFrame para evitar modificar el original
    model, category_columns_pred = train_popularity_model(df.copy())

    # Crear pestañas para cada gráfica
    tabs = st.tabs([
        "Distribución de calificaciones",            # Tab 0
        "Distribución de reseñas",                  # Tab 1
        "Relación reseñas vs calificaciones",       # Tab 2
        "Top categorías",                            # Tab 3
        "Promedio de calificaciones por ciudad",     # Tab 4
        "Proporción de altas calificaciones por ciudad",  # Tab 5
        "Predicción de popularidad"                   # Tab 6
    ])

    # 1. Distribución de calificaciones - Estrellas con tamaño proporcional
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

    # 2. Distribución de reseñas - Histograma
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

    # 3. Relación reseñas vs calificaciones - Scatter
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

    # 4. Top categorías - Barras verticales
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

    # 5. Promedio de calificaciones por ciudad - Barras horizontales
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

    # 6. Proporción de altas calificaciones por ciudad - Gráfica de pastel/pie
    with tabs[5]:
        st.subheader("Proporción de restaurantes con alta calificación por ciudad (Top 10)")
        fig = px.pie(
            names=proportion_high_rating_by_city.index,
            values=proportion_high_rating_by_city.values,
            title="Proporción de restaurantes con calificación ≥ 4 estrellas en las 10 principales ciudades",
            color_discrete_sequence=px.colors.sequential.YlOrRd
        )
        st.plotly_chart(fig, use_container_width=True)

    # 7. Predicción de popularidad de restaurantes
    with tabs[6]:
        st.subheader("Predicción de popularidad de un restaurante")
        st.markdown(
            """
            Introduzca las características básicas del restaurante para estimar la probabilidad de que sea popular (es decir,
            que reciba un número de reseñas por encima de la mediana del conjunto de datos).
            """
        )
        # Utilizar el modelo entrenado previamente y la lista de columnas de categorías
        # 'model' y 'category_columns_pred' provienen del ámbito superior de la función main

        # Selección de ciudad
        city_selected = st.selectbox(
            "Ciudad",
            sorted(df["city"].dropna().unique())
        )

        # Calificación en estrellas (float slider)
        min_star = float(df["stars"].min())
        max_star = float(df["stars"].max())
        stars_input = st.slider(
            "Calificación del restaurante (estrellas)",
            min_value=min_star,
            max_value=max_star,
            value=4.0,
            step=0.1
        )

        # Estado abierto/cerrado
        is_open_choice = st.radio(
            "¿Está abierto actualmente?",
            ("Sí", "No")
        )

        # Multi-selección de categorías
        categories_selected = st.multiselect(
            "Seleccione categorías (puede elegir múltiples)",
            options=sorted(category_columns_pred),
        )

        # Coordenadas
        latitude = st.number_input(
            "Latitud",
            value=float(df["latitude"].median())
        )
        longitude = st.number_input(
            "Longitud",
            value=float(df["longitude"].median())
        )

        # Número de categorías seleccionadas
        num_categories = len(categories_selected)

        # Preparar datos de entrada como DataFrame
        input_data = {
            'stars': [stars_input],
            'num_categories': [num_categories],
            'latitude': [latitude],
            'longitude': [longitude],
            'is_open_binary': [1 if is_open_choice == "Sí" else 0],
            'city': [city_selected],
        }

        # Añadir columnas de categorías con 1 o 0
        for col in category_columns_pred:
            input_data[col] = [1 if col in categories_selected else 0]

        input_df = pd.DataFrame(input_data)

        # Realizar predicción cuando el usuario pulse el botón
        if st.button("Predecir popularidad"):
            prob = model.predict_proba(input_df)[0][1]
            threshold = 0.53  # umbral optimizado de la versión final
            result = "Popular" if prob >= threshold else "No Popular"
            st.write(f"**Probabilidad de popularidad:** {prob:.2f}")
            st.write(f"**Resultado:** {result}")

if __name__ == "__main__":
    main()