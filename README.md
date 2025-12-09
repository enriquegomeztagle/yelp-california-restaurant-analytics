# Yelp California Restaurant Analytics

Dashboard interactivo de análisis exploratorio de datos (EDA) y predicción de popularidad para restaurantes en California usando datos de Yelp.

## 🎯 Características

- **Pipeline de Data Engineering** con limpieza y normalización de datos
- **Feature Engineering** avanzado con extracción de atributos de restaurantes
- **Visualizaciones interactivas** de distribución de calificaciones, reseñas y categorías
- **Análisis por ciudad** de promedios y proporciones de alta calificación
- **Modelo de Machine Learning** (Random Forest) para predecir popularidad
- **Dashboard intuitivo** construido con Streamlit
- **Evaluación completa de modelos** con métricas de rendimiento detalladas

## 📊 Visualizaciones

1. **Distribución de calificaciones** - Representación visual con estrellas proporcionales
2. **Distribución de reseñas** - Histograma del número de reseñas
3. **Relación reseñas vs calificaciones** - Scatter plot
4. **Top 10 categorías** - Gráfico de barras
5. **Promedio por ciudad** - Barras horizontales
6. **Proporción de altas calificaciones** - Gráfico circular

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Zoph-IA/yelp-california-restaurant-analytics.git
cd yelp-california-restaurant-analytics

# Instalar dependencias
pip install -r requirements.txt
```

* [ ] 💻 Uso

```bash
# Preparar y limpiar datos (opcional, ya procesado)
python data_preparation.py

# Ejecutar dashboard principal
streamlit run streamlit_app.py

# Evaluar modelos de ML
python model_evaluation.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 🤖 Modelo de Predicción

El modelo utiliza **Random Forest Classifier** para predecir si un restaurante será popular (número de reseñas por encima de la mediana).

**Features utilizadas:**

- **Numéricas:** num_categories, latitude, longitude, lat_long_interaction, price_range
- **Categóricas:** city
- **Binarias:** is_multi_category, has_premium_category, has_casual_category
- **Atributos:** has_delivery, has_takeout, takes_reservations, good_for_groups, outdoor_seating, has_wifi, has_tv, good_for_kids, serves_alcohol
- **Categorías de restaurante:** 50+ categorías específicas (Pizza, Mexican, Italian, Sushi Bars, etc.)

**Hiperparámetros:**

- n_estimators: 500
- max_depth: 20
- min_samples_split: 3
- min_samples_leaf: 1
- max_features: sqrt
- class_weight: balanced
- random_state: 42

## 📦 Dependencias

- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- statsmodels

## 📁 Estructura del Proyecto

```
yelp-california-restaurant-analytics/
├── streamlit_app.py          # Aplicación principal del dashboard
├── data_preparation.py       # Procesamiento y limpieza de datos
├── model_evaluation.py       # Evaluación de modelos ML
├── filtered_yelp_CA.csv      # Dataset de restaurantes
├── requirements.txt          # Dependencias
├── .gitignore                # Archivos ignorados
└── README.md                 # Documentación
```

## 📈 Dataset

El dataset final contiene **1,629 restaurantes** en **10 ciudades** de California extraídos de Yelp, con **350 columnas** que incluyen:

- **Información básica:** nombre, dirección, ciudad, estado
- **Ubicación:** latitud, longitud
- **Calificaciones:** estrellas (1.5-5.0)
- **Popularidad:** número de reseñas (5-3,834)
- **Estado operacional:** abierto/cerrado
- **Atributos:** precio, delivery, takeout, reservaciones, WiFi, etc.
- **Horarios:** horarios de operación
- **Categorías:** 50+ categorías de restaurantes y servicios

## 🛠️ Tecnologías

- **Python 3.x**
- **Streamlit** - Framework para la aplicación web
- **Plotly** - Visualizaciones interactivas
- **Scikit-learn** - Modelo de Machine Learning
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Statsmodels** - Análisis estadístico

## 👤 Autor

Zoph-IA's team

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
