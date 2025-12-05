import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from data_preparation import ALLOWED_CATEGORIES


def get_category_columns(df: pd.DataFrame, exclude_cols: list) -> list:
    category_columns = [
        col
        for col in df.columns
        if df[col].dropna().isin([0, 1]).all()
        and col not in exclude_cols
        and col in ALLOWED_CATEGORIES
    ]
    return category_columns


def extract_attribute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract useful features from attributes column"""
    
    def safe_eval(x):
        if pd.isna(x):
            return {}
        try:
            return ast.literal_eval(x)
        except:
            return {}
    
    df['attrs'] = df['attributes'].apply(safe_eval)
    
    # Price range (1-4, higher = more expensive)
    def get_price(x):
        val = x.get('RestaurantsPriceRange2')
        if val and val != 'None':
            try:
                return int(val)
            except:
                return 2
        return 2
    
    df['price_range'] = df['attrs'].apply(get_price)
    
    # Delivery & Takeout
    df['has_delivery'] = df['attrs'].apply(lambda x: 1 if x.get('RestaurantsDelivery') == 'True' else 0)
    df['has_takeout'] = df['attrs'].apply(lambda x: 1 if x.get('RestaurantsTakeOut') == 'True' else 0)
    
    # Reservations & Groups
    df['takes_reservations'] = df['attrs'].apply(lambda x: 1 if x.get('RestaurantsReservations') == 'True' else 0)
    df['good_for_groups'] = df['attrs'].apply(lambda x: 1 if x.get('RestaurantsGoodForGroups') == 'True' else 0)
    
    # Amenities
    df['outdoor_seating'] = df['attrs'].apply(lambda x: 1 if x.get('OutdoorSeating') == 'True' else 0)
    df['has_wifi'] = df['attrs'].apply(lambda x: 1 if x.get('WiFi') in ["'free'", "'paid'"] else 0)
    df['has_tv'] = df['attrs'].apply(lambda x: 1 if x.get('HasTV') == 'True' else 0)
    df['good_for_kids'] = df['attrs'].apply(lambda x: 1 if x.get('GoodForKids') == 'True' else 0)
    
    # Alcohol
    df['serves_alcohol'] = df['attrs'].apply(lambda x: 1 if x.get('Alcohol') and x.get('Alcohol') != 'None' else 0)
    
    df.drop('attrs', axis=1, inplace=True)
    return df


def evaluate_model():
    print("=" * 60)
    print("EVALUACIÓN DEL MODELO DE PREDICCIÓN DE POPULARIDAD")
    print("=" * 60)

    print("\n1. Cargando datos...")
    df = pd.read_csv("filtered_yelp_CA.csv")
    print(f"   Total de registros: {len(df)}")

    print("\n2. Preparando features...")
    
    # Extract attributes
    df = extract_attribute_features(df)
    
    median_reviews = df["review_count"].median()
    df["popular"] = (df["review_count"] >= median_reviews).astype(int)

    exclude_cols = [
        "is_open",
        "Restaurants",
        "high_rating",
        "is_open_binary",
        "popular",
        "is_open_target",
    ]
    category_columns = get_category_columns(df, exclude_cols)

    if category_columns:
        df["num_categories"] = df[category_columns].sum(axis=1)
    else:
        df["num_categories"] = 0

    # Feature engineering
    df["lat_long_interaction"] = df["latitude"] * df["longitude"]
    df["is_multi_category"] = (df["num_categories"] > 1).astype(int)
    
    # Categorías premium (típicamente más populares)
    premium_categories = ["Sushi Bars", "Steakhouses", "Wine Bars", "Seafood"]
    df["has_premium_category"] = df[[c for c in premium_categories if c in category_columns]].sum(axis=1).clip(0, 1)
    
    # Categorías casuales
    casual_categories = ["Fast Food", "Burgers", "Pizza", "Sandwiches"]
    df["has_casual_category"] = df[[c for c in casual_categories if c in category_columns]].sum(axis=1).clip(0, 1)

    feature_cols = [
        "num_categories",
        "latitude",
        "longitude",
        "lat_long_interaction",
        "is_multi_category",
        "has_premium_category",
        "has_casual_category",
        "price_range",
        "has_delivery",
        "has_takeout",
        "takes_reservations",
        "good_for_groups",
        "outdoor_seating",
        "has_wifi",
        "has_tv",
        "good_for_kids",
        "serves_alcohol",
        "city",
    ] + category_columns

    print(f"   Features utilizadas: {len(feature_cols)}")
    print(f"   - Numéricas: num_categories, latitude, longitude")
    print(f"   - Categóricas: city")
    print(f"   - Binarias (categorías): {len(category_columns)}")

    X = df[feature_cols]
    y = df["popular"]

    print(f"\n3. Distribución de clases:")
    print(f"   - Popular (1): {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"   - No Popular (0): {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")

    print("\n4. Split train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   - Train: {len(X_train)} registros")
    print(f"   - Test: {len(X_test)} registros")

    print("\n5. Entrenando modelo...")
    categorical_cols = ["city"]
    numeric_cols = ["num_categories", "latitude", "longitude", "lat_long_interaction", "price_range"]
    binary_cols = [
        "is_multi_category", "has_premium_category", "has_casual_category",
        "has_delivery", "has_takeout", "takes_reservations", "good_for_groups",
        "outdoor_seating", "has_wifi", "has_tv", "good_for_kids", "serves_alcohol"
    ] + category_columns

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("bin", "passthrough", binary_cols),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    pipeline.fit(X_train, y_train)
    print("   ✅ Modelo entrenado")

    print("\n6. Evaluando en conjunto de prueba...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MÉTRICAS DE EVALUACIÓN")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

    print("\n" + "=" * 60)
    print("MATRIZ DE CONFUSIÓN")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicho")
    print(f"                No Pop  Popular")
    print(f"Real No Pop     {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"Real Popular    {cm[1][0]:6d}  {cm[1][1]:6d}")

    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACIÓN")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["No Popular", "Popular"]))

    print("\n" + "=" * 60)
    print("INTERPRETACIÓN")
    print("=" * 60)
    print(f"• El modelo predice correctamente el {accuracy*100:.1f}% de los casos")
    print(
        f"• De los restaurantes predichos como populares, {precision*100:.1f}% realmente lo son"
    )
    print(
        f"• De los restaurantes populares reales, el modelo detecta {recall*100:.1f}%"
    )
    print(
        f"• Balance precision-recall (F1): {f1*100:.1f}% - {'Bueno' if f1 > 0.7 else 'Mejorable'}"
    )


if __name__ == "__main__":
    evaluate_model()
