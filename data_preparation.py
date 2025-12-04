import pandas as pd
import shutil
import re
from datetime import datetime

ALLOWED_CATEGORIES = [
    "Restaurants",
    "Food",
    "Nightlife",
    "Bars",
    "Sandwiches",
    "Pizza",
    "Coffee & Tea",
    "Breakfast & Brunch",
    "Burgers",
    "Mexican",
    "Italian",
    "Specialty Food",
    "Seafood",
    "Desserts",
    "Chinese",
    "Bakeries",
    "Salad",
    "Chicken Wings",
    "Cafes",
    "Ice Cream & Frozen Yogurt",
    "Beer",
    "Wine & Spirits",
    "Delis",
    "Sports Bars",
    "Japanese",
    "Pubs",
    "Cocktail Bars",
    "Sushi Bars",
    "Barbeque",
    "Juice Bars & Smoothies",
    "Local Flavor",
    "Steakhouses",
    "Diners",
    "Food Trucks",
    "Wine Bars",
    "Vegetarian",
    "Tacos",
    "Thai",
    "Tex-Mex",
    "Beer Bar",
    "Breweries",
    "Vegan",
    "Vietnamese",
    "Indian",
    "Greek",
    "Gluten-Free",
    "Comfort Food",
    "Bagels",
    "Buffets",
    "Cheesesteaks",
    "Caribbean",
    "Middle Eastern",
    "Korean",
    "Bubble Tea",
    "Food Stands",
    "Noodles",
    "Karaoke",
]


def normalize_city_name(city_name):
    if pd.isna(city_name):
        return city_name

    city = str(city_name).strip()

    city = re.sub(r"\s+", " ", city)

    city = re.sub(r"Santa\s+Barbara", "Santa Barbara", city, flags=re.IGNORECASE)
    city = re.sub(r"Santa\s+Barbra", "Santa Barbara", city, flags=re.IGNORECASE)

    city_normalizations = {
        "Santa Barbra": "Santa Barbara",
        "SantaBarbra": "Santa Barbara",
    }

    if city in city_normalizations:
        city = city_normalizations[city]

    city = re.sub(r"\s+", " ", city)
    city = city.strip()

    return city


def clean_city_names(df):
    cities_before = df["city"].value_counts()
    print(f"   Ciudades únicas antes de la limpieza: {len(cities_before)}")

    changes_made = {}

    df_copy = df.copy()
    for idx, city in df["city"].items():
        normalized = normalize_city_name(city)
        if city != normalized:
            if city not in changes_made:
                changes_made[city] = normalized

    df["city"] = df["city"].apply(normalize_city_name)

    cities_after = df["city"].value_counts()
    print(f"   Ciudades únicas después de la limpieza: {len(cities_after)}")

    if changes_made:
        print(f"\n   Cambios detectados en nombres de ciudades ({len(changes_made)}):")
        for city_before, city_after in changes_made.items():
            count_before = cities_before.get(city_before, 0)
            count_after = cities_after.get(city_after, 0)
            print(
                f"     '{city_before}' ({count_before} registros) -> '{city_after}' ({count_after} registros)"
            )

    return df


def filter_by_categories(df):
    existing_categories = [cat for cat in ALLOWED_CATEGORIES if cat in df.columns]

    if existing_categories:
        mask = df[existing_categories].sum(axis=1) > 0
        df_filtered = df[mask].copy()
        return df_filtered, existing_categories
    else:
        print(
            "❌ Error: No se encontraron ninguna de las categorías especificadas en el dataset."
        )
        return None, []


def prepare_data():
    print("=" * 60)
    print("SCRIPT DE PREPARACIÓN DE DATOS")
    print("=" * 60)

    print("\n1. Cargando datos...")
    df = pd.read_csv("filtered_yelp_CA.csv")

    shape_original = df.shape
    registros_antes = len(df)

    print(f"Shape original: {shape_original}")
    print(f"Registros antes del procesamiento: {registros_antes}")

    print("\n2. Limpiando nombres de ciudades...")
    df = clean_city_names(df)

    print("\n3. Filtrando por categorías permitidas...")
    df_filtered, existing_categories = filter_by_categories(df)

    if df_filtered is not None:
        registros_despues = len(df_filtered)

        print(f"\n4. Resumen del procesamiento:")
        print(f"   Shape original: {shape_original}")
        print(f"   Registros antes del filtro: {registros_antes}")
        print(f"   Registros después del filtro: {registros_despues}")
        print(f"   Registros eliminados: {registros_antes - registros_despues}")
        print(f"   Porcentaje conservado: {registros_despues/registros_antes*100:.2f}%")
        print(f"   Categorías encontradas: {len(existing_categories)}")

        print("\n5. Creando backup del archivo original...")
        backup_filename = (
            f"filtered_yelp_CA_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        shutil.copy("filtered_yelp_CA.csv", backup_filename)
        print(f"   Backup guardado como: {backup_filename}")

        print("\n6. Guardando archivo procesado...")
        df_filtered.to_csv("filtered_yelp_CA.csv", index=False)
        print("   ✅ Preparación de datos completada exitosamente!")

        print("\n7. Estadísticas finales:")
        print(f"   Ciudades únicas: {df_filtered['city'].nunique()}")
        print(f"   Top 5 ciudades:")
        top_cities = df_filtered["city"].value_counts().head(5)
        for city, count in top_cities.items():
            print(f"     - {city}: {count} restaurantes")
    else:
        print("\n❌ Error en el procesamiento de datos.")


if __name__ == "__main__":
    prepare_data()
