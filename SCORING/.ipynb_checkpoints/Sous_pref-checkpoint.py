import geopandas as gpd
import pandas as pd

# Exemple avec un shapefile ou un GeoPackage des subdivisions
gdf_admin = gpd.read_file("CIV_adm.gpkg")  # fichier COD-AB ou CNTIG, etc. [web:91]

# En fonction de la structure, tu auras des colonnes du type :
# ADM1_EN (district), ADM2_EN (region), ADM3_EN (departement), ADM4_EN (sous_prefecture)

ref_souspref_region = (
    gdf_admin[["ADM4_EN", "ADM2_EN"]]
    .drop_duplicates()
    .rename(columns={"ADM4_EN": "sous_prefecture", "ADM2_EN": "region"})
)

ref_souspref_region.to_csv("ref_souspref_region.csv", index=False)
