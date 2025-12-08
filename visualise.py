import geopandas as gpd

gdf = gpd.read_file("data/Kontula_2025-07-24.fgb")
print(gdf.head())
gdf.explore(column="predicted_class")
m = gdf.explore(column="predicted_class")
m.save("map.html")
