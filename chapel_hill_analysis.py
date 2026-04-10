import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
import os
from crimepy.query import query_esri, esri_time, cache_query
from crimepy.geo import convgpd, pip, base_folium, add_points, save_map
from crimepy.chain import NearChains
from crimepy.time import monthly_data, month_chart, weekly_data, week_chart
import crimepy.cdcplot # This applies the theme on import

# Configuration
CRIME_URL = "https://services2.arcgis.com/7KRXAKALbBGlCW77/arcgis/rest/services/Recoded_Incidents_New/FeatureServer/0/query"
BOUNDARY_URL = "https://gis-portal.townofchapelhill.org/server/rest/services/OpenData/UrbanServiceBoundary/MapServer/0/query"
PROJ_LOCAL = 'EPSG:2264'
START_YEAR = 2020
CACHE_FILE = 'chapel_hill_crimes.csv'

def fetch_chapel_hill_crimes(start_year=2020):
    """Downloads crime data year by year and returns a combined DataFrame."""
    all_crimes = []
    end_year = datetime.datetime.now().year
    
    for year in range(start_year, end_year + 1):
        print(f"Downloading crime data for {year}...")
        start_dt = datetime.datetime(year, 1, 1)
        end_dt = datetime.datetime(year + 1, 1, 1)
        start_dt_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_dt_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # We use a broad filter for vehicle/car to reduce volume
        where = f"Date_of_Occurrence >= timestamp '{start_dt_str}' AND Date_of_Occurrence < timestamp '{end_dt_str}' AND (Offense LIKE '%VEHICLE%' OR Offense LIKE '%CAR%')"
        
        try:
            year_crimes = query_esri(CRIME_URL, params={'where': where, 'outFields': '*', 'f': 'geojson'}, verbose=True)
            if not year_crimes.empty:
                all_crimes.append(year_crimes)
        except Exception as e:
            print(f"Error downloading data for {year}: {e}")

    if not all_crimes:
        return pd.DataFrame()
    
    return pd.concat(all_crimes, ignore_index=True)

def get_data(cache_file=CACHE_FILE):
    """Gets data from cache or downloads it."""
    df = cache_query(cache_file, fetch_chapel_hill_crimes, {'start_year': START_YEAR})
    
    if not isinstance(df, gpd.GeoDataFrame):
        # Convert to GeoDataFrame using the Latitude/Longitude fields
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs='EPSG:4326')
    
    return df

def analyze_breakins(df, boundary_gdf):
    # Convert time
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(esri_time(df['Date_of_Occurrence']))
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Filter for car break-ins (removed B&E and BREAK-IN)
    break_in_patterns = ['LARCENY F/VEHICLE', 'LARCENY FROM MOTOR VEHICLE']
    mask = df['Offense'].str.contains('|'.join(break_in_patterns), case=False, na=False)
    df_breakins = df[mask].copy()
    
    # Point in Polygon filter
    gdf_breakins = df_breakins.to_crs(PROJ_LOCAL)
    gdf_breakins = pip(gdf_breakins, boundary_gdf)
    print(f"Break-ins within city boundary: {len(gdf_breakins)}")
    
    return gdf_breakins

def create_plots(gdf_breakins):
    # 1. Monthly Chart
    print("Creating monthly chart...")
    m_data = monthly_data(gdf_breakins, 'datetime')
    month_chart(m_data, title='Monthly Car Break-ins in Chapel Hill', file='monthly_breakins_cdc.png')
    print("Saved monthly_breakins_cdc.png")
    
    # 2. Weekly Chart (Last 2 years)
    print("Creating weekly chart (last 2 years)...")
    two_years_ago = datetime.datetime.now() - datetime.timedelta(days=365*2)
    gdf_recent = gdf_breakins[gdf_breakins['datetime'] >= two_years_ago].copy()
    
    w_data = weekly_data(gdf_recent, 'datetime', smooth=8, z=2)
    week_chart(w_data, title='Weekly Car Break-ins (Past 2 Years) with Poisson Z Bands', file='weekly_breakins_cdc.png')
    print("Saved weekly_breakins_cdc.png")

def create_map(gdf_breakins, boundary_gdf):
    print("Creating Folium map...")
    # Create base map with boundary
    m = base_folium(boundary=boundary_gdf, zoom=12, legend_name="Chapel Hill Boundary")
    
    # Prepare data for folium
    gdf_4326 = gdf_breakins.to_crs('EPSG:4326').copy()
    gdf_4326['lat'] = gdf_4326.geometry.y
    gdf_4326['lon'] = gdf_4326.geometry.x
    
    # Format popup text with nicer HTML
    def format_popup(row):
        html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 200px;">
            <h4 style="margin-bottom: 5px; color: #286090;">{row['Offense']}</h4>
            <hr style="margin: 5px 0;">
            <b>Incident ID:</b> {row['Incident_ID']}<br>
            <b>Date/Time:</b> {row['datetime'].strftime('%Y-%m-%d %I:%M %p')}<br>
            <b>Street:</b> {row['Street']}
        </div>
        """
        return html

    gdf_4326['Popup_Text'] = gdf_4326.apply(format_popup, axis=1)
    
    add_points(m, 
               point_df=gdf_4326, 
               lat='lat', 
               lon='lon', 
               html_field='Popup_Text', 
               name="Car Break-ins", 
               cluster=True,
               show=True,
               popup_width=250,
               popup_height=150)
    
    save_map(m, file='breakins_map.html')
    print("Saved breakins_map.html")

def run_clustering(gdf_breakins):
    gdf_breakins['X'] = gdf_breakins.geometry.x
    gdf_breakins['Y'] = gdf_breakins.geometry.y
    
    print("Running NearChains clustering (1000ft, 7 days)...")
    nc = NearChains(gdf_breakins, x='X', y='Y', d='datetime')
    clusters = nc.get_clusters(time_thresh=7, space_thresh=1000)
    
    if clusters:
        summary = nc.get_component_summary(clusters)
        print("\nCluster Summary (Top 10 by size):")
        print(summary.head(10))
        summary.to_csv('cluster_summary.csv', index=False)
        print("Saved cluster_summary.csv")
    else:
        print("No clusters found.")

def main():
    print(f"Downloading Chapel Hill Boundary...")
    boundary_gdf = query_esri(BOUNDARY_URL, params={'where': '1=1', 'outFields': '*', 'f': 'geojson'}, verbose=True)
    boundary_gdf = boundary_gdf.to_crs(PROJ_LOCAL)

    df = get_data()
    if df.empty:
        print("No data found.")
        return
        
    gdf_breakins = analyze_breakins(df, boundary_gdf)
    
    if not gdf_breakins.empty:
        create_plots(gdf_breakins)
        create_map(gdf_breakins, boundary_gdf)
        run_clustering(gdf_breakins)

if __name__ == "__main__":
    main()
