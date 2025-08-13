import pandas as pd
import importlib_resources
from .geo import convgpd
import geopandas as gpd
import networkx as nx

dallas_prj = 'EPSG:2276'

dallas_nibr_num = {'LARCENY/ THEFT OFFENSES': 0, 
            'MOTOR VEHICLE THEFT': 1, 
            'DESTRUCTION/ DAMAGE/ VANDALISM OF PROPERTY': 2, 
            'ASSAULT OFFENSES': 3, 
            'DRUG/ NARCOTIC VIOLATIONS': 4, 
            'BURGLARY/ BREAKING & ENTERING': 5, 
            'ALL OTHER OFFENSES': 6, 
            'TRAFFIC VIOLATION - HAZARDOUS': 7, 
            'ROBBERY': 8, 
            'PUBLIC INTOXICATION': 9, 
            'WEAPON LAW VIOLATIONS': 10, 
            'FRAUD OFFENSES': 11, 
            'DRIVING UNDER THE INFLUENCE': 12, 
            'TRESPASS OF REAL PROPERTY': 13, 
            'FAMILY OFFENSES, NONVIOLENT': 14, 
            'STOLEN PROPERTY OFFENSES': 15, 
            'EMBEZZELMENT': 16, 
            'COUNTERFEITING / FORGERY': 17}

rev_dallas_nibrs = {v:k for k,v in dallas_nibr_num.items()}

dallas_loc_label = {0: 'Street',
             1: 'Apartment/Residence',
             2: 'Bar/Restaurant',
             3: 'Commercial',
             4: 'Gas/Convenience',
             5: 'Hotel/Motel',
             6: 'Other',
             7: 'Outdoor',
             8: 'Parking Lot',
             9: 'Store',
            10: 'School'}

def load_dallas_data():
    """
    Loads the dallasdata.csv.zip file included with the package into a pandas DataFrame.
    """
    # Use importlib_resources to access the data file packaged with the library
    data_path = importlib_resources.files('crimepy').joinpath('dallasdata.csv.zip')
    
    # read the zipped csv file
    with data_path.open('rb') as f:
        df = pd.read_csv(f, compression='zip')
    
    df['begin'] = pd.to_datetime(df['begin'])
    df['end'] = pd.to_datetime(df['end'])
    df['NIBR_DESC'] = df['nibrs_cat'].replace(rev_dallas_nibrs)
    df['LOC_DESC'] = df['location'].replace(dallas_loc_label)
    df = convgpd(df,xy=['lon','lat'])
    return df.to_crs(dallas_prj)


def load_dallas_border():
    data_path = importlib_resources.files('crimepy').joinpath('Dallas_MainArea_Proj.zip')
    
    # read the zipped csv file
    with data_path.open('rb') as f:
        dall_outline = gpd.read_file(f)
    return dall_outline


def load_network_data():
    """
    Loads the dallasdata.csv.zip file included with the package into a pandas DataFrame.
    """
    # Use importlib_resources to access the data file packaged with the library
    edges_path = importlib_resources.files('crimepy').joinpath('Edges_Gang1.csv')
    nodes_path = importlib_resources.files('crimepy').joinpath('Nodes_Gang1.csv')
    
    # read the csv files
    with edges_path.open('rb') as f:
        edges = pd.read_csv(f)
    
    with nodes_path.open('rb') as f:
        nodes = pd.read_csv(f)
   
    G = nx.Graph()
    lv = list(nodes.set_index('Id').to_dict(orient='index').items())
    G.add_nodes_from(lv)
    G.add_edges_from(edges.values)
    return G, nodes, edges