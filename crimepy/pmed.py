'''
Pmedian for districting
with workload equality constraints
'''


import pulp
import networkx as nx
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import osmnx as ox
import numpy as np
from pyproj import Transformer
import warnings
import libpysal as lp
from sklearn.neighbors import KDTree
from sklearn import linear_model
import numpy as np
warnings.filterwarnings('ignore')

av_solv = pulp.listSolvers(onlyAvailable=True)

# Calculating network distances, help via Claude
# need to create checkpoints and save to disk
def calculate_network_distances(df,distance_type,poly_df,buffer_distance):
    """
    Calculate network distances and drive times between coordinate pairs using OSMnx.
    
    Parameters:
    df: DataFrame with columns ID1, ID2, Distance, X1, Y1, X2, Y2
    distance_type: string, either 'travel_distance' or 'travel_time'
    poly_df: geopandas dataframe that gives the area to download street areas
    buffer_distance: float, distance to buffer the poly_df boundary to get dangles
    
    Returns:
    DataFrame with added columns for network distance and drive time
    """
    # Getting crs from the polygon
    coordinate_system = poly_df.crs.to_string()    
    
    # Make a copy of the dataframe
    result_df = df.copy()
    
    # If coordinates are not in lat/lon, convert them
    if coordinate_system != 'epsg:4326':
        transformer = Transformer.from_crs(coordinate_system, 'epsg:4326', always_xy=True)
        
        # Transform coordinates
        lon1_list, lat1_list = transformer.transform(df['X1'].values, df['Y1'].values)
        lon2_list, lat2_list = transformer.transform(df['X2'].values, df['Y2'].values)
        
        # Add transformed coordinates to dataframe
        result_df['lon1'] = lon1_list
        result_df['lat1'] = lat1_list
        result_df['lon2'] = lon2_list
        result_df['lat2'] = lat2_list
    else:
        # Assume X1,Y1,X2,Y2 are already lon,lat
        result_df['lon1'] = df['X1']
        result_df['lat1'] = df['Y1'] 
        result_df['lon2'] = df['X2']
        result_df['lat2'] = df['Y2']
    
    # Download road network from polygon
    poly_df2 = poly_df.copy()
    poly_df2['Const'] = 1
    poly_df2 = poly_df2[['Const','geometry']].dissolve(by='Const',aggfunc='sum')
    poly_df2['geometry'] = poly_df2['geometry'].buffer(buffer_distance)
    if coordinate_system != 'EPSG:4326':
        poly_df2 = poly_df2.to_crs('EPSG:4326')
    polygon = poly_df2.geometry.iloc[0]
    print("Downloading road network from OpenStreetMap...")
    try:
        # Download driving network with a buffer around the bounding box
        G = ox.graph_from_polygon(
                polygon,
                network_type='drive',
                simplify=True,
                retain_all=False
            )
        
        # Add speed and travel time attributes
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        
        print(f"Network downloaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
    except Exception as e:
        print(f"Error downloading network: {e}")
        return result_df
    
    # Initialize result columns
    result_df['network_distance'] = np.nan
    result_df['route_found'] = False
    
    # Calculate network distance and drive time for each pair
    tot_size = result_df.shape[0]
    if tot_size > 5000:
        check_size = 1000
    elif tot_size > 1000:
        check_size = 100
    else:
        check_size = 10
    print(f"Calculating {tot_size} routes")
    tot_bad = 0
    tot_good = 0
    tot_n = 0
    for idx, row in result_df.iterrows():
        tot_n += 1
        if tot_n == 100:
            if tot_bad == 100:
                print('First 100 attempts are bad, exiting out of solution')
                return None
        try:
            # Find nearest network nodes to origin and destination
            orig_node = ox.nearest_nodes(G, row['lon1'], row['lat1'])
            dest_node = ox.nearest_nodes(G, row['lon2'], row['lat2'])
            
            # Calculate shortest path by distance
            try:
                if distance_type == 'travel_distance':
                    route_distance = nx.shortest_path(G, orig_node, dest_node, weight='length')
                    # Sum up the edge lengths
                    total_distance = sum([G[route_distance[i]][route_distance[i+1]][0]['length'] 
                                         for i in range(len(route_distance)-1)])
                    result_df.at[idx, 'network_distance'] = total_distance
                    result_df.at[idx, 'route_found'] = True
                    tot_good += 1
                elif distance_type == 'travel_time':
                    route_time = nx.shortest_path(G, orig_node, dest_node, weight='travel_time')
                    # Sum up the edge travel times
                    total_time = sum([G[route_time[i]][route_time[i+1]][0]['travel_time'] 
                                    for i in range(len(route_time)-1)])
                    result_df.at[idx, 'network_distance'] = total_time / 60  # Convert to minutes
                    result_df.at[idx, 'route_found'] = True
                    tot_good += 1
            except nx.NetworkXNoPath:
                if tot_bad < 5:
                    print(f"No route found for row {idx} (distance)")
                elif tot_bad < 100:
                    if tot_bad % 10 == 0:
                        print(f"Total routes not found {tot_bad}")
                else:
                    if tot_bad % 100 == 0:
                        print(f"Total routes not found {tot_bad}")
                tot_bad += 1
                continue
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
        
        # Progress indicator
        if (idx + 1) % check_size == 0:
            print(f"Processed {idx + 1}/{len(result_df)} routes")
    
    # Summary statistics
    successful_routes = result_df['route_found'].sum()
    print(f"\nSummary:")
    
    if successful_routes > 0:
        print(f"Successfully calculated routes: {successful_routes}/{len(result_df)}")
    
    return result_df

def prep_dicts(gdf,id_field,calls_field):
    """
    Prepare the data (minus the distance matrix) for the p-median model
    
    gdf -- geopandas dataframe, should be projected
    id_field -- unique identifier field
    
    returns list of areas, continuity dictionary (Rook), and calls dictionary
    """
    cr = gdf[[id_field,calls_field,'geometry']].copy()
    cr.set_index(id_field,inplace=True)
    gdf_neighbors = lp.weights.Rook.from_dataframe(cr,use_index=True)
    gdf_adj_list = gdf_neighbors.to_adjlist(drop_islands=True)
    # identify missing locations
    # loop over focal, turn into dictionary
    cont_dict = {}
    for f in cr.index:
        li = gdf_adj_list[gdf_adj_list['focal'] == f]['neighbor'].tolist()
        if len(li) == 0:
            print(f'WARNING: focal area {f} does not have any neighbors')
        cont_dict[f] = li
    # call dict
    call_dict = cr[calls_field].to_dict()
    # areas
    areas = cr.index.tolist()
    return areas, cont_dict, call_dict

# This returns a distance matrix, 
def get_distance(gdf,id_field,limit,buffer_distance,distance_type='travel_time',prior_df=None):
    # The way the KDtree works, need to always redo that even if you have prior_df
    cr = gdf[[id_field,'geometry']].reset_index(drop=True)
    # should do an error if in epsg:4326
    # first getting the limited set if they are within geo-distance
    cent = cr.geometry.centroid
    cent_xy = np.vstack([cent.x,cent.y]).T
    tree = KDTree(cent_xy)
    idx, dis = tree.query_radius(cent_xy,r=limit,return_distance=True)
    res_pairs = []
    for i in range(cent_xy.shape[0]):
        n = cr.iloc[idx[i],0]
        tn = n.shape[0]
        il = [i]*tn
        f = cr.iloc[il,0]
        fx = cent_xy[il,0]
        fy = cent_xy[il,1]
        dx = cent_xy[idx[i],0]
        dy = cent_xy[idx[i],1]
        res_pairs.append(np.vstack([f,n,dis[i],fx,fy,dx,dy]).T)
    # the resulting numpy matrix
    res_pairs_np = np.concatenate(res_pairs)
    res_pairs_np = res_pairs_np[res_pairs_np[:,2] > 0,:]
    col = ['ID1','ID2','distance','X1','Y1','X2','Y2']
    res_pairs_df = pd.DataFrame(res_pairs_np,columns=col)
    # I should do a check to make sure that all locations are represented
    unloc = pd.unique(res_pairs_df['ID1'])
    tot_missing = (~cr[id_field].isin(unloc)).sum()
    if tot_missing > 0:
        print('Warning, the distance threshold is not large enough to connect')
        print('all locations, please choose a larger threshold')
    # filtering out the locations in the prior_df if available
    if prior_df is not None:
        print(f'Total size is {res_pairs_df.shape[0]}, filtering out prior {prior_df.shape[0]}')
        checkprior = res_pairs_df.shape[0] - prior_df.shape[0]
        res_pairs_mer = pd.merge(res_pairs_df,prior_df[['ID1','ID2','imputed_distance']],on=['ID1','ID2'],how='left')
        mis = res_pairs_mer['imputed_distance'].isna()
        res_pairs_df = res_pairs_mer[mis].reset_index(drop=True)
        if mis.sum() != checkprior:
            dif = mis.sum() - checkprior
            print(f'Note not all prior are merged in! Missing {dif} more than expected')
    result = calculate_network_distances(res_pairs_df,distance_type,gdf,buffer_distance)
    # combining back again
    if prior_df is not None:
        result = pd.concat([prior_df,result],ignore_index=True)
    # imputing distances using linear regression for those missing
    mis_data = result['network_distance'].isna()
    if mis_data.sum() > 0:
        print('Imputing missing data using linear regression')
        reg = linear_model.LinearRegression()
        rdnm = result[result['route_found']] # using all real data
        reg.fit(rdnm[['distance']],rdnm['network_distance'])
        pred = pd.Series(reg.predict(result[['distance']]),index=result.index)
        result['imputed_distance'] = result['network_distance'].fillna(pred)
    else:
        result['imputed_distance'] = result['network_distance']
    return result

# pmed cannot calculate the distance matrix
# in the function, it takes too long and is to
# error prone
# have that outside
# result = get_distance(gdf,id_field,th,buffer_distance,distance_type)

class pmed():
    """
    gdf - geopandas dataframe with sub-areas
    call - string call field
    id_field - string field with unique IDs for areas
    ta - integer number of areas to create
    ine - float inequality constraint
    th - float distance threshold to make a decision variables
    dist_matrix - pandas dataframe with the distance matrix info
    """
    def __init__(self,gdf,calls_field,id_field,
                 ta,ine,th,dist_matrix):
        self.gdf = gdf
        self.Ta = ta
        self.In = ine
        self.Th = th
        self.id_field = id_field
        self.calls_field = calls_field
        self.mod_iter = 0
        # Creating the base dictionaries
        areas, cont_dict, call_dict = prep_dicts(gdf,id_field,calls_field)
        self.Ar = areas
        self.Co = cont_dict
        self.Ca = call_dict
        # Creating the network dictionary
        result = dist_matrix[['ID1','ID2','imputed_distance']].copy()
        # adding in self locations
        sa = pd.DataFrame([(a,a,0.0) for a in areas],columns=['ID1','ID2','imputed_distance'])
        result = pd.concat([result,sa],ignore_index=True)
        result['di'] = result[['ID2','imputed_distance']].apply(lambda x: {x.iloc[0]: x.iloc[1]},axis=1)
        def merge_di(x):
            merge_di = {}
            for i in x:
                merge_di.update(i)
            return merge_di
        res = result.groupby('ID1')['di'].agg(merge_di)
        res_di = res.to_dict()
        self.dist_matrix = result
        # Need to add in self locations as 0
        self.Di = res_di # this expects the full distance matrix
        # not running create problem, as you may need to modify some of these data elements
    def modify_cont(self,pairs):
        for a,b in pairs:
            self.Co[a].append(b)
            self.Co[b].append(a)
    def create_problem(self):
        # Assigning initial properties of object
        Ar = self.Ar
        Di = self.Di
        Co = self.Co
        Ca = self.Ca
        Ta = self.Ta
        In = self.In
        Th = self.Th
        DM = self.dist_matrix
        self.subtours = [] #empty subtours to start
        self.objective = -1 #objective values
        self.pairs = None #where to stuff the matched areas
        # Creating inequality metrics
        SumCalls = sum(Ca.values())
        MaxIneq = (SumCalls/Ta)*(1 + In)
        MinIneq = (SumCalls/Ta)*(1 - In)
        self.ineq = [MaxIneq,MinIneq]
        # Creating contiguity graph
        G = nx.Graph()
        for i in Ar:
            for j in Co[i]:
                G.add_edge(i,j)
        self.co_graph = G
        # Creating threshold vectors for decision variables
        near_locs = (DM['imputed_distance'] < Th)
        Thresh = DM[near_locs][['ID1','ID2']].values.tolist()
        NearAreas = DM[near_locs].groupby('ID1')['ID2'].agg(lambda x: x.tolist()).to_dict()
        RevNearAreas = DM[near_locs].groupby('ID2')['ID1'].agg(lambda x: x.tolist()).to_dict()
        self.NearAreas = NearAreas
        self.RevNearAreas = RevNearAreas
        self.Thresh = Thresh
        # Setting up the pulp problem
        P = pulp.LpProblem("P-Median",pulp.LpMinimize)
        # Decision variables
        assign_areas = pulp.LpVariable.dicts("SD",
                       [(s,d) for (s,d) in Thresh], 
                       lowBound=0, upBound=1, cat=pulp.LpInteger)
        # Just setting the y_vars as the diagonal sources/destinations
        y_vars = {s:assign_areas[(s,s)] for s in Ar}
        tot_constraints = 0
        self.assign_areas = assign_areas
        self.y_vars = y_vars
        # Function to minimize
        P += pulp.lpSum(Ca[d]*Di[s][d]*assign_areas[(s,d)] for (s,d) in Thresh)
        # Constraint on max number of areas
        P += pulp.lpSum(y_vars[s] for s in Ar) == Ta
        tot_constraints += 1
        # Constraint no offbeat if local is not assigned (1)
        # Second is contiguity constraint
        for s,d in Thresh:
            P += assign_areas[(s,d)] - y_vars[s] <= 0
            tot_constraints += 1
            if s != d:
                # Identifying locations contiguous in nearest path
                both = set(nx.shortest_path(G,s,d)) & set(Co[d])
                # Or if nearer to the source
                nearer = [a for a in Co[d] if Di[s][a] < Di[s][d]]
                # Combining, should alwayss have at least 1
                comb = list( both | set(nearer) )
                # Contiguity constraint
                P += pulp.lpSum(assign_areas[(s,a)] for a in comb if a in NearAreas[s]) >= assign_areas[(s,d)]
                tot_constraints += 1
        # Constraint every destination covered once
        # Then Min/Max inequality constraints
        for (sl,dl) in zip(Ar,Ar):
            P += pulp.lpSum(assign_areas[(s,dl)] for s in RevNearAreas[dl]) == 1
            P += pulp.lpSum(assign_areas[(sl,d)]*Ca[d] for d in NearAreas[sl]) <= MaxIneq
            P += pulp.lpSum(assign_areas[(sl,d)]*Ca[d] for d in NearAreas[sl]) >= MinIneq*y_vars[sl]
            tot_constraints += 3
        self.model = P
        print(f'Total number of decision variables {len(Thresh)}')
        print(f'Total number of constraints {tot_constraints}')
        av_solv = pulp.listSolvers(onlyAvailable=True)
        print(f'Available solvers from pulp, {av_solv}')
    def write_lp(self,filename,**kwargs):
        self.model.writeLP(filname,**kwargs)
    def solve(self,solver=None):
        """
        For solver can either pass in None for default pulp, or various pulp solvers, e.g.
        solver = pulp.CPLEX()
        pulp.CPLEX_CMD(msg=True, warmStart=True)
        solver = pulp.PULP_CBC_CMD(timeLimit=1000)
        solver = pulp.GLPK_CMD()
        etc.
        run print( pulp.listSolvers(onlyAvailable=True) )
        to see available solvers on your machine
        """
        print(f'Starting to solve function at {datetime.now()}')
        self.mod_iter += 1
        if solver == None:
            self.model.solve()
        else:
            self.model.solve(solver)
        print(f'Solve finished at {datetime.now()}')
        stat = pulp.LpStatus[self.model.status]
        if stat != "Optimal":
            print(f"Status is {stat}")
            try:
                self.objective = pulp.value(self.model.objective)
                print(f'Objective value is {self.objective}, but beware not optimal')
            except:
                print('Unable to grab objective value')
        else:
            self.objective = pulp.value(self.model.objective)
            print(f"Status is optimal\ntotal weighted travel is {self.objective}")
        results = []
        try:
            for (s,d) in self.Thresh:
                # Making approximate against potential floating point low values
                if self.assign_areas[(s,d)].varValue >= 0.98:
                    results.append((s,d,self.Di[s][d],self.Ca[d],self.Ca[d]*self.Di[s][d]))
            results_df = pd.DataFrame(results,columns=['Source','Dest','Dist','Calls','DWeightCalls'])
            self.pairs = results_df
            self.agg_stats = results_df.groupby('Source',as_index=False)['Calls'].sum()
            # Calculating number of unique areas as a check
            source_areas = pd.unique(results_df['Source'])
            tot_source = len(source_areas)
            if tot_source == self.Ta:
                print(f'Total source areas is {tot_source}, as you specified')
            else:
                print(f'Potential Error, total source areas is {tot_source}, specified {self.Ta} areas')
        except:
            print('Unable to append results')
    def map_plot(self,savefile=None,show=False,ax=None):
        geo_map = self.gdf
        id_str = self.id_field
        # Merging in data into geoobject
        geo_mer = geo_map[[id_str,'geometry']].merge(self.pairs, left_on=id_str, right_on='Dest',indicator='check_merge')
        total_merge = (geo_mer['check_merge'] == 'both').sum()
        if total_merge != geo_map.shape[0]:
            print('Check the pairs/merge, not all are merged into basemap')
            print( geo_mer['check_merge'].value_counts() )
        # making centroid object for source and dissolve object
        source_locs = geo_mer[geo_mer['Source'] == geo_mer['Dest']].copy()
        diss_areas = geo_mer[['Source','geometry','Calls','DWeightCalls']].dissolve(by='Source',aggfunc='sum')
        # Now making the plot
        ax = geo_mer.plot(column='Source', cmap='Spectral', categorical=True,ax=ax)
        source_locs.geometry.centroid.plot(ax=ax,color='k',edgecolor='white')
        diss_areas.boundary.plot(ax=ax,facecolor=None,edgecolor='k')
        # no x/y ticks
        xticks = ax.get_xaxis().set_ticks([])
        yticks = ax.get_yaxis().set_ticks([])
        if savefile:
            plt.savefig(savefile, dpi=500, bbox_inches='tight')
        elif show is False:
            return ax
        else:
            plt.show()
    def collect_subtours(self):
        subtours = [] 
        areas = pd.unique(self.pairs['Source']).tolist()
        for a in areas:
            a0 = self.pairs['Source'] == a
            a1_dest = self.pairs.loc[a0,'Dest'].tolist()
            subg = self.co_graph.subgraph(a1_dest).copy()
            # Connected components
            cc = [list(c) for c in nx.connected_components(subg)]
            # Any component that does not the source in it is a subtour
            if len(cc) == 1:
                print(f'Source {a} has no subtour')
            else:
                print(f'Source {a} has {len(cc)-1} subtours')
                for c in cc:
                    if a in c:
                        pass
                    else:
                        subtours.append((a,c))
        if len(subtours) >= 1:
            res_subtours = {}
            tot_sub = 0
            # Stats for how many calls/crimes are in those subtours
            for i,s in enumerate(subtours):
                tot_calls = 0
                for a in s[1]:
                    tot_calls += self.Ca[a]
                print(f'{i}: Subtour {s} has total {tot_calls} calls')
                tot_sub += tot_calls
            # Adding subtour constraints back into main problem
            for src,des in subtours:
                sub_check = len(des) - 1
                self.model += pulp.lpSum(self.assign_areas[(src,d)] for d in des) <= sub_check
            # if all locations have 0 calls, just reassigning them to contiguous areas
            # as they do not impact the objective function
            if tot_sub == 0:
                print('All subtours have 0 calls, can assign these locations to wherever convenient')
                self.last_subtour = subtours
                return 0
            # Adding subtours into model object
            self.last_subtour = subtours
            self.subtours += subtours
            # Message to warm start
            print('When resolving model, may wish to use warmStart=True if available for solver')
            return -1
        else:
            print('No subtours found, your solution appears OK')
            return 1
    def fix_zero_subtours(self):
            # todo, a function to clean up these 0 area subtours
            # and assign them to a contiguous area
            pass