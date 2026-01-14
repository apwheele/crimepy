'''
Pmedian for districting
with workload equality constraints

or

site selection
'''


import pulp
import networkx as nx
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, to_rgb
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


# This returns the euclidean distance matrix between two matrices
def get_euclid_distance(d1,d2,limit,d1xy=['x','y'],d2xy=['x','y']):
    d1n = d1[d1xy].values
    d2n = d2[d2xy].values
    # makes the larger matrix the tree and the smaller the search
    if d1.shape[0] > d2.shape[0]:
        tree = KDTree(d1n)
        idx, dis = tree.query_radius(d2n,r=limit,return_distance=True)
        tdo, tso = 'd1', 'd2'
    else:
        tree = KDTree(d2n)
        idx, dis = tree.query_radius(d1n,r=limit,return_distance=True)
        tdo, tso = 'd2', 'd1'
    res_pairs = []
    for i in range(idx.shape[0]):
        sl = idx[i]
        il = np.repeat(i,sl.shape[0])
        dl = dis[i]
        res_pairs.append(np.vstack([sl,il,dl]).T)
    res_pairs = pd.DataFrame(np.concatenate(res_pairs),columns=[tdo,tso,'dist'])
    return res_pairs[['d1','d2','dist']]

# This returns a network distance matrix for a file against itself
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


def intersection_length(poly1,poly2,smb=1e-15):
    '''
    Length of the intersection between two shapely polygons
    
    poly1 - shapely polygon
    poly2 - shapely polygon
    smb - float, defaul 1e-15, small distance to buffer
    
    The way this works, I compute a very small buffer for
    whatever polygon is simpler (based on length)
    then take the intersection and divide by 2
    so not exact, but close enough for this work
    '''
    # buffer the less complicated edge of the two
    if poly1.length > poly2.length:
        p2, p1 = poly1, poly2
    else:
        p1, p2 = poly1, poly2
    # This basically returns a very skinny polygon
    pb = p1.buffer(smb,cap_style='flat').intersection(p2)
    return (pb.length-2*smb)/2

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
    def clean_zero(self):
        ls = []
        for s,d in self.last_subtour:
            ls += d
        while ls:
            # step2, calculate shared border
            max_bord = 0
            for l in ls:
                # remove the other disjointed nodes
                G2 = self.co_graph.copy()
                G2.remove_nodes_from(set(ls) - set([l]))
                # get neighbors that are attached
                ne = G2.neighbors(l)
                l1 = self.gdf.loc[self.gdf[self.id_field] == l,'geometry'].iloc[0]
                # get the shared border length
                # could also select based on nearest to source
                for n in ne:
                    # if max, select that one
                    l2 = self.gdf.loc[self.gdf[self.id_field] == n,'geometry'].iloc[0]
                    resl = intersection_length(l1,l2)
                    if resl > max_bord:
                        max_bord = resl
                        fin_l = l
                        share_l = n
            # just a double check to exit if there is a problem
            if max_bord == 0:
                print(f'Remaining areas to merge are {ls}')
                print('There is some issue with no further shared border locations available')
            # step3, for edge longest shared border not in subtour, merge
            new_source = self.pairs.loc[self.pairs['Dest'] == share_l,'Source'].iloc[0]
            self.pairs.loc[self.pairs['Dest'] == fin_l,'Source'] = new_source
            print(f'Reassigning location {fin_l} to new source {new_source}')
            # might also change the problem vars
            # remove from ls
            ls.remove(fin_l)


class SiteSelection:
    def __init__(self, crime_locations, site_locations, num_sites, limit_dist=1e9,
                 problem='minsum'):
        """
        Initialize the site selection p-median problem solver.
        
        Parameters:
        crime_locations: dataframe [(x, y, count), ...] where x,y are coordinates and count is crime count
        site_locations: dataframe [(x, y), ...] where x,y are potential site coordinates
        num_sites: int, number of sites to select
        limit_dist: float, limit the distance matrix to within a specified value
        problem: string, either 'minsum' (default), or 'minmax'. Minsum minimizes the sum of
                 all the distances, minmax minimizes the maximum travel
        """
        self.num_sites = num_sites
        self.limit_dist = limit_dist
        self.problem = problem
        
        # Create crime dataframe
        self.crime_df = crime_locations[['x','y','count']].copy()
        self.crime_df.columns = ['cx', 'cy', 'count']
        self.crime_df['cid'] = range(len(self.crime_df))
        
        # Create site dataframe
        self.site_df = site_locations[['x','y']].copy()
        self.site_df.columns = ['dx', 'dy']
        self.site_df['did'] = range(len(self.site_df))
        
        # Create distance matrix
        self._create_distance_matrix()
        
        # Initialize problem variables
        self.prob = None
        self.x = None  # site location variables
        self.y = None  # assignment variables
        self.z = None  # if minmax, max distance
        self.solved = False
        self.solution = None
        
    def _create_distance_matrix(self):
        """Create the distance matrix between all sites and crime locations."""
        self.dist = get_euclid_distance(self.site_df,self.crime_df,self.limit_dist,
                                        d1xy=['dx','dy'],d2xy=['cx','cy'])
        self.dist.columns = ['did','cid','dist']
        self.dist[['did','cid']] = self.dist[['did','cid']].astype(int)
        self.dist['count'] = self.crime_df.iloc[self.dist['cid'],2].values
        self.dist.set_index(['did', 'cid'], inplace=True)
    
    def _create_problem(self):
        """Create the PuLP optimization problem."""
        # Create the problem
        self.prob = pulp.LpProblem("P-median", pulp.LpMinimize)
        
        # Decision variables
        # x[i] = 1 if site is placed at location i, 0 otherwise
        self.x = pulp.LpVariable.dicts("site_location", 
                                      self.site_df['did'].tolist(), 
                                      cat='Binary')
        
        # y[i,j] = 1 if crime location j is assigned to site at location i, 0 otherwise
        self.y = pulp.LpVariable.dicts("assignment", 
                                      [(i, j) for i,j in self.dist.index], 
                                      cat='Binary')
        
        # If minmax, additional z-variable for the max
        if self.problem == 'minmax':
            max_distance = self.dist['dist'].max()
            self.z = pulp.LpVariable("maxdist",0,max_distance)
            # Objective, minimize max distance -- count does not matter!
            self.prob += pulp.lpSum(self.z)
            # Constraint, z is always greater than y*dist
            for i,j in self.dist.index:
                self.prob += pulp.lpSum(self.y[(i,j)]*self.dist.loc[(i,j),'dist']) <= self.z
        else:
            # Objective function: minimize total weighted distance
            self.prob += pulp.lpSum([self.crime_df.loc[j, 'count'] * self.dist.loc[(i,j), 'dist'] * self.y[(i,j)] 
                               for i,j in self.dist.index])
        
        # Constraint 1: Select exactly num_sites site locations
        self.prob += pulp.lpSum([self.x[i] for i in self.site_df['did']]) == self.num_sites
        
        # Constraint 2: Each crime location must be assigned to exactly one site
        for j in self.crime_df['cid']:
            self.prob += pulp.lpSum([self.y[(i,j)] for i in self.site_df['did'] if (i,j) in self.y]) == 1
        
        # Constraint 3: Can only assign to selected site locations
        for i,j in self.dist.index:
            self.prob += self.y[(i,j)] <= self.x[i]
    
    def solve(self,solver=None):
        """
        Solve the p-median problem and return the solution.
        For solver can either pass in None for default pulp, or various pulp solvers, e.g.
        solver = pulp.CPLEX()
        pulp.CPLEX_CMD(msg=True, warmStart=True)
        solver = pulp.PULP_CBC_CMD(timeLimit=1000)
        solver = pulp.GLPK_CMD()
        etc.
        run print( pulp.listSolvers(onlyAvailable=True) )
        to see available solvers on your machine
        
        Returns:
        dict: Solution dictionary containing status, objective value, selected sites, and assignments
        """
        if self.prob is None:
            self._create_problem()
        
        print(f'Starting to solve function at {datetime.now()}')
        # Solve the problem
        if solver == None:
            self.prob.solve()
        else:
            self.prob.solve(solver)
        self.solved = True
        print(f'Solve finished at {datetime.now()}')
        # Extract solution
        solution = {
            'status': pulp.LpStatus[self.prob.status],
            'optimal_value': pulp.value(self.prob.objective),
            'selected_locations': [],
            'assignments': []
        }
        
        # Get selected site locations
        for i in self.site_df['did']:
            if self.x[i].varValue == 1:
                solution['selected_locations'].append({
                    'site_id': i,
                    'position': (self.site_df.loc[i, 'dx'], self.site_df.loc[i, 'dy'])
                })
        
        # Get crime assignments
        for i,j in self.dist.index:
            if self.y[(i,j)].varValue == 1:
                solution['assignments'].append({
                         'crime_id': j,
                        'site_id': i,
                        'crime_position': (self.crime_df.loc[j, 'cx'], self.crime_df.loc[j, 'cy']),
                        'site_position': (self.site_df.loc[i, 'dx'], self.site_df.loc[i, 'dy']),
                        'distance': self.dist.loc[(i,j), 'dist'],
                        'crime_count': self.crime_df.loc[j, 'count']
                    })
        solution['assignments'] = pd.DataFrame(solution['assignments'])
        
        self.solution = solution
    
    def plot_solution(self, ax=None, figsize=(12, 8),colors=None,size_range=(20,300),lines=True):
        """
        Create a visualization of the solution.
        
        Parameters:
        ax: matplotlib axes object (optional, if None will create new figure)
        figsize: tuple, figure size (default (12, 8)) - only used if ax is None
        colors: list, custom colors for site-crime pairs (optional)
        size_mod: float, plots crimes as varying circle sizes, default 20
        """
        if not self.solved:
            raise ValueError("Problem must be solved before plotting. Call solve() first.")
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get selected sites
        rays = self.solution['assignments'].copy()
        site_df = rays[['site_id','site_position']].drop_duplicates()
        selected_sites = site_df['site_id'].tolist()
        
        # Define colors for each selected pair
        if colors is None:
            # Use a colormap to generate distinct colors
            color_map = plt.cm.Set1  # or plt.cm.tab10 for more colors
            color_vals = [color_map(i) for i in range(len(selected_sites))]
        else:
            color_vals = [to_rgb(c) for c in colors]
        
        color_di = dict(zip(selected_sites,color_vals))
        rays['colors'] = rays['site_id'].map(color_di)
        rays[['x','y']] = pd.DataFrame(rays['crime_position'].to_list(),index=rays.index)
        site_df['colors'] = site_df['site_id'].map(color_di)
        site_df[['x','y']] = pd.DataFrame(site_df['site_position'].to_list(),index=site_df.index)
        
        # Plot lines first
        if lines:
            seg = rays[['crime_position','site_position']].values.tolist()
            lc = LineCollection(seg,colors=rays['colors'])
            ax.add_collection(lc)
        
        # Then plot crimes
        cmin, cmax = rays['crime_count'].min(), rays['crime_count'].max()
        cscale = (rays['crime_count'] - cmin)/(cmax-cmin)
        vmin, vmax = size_range
        vscale = cscale*(vmax - vmin) + vmin
        ax.scatter(rays['x'],rays['y'],s=vscale,
                   c=rays['colors'],edgecolors='k',alpha=0.7)
        
        # Then plot the selected site locations
        ax.scatter(site_df['x'],site_df['y'], 
                   s=0.75*(vmax-vmin) + vmin, c=site_df['colors'],marker='s', edgecolors='k', linewidth=2)
        
        # Create legend, just use grey (does not use for now)
        #legend_elements = []
        #legend_elements.append(
        #        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
        #               markersize=10, markeredgecolor='black', markeredgewidth=1,
        #               label=f'Selected Site')
        #    )
        #legend_elements.append(
        #        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
        #               markersize=8, markeredgecolor='black', markeredgewidth=1,
        #               alpha=0.7, label=f'Crimes')
        #    )
        
        # Only call tight_layout and show if we created the figure
        if ax.figure == plt.gcf():
            plt.tight_layout()
            #ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
        
        return ax