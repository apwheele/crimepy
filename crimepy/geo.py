'''
Geographic data
helpers
'''

from datetime import datetime
import pyproj
import contextily as cx
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib.legend_handler import HandlerBase
from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd
import geopandas as gpd
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import FloatImage, Geocoder
from matplotlib.colors import rgb2hex
import os
from pylab import cm
from folium.plugins import MarkerCluster
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.ops import unary_union
import re

# For the folium maps
# I round the coordinates to 6 digits
# this is sub-meter precision, and 
# can save considerable space in the
# final map with many elements
FOLIUM_ROUND = 6

def nice_outline(geometry,buffer=None,simplify=None,perserve=False):
    g = geometry.copy()
    if buffer:
        g = g.buffer(buffer)
        g = g.buffer(-buffer)
    if simplify:
        g = g.simplify(simplify,perserve_topology=preserve)
    return g

def round_geo(geometry,rnd=FOLIUM_ROUND):
    '''
    This returns geojson
    literally rounded to precision I want
    using shapely set_precision and
    post-hoc regex of the string
    '''
    # mode='pointwise' still does not do it
    rj = geometry.set_precision(grid_size=10**-rnd).to_json()
    rs = r'(\d{2}\.|-\d{2}\.)(\d{6})(\d+)'
    return re.sub(rs,r'\1\2',rj)

# Convert XY or latlon into geopandas
def convgpd(data,xy,proj='EPSG:4326'):
    # default proj in Lat/Lon
    miss_xy = data[xy].isna().sum(axis=1) == 0
    d2 = data[miss_xy].reset_index(drop=True)
    geo = gpd.points_from_xy(d2[xy[0]],d2[xy[1]])
    gdf = gpd.GeoDataFrame(d2,geometry=geo,crs=proj)
    return gdf

def proj_xy(data,x,y,proj,inverse):
    p = pyproj.Proj(proj)
    resx, resy = p(data[x],data[y],inverse=inverse)
    return pd.DataFrame(zip(resx,resy),columns=['Lon','Lat'])

# Point-in-Poly
def pip(points,boundary,keep=[]):
    b2 = boundary.copy()
    b2['BOUNDARY_ID'] = range(b2.shape[0])
    try:
        jp = gpd.sjoin(points,b2[['geometry']+keep],how='inner',predicate='within')
    except:
        jp = gpd.sjoin(points,b2[['geometry']+keep],how='inner',op='within')
    return jp[list(points) + keep]

# this just returns a boolean not in polygons
def pnip(points,polys,not_in=True):
    b2 = polys.copy()
    b2['CONST'] = 1
    b2 = b2.dissolve('CONST')
    inp = points.geometry.within(b2.geometry[1],align=True)
    if not_in:
        return ~inp
    else:
        return inp

# Functions for DBSCAN hotspots
def dissolve_overlap(data, id='lab'):
    # via https://gis.stackexchange.com/a/271737/751
    s = data.geometry
    overlap_matrix = s.apply(lambda x: s.intersects(x)).values.astype(int)
    n, ids = connected_components(overlap_matrix)
    new_data = data.reset_index(drop=True)
    new_data[id] = ids
    new_data = new_data.dissolve(by=id, aggfunc='sum')
    return new_data.reset_index()


def db_hotspots(data,distance,min_samp,sf=[],weight=None):
    # Create data and fit DBSCAN
    d2 = data.reset_index(drop=True)
    if weight is None:
        weight = 'weight'
        d2[weight] = 1
    xy = pd.concat([d2.geometry.x,d2.geometry.y],axis=1)
    db = DBSCAN(eps=distance, min_samples=min_samp)
    db.fit(xy,sample_weight=d2[weight])
    max_labs = max(db.labels_)
    if max_labs == -1:
        print('No Hotspots, returning -1')
        return -1
    # Now looping over the samples, creating buffers
    # and return geopandas buffered DF
    res_buff = []
    sf2 = [weight] + sf
    for i in range(max_labs+1):
        sub_dat1 = d2[db.labels_ == i].copy()
        sd = sub_dat1[sf2].sum().to_dict()
        sub_dat2 = sub_dat1[sub_dat1.index.isin(db.core_sample_indices_)].copy()
        sub_dat2['lab'] = i
        sub_dat2.geometry = sub_dat2.buffer(distance)
        sub_dat2 = sub_dat2.dissolve('lab')
        sub_dat2['lab'] = i
        for k,v in sd.items():
            sub_dat2[k] = v
        sub_dat2 = sub_dat2[['lab'] + list(sd.keys()) + ['geometry']]
        res_buff.append(sub_dat2.copy())
    fin_file = pd.concat(res_buff).reset_index(drop=True)
    dis_file = dissolve_overlap(fin_file)
    # redoing label
    dis_file.sort_values(by=weight,ascending=False,ignore_index=True,inplace=True)
    dis_file['lab'] = range(dis_file.shape[0])
    dis_file['lab'] = dis_file['lab']+1
    return dis_file


# Generating spatial grid over the city
# adapted via https://gis.stackexchange.com/a/316460/751
def grid_over(base, size, percent=None):
    b2 = base.copy()
    b2['XXX_BASECONSTANT_XXX'] = 1
    xmin, ymin, xmax, ymax = base.total_bounds
    xl = np.arange(xmin, xmax, size)
    yl = np.arange(ymin, ymax, size)
    polygons = []
    xc = []
    yc = []
    half = size/2.0
    for x in xl:
        for y in yl:
            polygons.append(Polygon([(x,y), (x+size, y), (x+size, y+size), (x, y+size)]))
            xc.append(x+half)
            yc.append(y+half)
    grid = gpd.GeoDataFrame({'geometry':polygons}).set_crs(base.crs)
    grid['X'] = xc
    grid['Y'] = yc
    grid_fields = list(grid)
    #gj = gpd.sjoin(grid,base,how='left',op='intersects')
    gj = gpd.sjoin(grid,b2,how='left',predicate='intersects')
    gloc = gj[~gj['XXX_BASECONSTANT_XXX'].isna()]
    gloc = gloc[grid_fields].reset_index(drop=True)
    if percent:
        gj2 = gpd.overlay(gloc,b2,how='intersection')
        perc = gj2.geometry.area/gloc.geometry.area
        gloc = gloc[perc > percent].reset_index()
    return gloc

# This modifies poly in place
def count_points(poly,points,var_name):
    #join = gpd.sjoin(points, poly, how="left", op='intersects')
    join = gpd.sjoin(points, poly, how="left",predicate='intersects')
    cnt = join['index_right'].value_counts()
    poly[var_name] = cnt
    poly[var_name].fillna(0,inplace=True)

# hexagon map, https://github.com/mrcagney/geohexgrid
# raster KDE map
# nearby points

# I do not like this at all! Most maps IMO do not need 
# a north arrow unless they are oriented in a way north is not north
#Add north arrow, https://stackoverflow.com/a/58110049/604456
def north_arrow(ax,
                aspecs=[0.85,0.10,0.07],
                width=5,
                headwidth=15,
                fontsize=20):
    x, y, arrow_length = aspecs
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=width, headwidth=headwidth),
                ha='center', va='center', fontsize=fontsize,
                xycoords=ax.transAxes)

# Functions to create a basemap, cx.providers.CartoDB.Voyager
outline_kwargs = {'fill':'k',
                  'linewidth':3,
                  'figsize': (10,10),
                  'label': 'City Boundary',
                  'edgecolor': 'k'}

scalebar_kwargs = {'dx': 1,
                   'si-length': 'km',
                   'location': 'lower right'}


narrow_kwargs = {'fontsize': 20}

# Legend helpers for static map
# Convert shapely geometry to matplotlib path
def shapely_to_path(geom):
    """Convert a Shapely geometry to a matplotlib Path"""
    if geom.geom_type == 'Polygon':
        # Handle simple polygon
        coords = np.array(geom.exterior.coords)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
        return Path(coords, codes)
    elif geom.geom_type == 'MultiPolygon':
        # Handle multiple polygons by creating compound path
        paths = []
        for polygon in geom.geoms:
            coords = np.array(polygon.exterior.coords)
            codes = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
            paths.append(Path(coords, codes))
        return Path.make_compound_path(*paths)
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")


# Just copy/paste from SVG
def create_shapely_union(width,height,xdescent,ydescent):
    scale_x = width/20
    scale_y = height/20
    c1x, c1y, r1 = 6.5 * scale_x, 7 * scale_y, 5 * scale_x
    c2x, c2y, r2 = 14 * scale_x, 7 * scale_y, 5 * scale_x  
    c3x, c3y, r3 = 12 * scale_x, 12 * scale_y, 5 * scale_x
    # Adjust for legend positioning
    c1x += xdescent
    c1y += ydescent
    c2x += xdescent
    c2y += ydescent
    c3x += xdescent
    c3y += ydescent
    # Create shapely circles (using buffer on points)
    circle1 = Point((c1x, c1y)).buffer(r1)
    circle2 = Point((c2x, c2y)).buffer(r2)
    circle3 = Point((c3x, c3y)).buffer(r3)
    # Compute the union
    union_shape = unary_union([circle1, circle2, circle3])
    return union_shape


# Handler for Hotspot
class HotSpotHandler(HandlerBase):
    def __init__(self,fill="#880808",fill_alpha=0.5,edge="#8B0000",edge_alpha=1,
                 edge_width=1):
        super().__init__()
        self.fill = fill
        self.fill_alpha = fill_alpha
        self.edge = edge
        self.edge_alpha = edge_alpha
        self.stroke_color = edge
        self.stroke_width = edge_width
    def create_artists(self, legend, orig_handle, xdescent, ydescent, 
                      width, height, fontsize, trans):
        circle_path = shapely_to_path(create_shapely_union(width,height,xdescent,ydescent))
        # Creating two artists, one for the background and the other for the fill
        interior = patches.PathPatch(circle_path,facecolor=self.fill, 
                                    alpha=self.fill_alpha,
                                    edgecolor='none',
                                    linewidth=0,
                                    transform=trans)
        exterior = patches.PathPatch(circle_path,facecolor='none', 
                                    alpha=self.edge_alpha,
                                    edgecolor=self.edge,
                                    linewidth=self.stroke_width,
                                    transform=trans)
        artists = [interior,exterior]
        return artists

# Handler for geographic area
class GeoAreaHandler(HandlerBase):
    def __init__(self, fill='grey', fill_alpha=1.0, edge='black', edge_alpha=1.0,
                 edge_width=1, scale_factor=2.0, 
                 x_scale_factor=0.5, y_scale_factor=1.0,xshift=1.5,yshift=-3.5):
        super().__init__()
        self.fill = fill
        self.fill_alpha = fill_alpha
        self.edge = edge
        self.edge_alpha = edge_alpha
        self.stroke_width = edge_width
        self.scale_factor = scale_factor
        self.x_scale_factor = x_scale_factor
        self.y_scale_factor = y_scale_factor
        self.xshift = xshift
        self.yshift = yshift
    def create_artists(self, legend, orig_handle, xdescent, ydescent, 
                      width, height, fontsize, trans):
        x_pts = [3,17,17,10,3]
        y_pts = [3,3 ,10,17,17]
        max_y = 20  # or max(y_pts) + min(y_pts) if you want to be more general
        y_pts_flipped = [max_y - i for i in y_pts]
        x_pts = [(i/20)*width*self.x_scale_factor*self.scale_factor + xdescent+self.xshift for i in x_pts]
        y_pts = [(i/20)*height*self.y_scale_factor*self.scale_factor + ydescent+self.yshift for i in y_pts_flipped]
        interior = patches.Polygon(list(zip(x_pts, y_pts)), closed=True,
                                    facecolor=self.fill, 
                                    alpha=self.fill_alpha,
                                    edgecolor='none',
                                    linewidth=0,
                                    transform=trans)
        exterior = patches.Polygon(list(zip(x_pts, y_pts)), closed=True,
                                    facecolor='none', 
                                    alpha=self.edge_alpha,
                                    edgecolor=self.edge,
                                    linewidth=self.stroke_width,
                                    transform=trans)
        artists = [interior,exterior]
        return artists

class HotSpotLegendItem:
    def __init__(self, label="HotSpot"):
        self.label = label

class GeoAreaLegendItem:
    def __init__(self, label="Boundary"):
        self.label = label

# simpler to remember this
colalpha = colorConverter.to_rgba

# Should also do one for lines, circles, or other point glyphs
handle_di = {'GeoArea': (GeoAreaLegendItem,GeoAreaHandler),
             'HotSpot': (HotSpotLegendItem,HotSpotHandler)
             }

def map_legend(types,styles):
    art = []
    han_map = {}
    for t,s in zip(types,styles):
        hi, ho = handle_di[t]
        art.append(hi)
        han_map[hi] = ho(**s)
    return art, han_map


# helper for geopandas plot, I do this to handle alpha
# transparency separately for interior/exterior

def geo_map(area,ax,fill,edge='k',fill_alpha=1,edge_alpha=1,edge_width=1,leg_type='GeoArea',**kwargs):
    area.plot(ax=ax,color=colalpha(fill,fill_alpha),
              edgecolor=colalpha(edge,edge_alpha),
              linewidth=edge_width)
    hi, ho = handle_di[leg_type]
    # doing the new class makes it locally scoped
    # so I am not overwriting others
    class LocalItemClass:
        def __init__(self, label="Boundary"):
            self.label = label
    han = ho(fill=fill,edge=edge,fill_alpha=fill_alpha,edge_alpha=edge_alpha,edge_width=edge_width,**kwargs)
    hi = LocalItemClass()
    return hi, han

# SVG helpers for folium legends

# This is for a polygon area (such as a city boundary, or a choropleth map)
def poly_svg(text="Polygon",fill="grey",fill_opacity=0.5,stroke="black",stroke_width=1,stroke_opacity=1,height=20,width=20):
    x_pts = [3,17,17,10,3]
    y_pts = [3,3 ,10,17,17]
    x_pts = [(i/20)*width for i in x_pts]
    y_pts = [(i/20)*height for i in y_pts]
    svg = '<span>\n'
    svg += f'<svg height="{height}" width="{width}">\n'
    poly_pts = " ".join([f'{x},{y}' for x,y in zip(x_pts,y_pts)])
    poly_pts = poly_pts.replace(".0 "," ").replace(".0,",",")
    svg += f'  <polygon points="{poly_pts}" fill="{fill}" fill-opacity="{fill_opacity}" '
    svg += f'stroke="{stroke}" stroke-width="{stroke_width}" stroke-opacity="{stroke_opacity}" />'
    svg += f'\n</svg>  {text}</span>'
    return svg

mask3 = '''    <clipPath id="shape">
      <use href="#circle1" />
      <use href="#circle2" />
      <use href="#circle3" />
    </clipPath>
    <mask id="maskC1">
      <use href="#canvas" />
      <use href="#circle2" />
      <use href="#circle3" />
    </mask>
    <mask id="maskC2">
      <use href="#canvas" />
      <use href="#circle1" />
      <use href="#circle3" />
    </mask>    
    <mask id="maskC3">
      <use href="#canvas" />
      <use href="#circle1" />
      <use href="#circle2" />
    </mask>
    <mask id="maskC2fill">
      <use href="#canvas" />
      <use href="#circle3" />
    </mask>
  </defs>\n'''


# This is for blobby hotspots, forcing to be square since based on circles
def hot_svg(text="HotSpot",fill="grey",fill_opacity=0.9,stroke="black",stroke_width=1,stroke_opacity=1,side=20):
    c1x, c1y, r1 = (6.5/20)*side, (7/20)*side, (5/20)*side
    c2x, c2y, r2 = (14/20)*side, (7/20)*side, (5/20)*side
    c3x, c3y, r3 = (12/20)*side, (12/20)*side, (5/20)*side
    svg = "<span>\n"
    svg += f'<svg width="{side}" height="{side}" xmlns="http://www.w3.org/2000/svg">\n  <defs>\n'
    svg += '    <rect id="canvas" width="100%" height="100%" fill="white" />\n'
    svg += f'    <circle id="circle1" cx="{c1x}" cy="{c1y}" r="{r1}" />\n'
    svg += f'    <circle id="circle2" cx="{c2x}" cy="{c2y}" r="{r2}" />\n'
    svg += f'    <circle id="circle3" cx="{c3x}" cy="{c3y}" r="{r3}" />\n'
    svg += mask3
    svg += f'  <use href="#circle1" stroke="none" fill="{fill}" fill-opacity="{fill_opacity}" mask="url(#maskC1)" />\n'
    svg += f'  <use href="#circle2" stroke="none" fill="{fill}" fill-opacity="{fill_opacity}" mask="url(#maskC2fill)" />\n'
    svg += f'  <use href="#circle3" stroke="none" fill="{fill}" fill-opacity="{fill_opacity}" />\n'
    svg += f'  <use href="#circle1" stroke="{stroke}" stroke-width="{stroke_width}" fill="none" mask="url(#maskC1)"/>\n'
    svg += f'  <use href="#circle2" stroke="{stroke}" stroke-width="{stroke_width}" fill="none" mask="url(#maskC2)"/>\n'
    svg += f'  <use href="#circle3" stroke="{stroke}" stroke-width="{stroke_width}" fill="none" mask="url(#maskC3)"/>\n'
    svg += f"</svg>  {text}</span>"
    svg = svg.replace('.0',"")
    return svg

# Creating a base folium map
def base_folium(boundary=None,
                zoom=12,
                weight=4,
                color="black",
                opacity=0.3,
                logo=False,
                legend_name="City Boundary",
                geo=False,
                geo_loc='topleft',
                location=None,
                show=True):
    if boundary is not None:
        b2 = boundary.copy()
        b2['area'] = boundary.geometry.area
        b2.sort_values(by='area',inplace=True,ascending=False)
        b2.reset_index(drop=True,inplace=True)
        center = b2.geometry.centroid[[0]].to_crs('EPSG:4326')[0]
        b2 = b2.to_crs('EPSG:4326')
    if location is None:
        mapf = folium.Map(location=[round(center.y,FOLIUM_ROUND),round(center.x,FOLIUM_ROUND)],
                      zoom_start=zoom,
                      control_scale= True,
                      tiles=None)
    else:
        mapf = folium.Map(location=location,
              zoom_start=zoom,
              control_scale= True,
              tiles=None)
    #show=True,overlay=False
    cartodb = folium.TileLayer(tiles='cartodbpositron',name='CartoDB Positron Basemap',control=True)
    cartodb.add_to(mapf)
    osm_base = folium.TileLayer(tiles='OpenStreetMap',name='OSM Basemap',control=True,show=True)
    osm_base.add_to(mapf)
    if boundary is not None:
        # Add in boundary, rounded precision
        bound2 = round_geo(b2.boundary)
        def bound_func(x):
            di = {"color":color,
                  "weight": weight,
                  "opacity": opacity}
            return di
        # This is currently not working for hex color codes
        #bound_name = f'''<span><svg width="12" height="12">
        #             <rect width="12" height="12" fill-opacity="100%" fill="white"
        #             style="stroke-width:4;stroke:{color};opacity:{opacity}" />
        #             </svg> {legend_name}</span>'''
        # May also do fill="none"
        bound_name = poly_svg(legend_name,fill="white",fill_opacity=1,stroke=color,
                              stroke_width=4,stroke_opacity=opacity)
        boundfol = folium.GeoJson(bound2, style_function=bound_func, name=bound_name, overlay=True, control=True, show=show)
        boundfol.add_to(mapf)
    # CrimeDeCoder logo
    if logo:
        fi = FloatImage("https://crimede-coder.com/images/CrimeDeCoder_Logo_Small.PNG", bottom=10, left=0.4)
        fi.add_to(mapf)
    # Layer control needs to be last
    # Geocoder if you want it
    if geo:
        geoc = Geocoder(position='geo_loc',add_marker=True)
        geoc.add_to(mapf)
    #folium.LayerControl(collapsed=False).add_to(mapf)
    return mapf

# Adding hotspots
def add_hotspots(mapf,
                 poly_df,
                 tab_fields,
                 title = None,
                 footer = None,
                 name="Hot Spots",
                 fill="#880808",
                 edge="#8B0000",
                 opacity=0.5,
                 tab_names = ['Crime','Count'],
                 sort_crimes=True,
                 svg_func=hot_svg):
    poly2 = poly_df.to_crs('EPSG:4326')
    poly2['area'] = poly_df.geometry.area
    # I do this so smaller geometries are placed on the top
    poly2 = poly2.sort_values(by='area',ascending=False).reset_index(drop=True)
    #svg_name = f'''<span><svg width="12" height="12">
    #            <rect width="12" height="12" fill-opacity="{opacity}" fill="{fill}"
    #             style="stroke-width:4;stroke:{edge}" />
    #             </svg> {name}</span>
    #'''
    svg_name = svg_func(text=name,fill=fill,fill_opacity=opacity,
                       stroke=edge,stroke_width=1.5,stroke_opacity=1)
    fg = folium.FeatureGroup(name=svg_name,overlay=True,control=True)
    def style_func(x):
        di = {"fillColor": fill,
              "fillOpacity": opacity,
              "color": edge}
        return di
    def high_func(x):
        di = {"fillColor": fill,
              "fillOpacity": 0.05,
              "color": edge,
              "weight": 4}
        return di
    for i in range(poly_df.shape[0]):
        sub_data = poly2.loc[[i]].copy()
        geo_js = round_geo(sub_data.geometry)
        geo_fol = folium.GeoJson(data=geo_js,
                                 style_function=style_func,
                                 highlight_function=high_func,
                                 name=svg_name,
                                 overlay=True,
                                 control=True)
        lab_data = pd.DataFrame(sub_data[tab_fields].T.reset_index())
        lab_data.columns = tab_names
        if sort_crimes:
            lab_data.sort_values(by=tab_names[1],ascending=False,inplace=True)
            lab_data[tab_names[1]] = lab_data[tab_names[1]].map('{:,.0f}'.format)
        html_lab = lab_data.to_html(index=False,header=True)
        if title is not None:
            html_lab = sub_data[title][i] + html_lab
        if footer is not None:
            html_lab += sub_data[footer][i]
        popup = folium.Popup(html_lab)
        popup.add_to(geo_fol)
        geo_fol.add_to(fg)
    fg.add_to(mapf)

# Adding polylines
def line_svg(text="Line",stroke="black",stroke_width=2,stroke_opacity=1,side=26):
    svg = "<span>\n"
    svg += f'<svg width="{side}" height="{side}" xmlns="http://www.w3.org/2000/svg">\n'
    svg += f'<line x1="0" y1="{side/2 - 1}" x2="{side}" y2="{side/2 - 1}"'
    svg += f'stroke={stroke} stroke-width={stroke_width} stroke-opacity={stroke_opacity} />'
    svg += f"</svg>  {text}</span>"
    svg = svg.replace('.0',"")
    return svg

def add_lines(mapf,
              line_df,
              html_field = None,
              tooltip_field = None,
              name="Lines",
              edge="#8B0000",
              opacity=0.5,
              svg_func=line_svg,
              show=True,
               popup_width=100,
               popup_height=100,
               width=2,
               highlight_width=5):
    poly2 = line_df.to_crs('EPSG:4326')
    #poly2['length'] = line_df.geometry.length
    # I do this so smaller geometries are placed on the top
    #poly2 = poly2.sort_values(by='length',ascending=False).reset_index(drop=True)
    #svg_name = f'''<span><svg width="12" height="12">
    #            <rect width="12" height="12" fill-opacity="{opacity}" fill="{fill}"
    #             style="stroke-width:4;stroke:{edge}" />
    #             </svg> {name}</span>
    #'''
    svg_name = svg_func(text=name,stroke=edge,stroke_width=2,stroke_opacity=1)
    fg = folium.FeatureGroup(name=svg_name,overlay=True,control=True,show=show)
    def style_func(x):
        di = {"color": edge,
              "opacity": opacity,
              "weight": width}
        return di
    def high_func(x):
        di = {"color": edge,
              "weight": highlight_width}
        return di
    for i in range(poly2.shape[0]):
        d = poly2.iloc[i].T.to_dict()
        sub_data = poly2.loc[[i]].copy()
        geo_js = round_geo(sub_data.geometry)
        if html_field:
            html = d[html_field]
            iframe = folium.IFrame(html=html,width=popup_width,height=popup_height)
            popup = folium.Popup(iframe,max_width=1000)
        else:
            popup = None
        if tooltip_field:
            thtml = d[tooltip_field]
            tooltip = folium.map.Tooltip(thtml)
        else:
            tooltip = None
        geo_fol = folium.GeoJson(data=geo_js,
                                 style_function=style_func,
                                 highlight_function=high_func,
                                 tooltip = tooltip,
                                 popup = popup,
                                 name=svg_name,
                                 overlay=True,
                                 control=True)
        geo_fol.add_to(fg)
    fg.add_to(mapf)

# SVG via https://www.svgrepo.com/svg/302636/map-marker
# making as tiny as possible
def svg_marker(fill="#FF6E6E",inner="#0C0058"):
    svg = f'''<svg width="26px" height="26px" viewBox="-4 0 36 36" xmlns="http://www.w3.org/2000/svg">
<path d="M14,0 C21.732,0 28,5.641 28,12.6 C28,23.963 14,36 14,36 C14,36 0,24.064 0,12.6 C0,5.641 6.268,0 14,0 Z" id="Shape" fill="{fill}"></path>
<circle id="Oval" fill="{inner}" fill-rule="nonzero" cx="14" cy="14" r="7">
</circle></svg>'''
    return svg.replace("\n","")

def svg_markerC(fill="#FF6E6E",inner="#0C0058",div=True):
    divS = '<div style="margin-left: -8px; margin-top: -19px; width: 26px; height: 26px; outline: none;">'
    svg = f'''<svg width="26px" height="26px" viewBox="-4 0 36 36" xmlns="http://www.w3.org/2000/svg">
<path d="M14,0 C21.732,0 28,5.641 28,12.6 C28,23.963 14,36 14,36 C14,36 0,24.064 0,12.6 C0,5.641 6.268,0 14,0 Z" id="Shape" fill="{fill}"></path>
<circle id="Oval" fill="{inner}" fill-rule="nonzero" cx="14" cy="14" r="7">
</circle></svg></div>'''
    if div:
        svg = divS + svg + "</div>"
    return svg.replace("\n","")

# adding in a point method
def add_points(mapf,
               point_df,
               lat = 'lat',
               lon = 'lon',
               html_field = None,
               tooltip_field = None,
               name="Points",
               fill="#286090",
               inner="#9EACC5",
               svg_func=svg_markerC,
               show=False,
               popup_width=100,
               popup_height=100,
               cluster=False,
               cluster_options=None):
    point2 = point_df.copy()
    # checking for missing data
    mis = point2[[lat,lon]].isna().sum(axis=1) == 0
    point2 = point2[mis].copy()
    # creating svg
    svg = svg_func(fill=fill,inner=inner,div=False)
    svg_name = "<span>\n" + svg + f"</svg>  {name}</span>"
    if cluster:
        fg = MarkerCluster(name=svg_name,overlay=True,control=True,show=show,options=cluster_options)
    else:
        fg = folium.FeatureGroup(name=svg_name,overlay=True,control=True,show=show)
    svg_div = svg_func(fill=fill,inner=inner,div=True)
    # now looping over dataframe and extracting info
    for i in range(point2.shape[0]):
        d = point2.iloc[i].T.to_dict()
        latv = round(d[lat],FOLIUM_ROUND)
        lonv = round(d[lon],FOLIUM_ROUND)
        if html_field:
            html = d[html_field]
            iframe = folium.IFrame(html=html,width=popup_width,height=popup_height)
            popup = folium.Popup(iframe,max_width=1000)
        else:
            popup = None
        if tooltip_field:
            thtml = d[tooltip_field]
            tooltip = folium.map.Tooltip(thtml)
        else:
            tooltip = None
        fm = folium.Marker(location=[latv,lonv],
                           popup=popup,tooltip=tooltip,
                           icon=folium.DivIcon(svg_div))
        fm.add_to(fg)
    fg.add_to(mapf)


def circle_svg(fill,fill_opacity,stroke,stroke_opacity,height=20,width=20):
    cir = f'<svg width="{width}px" height="{height}px" xmlns="http://www.w3.org/2000/svg">'
    cir += f'<circle r="{min(height,width)/2.5}" cx="{width/2}" cy="{height/2}" stroke="{stroke}" '
    cir += f'stroke-opacity="{stroke_opacity}" stroke-width="3" fill="{fill}" fill-opacity="{fill_opacity}" />'
    cir +=  '</svg>'
    return cir


# Adding Circle Points
# adding in a point method
# if using the Circle type, the radius is in meters
def add_circle_points(mapf,
                      point_df,
                      radius,
                      lat = 'lat',
                      lon = 'lon',
                      html_field = None,
                      tooltip_field = None,
                      name="CirclePoints",
                      fill="#286090",
                      color="#9EACC5",
                      fill_opacity=0.5,
                      opacity=1.0,
                      svg_func=svg_markerC,
                      show=False,
                      popup_width=100,
                      popup_height=100,
                      circle_type="CircleMarker"):
    point2 = point_df.copy()
    # checking for missing data
    mis = point2[[lat,lon]].isna().sum(axis=1) == 0
    point2 = point2[mis].copy()
    # creating svg
    if fill is None:
        svg = circle_svg(fill="white",fill_opacity=0,stroke=color,stroke_opacity=opacity)
    else:
        svg = circle_svg(fill=fill,fill_opacity=fill_opacity,stroke=color,stroke_opacity=opacity)
    svg_name = "<span>\n" + svg + f"</svg>  {name}</span>"
    fg = folium.FeatureGroup(name=svg_name,overlay=True,control=True,show=show)
    # now looping over dataframe and extracting info
    for i in range(point2.shape[0]):
        d = point2.iloc[i].T.to_dict()
        latv = round(d[lat],FOLIUM_ROUND)
        lonv = round(d[lon],FOLIUM_ROUND)
        if html_field:
            html = d[html_field]
            iframe = folium.IFrame(html=html,width=popup_width,height=popup_height)
            popup = folium.Popup(iframe,max_width=1000)
        else:
            popup = None
        if tooltip_field:
            thtml = d[tooltip_field]
            tooltip = folium.map.Tooltip(thtml)
        else:
            tooltip = None
        if circle_type == "CircleMarker":
            ff = folium.CircleMarker
        else:
            ff = folium.Circle
        if fill is not None:
            fill_none = True
        else:
            fill_none = False
        fm = ff(location=[latv,lonv],radius=radius,
                fill_color=fill,fill=fill_none,color=color,
                fill_opacity=fill_opacity,opacity=opacity,
                popup=popup,tooltip=tooltip)
        fm.add_to(fg)
    fg.add_to(mapf)

# This gets hex codes for a pallette
# eg get_map('Blues',5)
# or get_map('viridis',4)
def get_map(name, n):
    cmap = cm.get_cmap(name, n)
    res_hex = []
    # not sure if it matters to use 
    for i in range(cmap.N):
        hex = rgb2hex(cmap(i))
        res_hex.append(hex)
    return res_hex

# Makes a hex palette given labels
def make_palette(labs, name):
    hex_map = get_map(name, len(labs))
    res_map = {l:h for l,h in zip(labs,hex_map)}
    return res_map

def create_cols(data,var,new_var,cuts,col='PuBu',extra=True,int_c=True):
    labs = []
    for i,c in enumerate(cuts[:-1]):
        beg, end = cuts[i],cuts[i+1]
        if int_c:
            if (beg == 0) & (end == 1):
                labs.append(f'0')
            else:
                labs.append(f'{beg}-{end-1}')
        else:
            if beg == 0:
                labs.append(f'[{beg}-{end})')
            else:
                labs.append(f'[{beg}-{end})')
    # fix zero boundary issue
    if cuts[0] == 0:
        cuts[0] = -0.01
    data[new_var] = pd.cut(data[var],cuts,labels=labs,right=False).astype(str)
    # fig zero boundary
    if extra:
        lab_cols = make_palette(['Extra'] + labs,col)
        lab_cols.pop('Extra')
        return lab_cols
    else:
        lab_cols = make_palette(labs,col)
        return lab_cols


# di should have {'label':'color'}
# and be in the order you want
def build_svg(di,group_name,edge='#D3D3D3',fill_opacity=0.5,edge_weight=1):
    # If edge_weight is 0, do it as 0
    if edge_weight == 0:
        loc_edge = 0
    else:
        loc_edge = 2
    fin_leg = f"<span>{group_name}"
    for lab,col in di.items():
        fin_leg += '<br><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<svg width="10" height="10">'
        fin_leg += f'<rect width="12" height="12" fill-opacity="{fill_opacity}"'
        fin_leg += f' fill="{col}" style="stroke-width:{loc_edge};stroke:{edge}" />'
        fin_leg += f'</svg> {lab}</span>'
    fin_leg += "</span>"
    return fin_leg

# Currying the style function
# https://leafletjs.com/reference.html#path-option
def style_wrap(fillColor, fillOpacity, color, weight):
    def style_func(x):
        di = {"fillColor": fillColor,
              "fillOpacity": fillOpacity,
              "color": color,
              "weight": weight}
        return di
    return style_func

# Adding Choropleth
def add_choro(mapf,
              poly_df,
              col_field,
              lab_di,
              tab_fields,
              title = None,
              footer = None,
              name="Choropleth",
              edge='#D3D3D3',
              edge_weight=1,
              opacity=0.65,
              tab_names = ['Field','Value']):
    poly2 = poly_df.to_crs('EPSG:4326')
    poly2['area'] = poly_df.geometry.area
    poly2 = poly2.sort_values(by='area',ascending=False).reset_index(drop=True)
    # creating the legend
    svg_name = build_svg(lab_di,name,edge,opacity,edge_weight)
    fg = folium.FeatureGroup(name=svg_name,overlay=True,control=True)
    # Making the necessary style functions
    sf = {}
    hf = {}
    for lab,col in lab_di.items():
        sf[lab] = style_wrap(col,opacity,edge,edge_weight)
        # highlight function
        hf[lab] = style_wrap(col,opacity*0.5,edge,4)
    # Looping over polygons, adding into map
    for i in range(poly_df.shape[0]):
        sub_data = poly2.loc[[i]].copy()
        geo_js = round_geo(sub_data.geometry)
        choro_lab = sub_data[col_field][i]
        geo_fol = folium.GeoJson(data=geo_js,
                                 style_function=sf[choro_lab],
                                 highlight_function=hf[choro_lab],
                                 name=svg_name,
                                 overlay=True,
                                 control=True)
        lab_data = pd.DataFrame(sub_data[tab_fields].T.reset_index())
        lab_data.columns = tab_names
        html_lab = lab_data.to_html(index=False,header=True)
        if title is not None:
            html_lab = sub_data[title][i] + html_lab
        if footer is not None:
            html_lab += sub_data[footer][i]
        popup = folium.Popup(html_lab)
        popup.add_to(geo_fol)
        geo_fol.add_to(fg)
    fg.add_to(mapf)


# This adds crime de-coder logo
# and methods note to leaflet map

today = datetime.now().strftime('%Y-%m-%d')

logo_js = '''
</script>
<style>
/* These are styles at the end
   the table css above comes at the
   beginning */

/* Marker clusters different colors */
.marker-cluster-small div {
    background-color: rgba(247, 104, 161, 0.6);
}

.marker-cluster-small {
    background-color: rgba(247, 104, 161, 0.4);
}


.marker-cluster-medium div {
    background-color: rgba(197,27,138,0.6);
}

.marker-cluster-medium {
    background-color: rgba(197,27,138,0.4);
}

.marker-cluster-large div {
    background-color: rgba(122,1,119,0.6);
}

.marker-cluster-large {
    background-color: rgba(122,1,119,0.4);
}
</style>
<script>var logo = '<a href="https://crimede-coder.com/" target="_blank">' +
'<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 325.16 171.35">' +
'<defs><style>.cls-1{fill:#010101;}.cls-2{fill:none;stroke:#fff;stroke-miterlimit:10;stroke-width:3px;}' +
'.cls-3{font-size:84.54px;font-family:Helvetica-Bold,Helvetica;font-weight:700;}.cls-3,.cls-4{fill:#fff;}' +
'.cls-4{font-size:65.23px;font-family:Helvetica;}</style></defs><rect class="cls-1" width="325.16" height="171.35">' +
'</rect><rect class="cls-2" x="6.61" y="5.42" width="313.29" height="160.52"></rect>' +
'<text class="cls-3" transform="translate(15.42 77.15) scale(1.06 1)">CRIME</text>' +
'<text class="cls-4" transform="translate(15.42 141.98) scale(1.01 1)">De-Coder</text></svg>'+
'</a>'

var methods = '<p>Map created by Andrew Wheeler on ???today???.</p>'

window.onload = function() {
    document.querySelector("section.leaflet-control-layers-list").insertAdjacentHTML("afterbegin",logo);
    document.querySelector("div.leaflet-control-attribution").insertAdjacentHTML("afterbegin",methods);
    let rad = document.querySelectorAll("input.leaflet-control-layers-selector")
    // for only 2 does it for the radio buttons
    // if I do it for everything, when clicking
    // on or off layers appends it
    for (let i = 0; i < 2; i++) {
        rad[i].addEventListener("change", add_note);
    }

    function add_note() {
      document.querySelector("div.leaflet-control-attribution").insertAdjacentHTML("afterbegin",methods);
    }

    // making sure the first radio button is selected
    // document.querySelectorAll('input[type=text]')
    //rad[0].checked = true;

    rad[0].click();
};

'''

logo_js_today = logo_js.replace('???today???',today)

# If I want to put stuff in header, do it here
# gets inserted at the top of the HTML
table_css = '''<style>
/* Alternate row coloring */
tr:nth-child(even) {
  background-color: #f2f2f2;
}

/* Right align columns 2/3 */
td:nth-child(2), td:nth-child(3),
th:nth-child(2), th:nth-child(3) {
  text-align: right;
}

/* Background color of header */
th {
  background-color: #DDDDDD
}

/* No borders in Table */
table {
  border: none;
  border-collapse:collapse;
  width: 200px;
  border-bottom: none;
  border-top: none;
}

/* No vertical borders in header and cells */
table, th, td {
  border-left: none;
  border-right: none;
  border-spacing: 2%;
}

/* Cell padding */
th, td {
  padding: 0% 2% 0% 2%;
}
</style>'''


def save_map(mapf,file="temp.html",add_css=table_css,add_js=logo_js_today,layer=True):
    # Need to add in layercontrol at the very end
    if layer:
        folium.LayerControl(collapsed=False).add_to(mapf)
    # Adding in CSS and javascript
    css_element = folium.Element(add_css)
    js_element = folium.Element(add_js)
    mapf.get_root().header.add_child(css_element)
    # now adding in javascript at the end
    # https://github.com/python-visualization/folium/issues/86
    html = mapf.get_root()
    html.script.get_root().render()
    html.script._children['XXX_LogoJavascript'] = js_element
    # cleaning up UUID, only keeping first 8
    ru = r'([0-9a-f]{8})[0-9a-f]{4}[0-9a-f]{4}[0-9a-f]{4}[0-9a-f]{12}'
    res = html.script.get_root().render()
    res = re.sub(ru,r'\1',res)
    # cleaning up extra whitespace
    rl = []
    for s in res.split('\n'):
        ss = s.strip()
        if len(ss) > 0:
            rl.append(ss)
    rlc = '\n'.join(rl)
    #mapf.save(file)
    if os.path.exists(file):
        os.remove(file)
    with open(file, "w") as f:
        f.write(rlc)