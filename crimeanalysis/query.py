'''
Functions to query data
'''

from bs4 import BeautifulSoup as bs
from datetime import datetime
import geopandas as gpd
from io import StringIO
import numpy as np
import pandas as pd
import requests
from urllib.parse import quote, urljoin
import os
import urllib3
import ssl
import time
import traceback
import re

# This grabs CSV file from web apps
def get_csv(url):
    res_csv = requests.get(url,verify=False)
    res_pd = pd.read_csv(StringIO(res_csv.text),low_memory=False)
    return res_pd

# Reads a dataframe from local or CSV
def read_data(file):
    fe = os.path.splitext(file)[-1]
    if file[:4] == 'http':
        if fe == '.csv':
            res = get_csv(url)
        elif (fe == '.xlsx') | (fe == '.xls') | (fe == '.xlsb'):
            res = pd.read_excel(file)
    else:
        try:
            if fe == '.csv':
                res = pd.read_csv(file,low_memory=False)
            elif (fe == '.xlsx') | (fe == '.xls') | (fe == '.xlsb'):
                res = pd.read_excel(file)
            elif fe == '.zip':
                res = pd.read_csv(file,low_memory=False)
        except Exception:
            er = traceback.format_exc()
            err_type = er.split('\n')[-2]
            if err_type == 'pandas.errors.EmptyDataError: No columns to parse from file':
                res = None
            else:
                print(f'\nfile {file} not read properly and it is not due to being empty\n')
                print(er)
                res = None
    return res


# caches file locally if downloaded from URL
def cache(url,file,exist_only=False):
    res = None
    if os.path.exists(file):
        res = read_data(file)
    else:
        if exist_only:
            return res
        res = read_data(url)
        res.to_csv(file,index=False)
    return res


def cache_query(file, func, query_kwargs,exist_only=False):
    res = None
    if os.path.exists(file):
        res = read_data(file)
    else:
        if exist_only:
            return res
        res = func(**query_kwargs)
        res.to_csv(file,index=False)
    return res


def head_check(file, url):
    rh = requests.head(url)
    last_modified = rh.headers['Last-Modified']
    with open(file, "r") as f:
        old_modified = f.read()
    check = last_modified == old_modified
    return check, last_modified, old_modified

def over_modified(file,text):
    with open(file,"w") as f:
        f.write(text)

def get_files(url,extensions):
    res = requests.get(url,verify=False)
    soup = bs(res.text,'lxml')
    href = soup.find_all("a")
    url_links = []
    for h in href:
        link = h['href']
        fe = os.path.splitext(link)[-1]
        if fe in extensions:
            url_links.append(urljoin(url,link))
    return url_links

# ESRIs time unit
def esri_time(field,offset=-5*60*60):
    fl = pd.to_numeric(field,errors='coerce')
    return pd.to_datetime(fl/1000 + offset,errors='coerce',unit='s')

def rev_esri(offset=-5*60*60):
    pass

# Querying ESRI API
def query_esri(base='https://services.arcgis.com/v400IkDOw1ad7Yad/arcgis/rest/services/Police_Incidents/FeatureServer/0/query',
               params={'outFields':"*",'where':"1=1"},
               verbose=False,
               limitSize=None,
               gpd_query=False):
    if verbose:
        print(f'Starting Queries @ {datetime.now()}')
    req = requests
    p2 = params.copy()
    # try geojson first, if fails use normal json
    if 'f' in p2:
        p2_orig_f = p2['f']
    else:
        p2_orig_f = 'geojson'
    p2['f'] = 'geojson'
    fin_url = base + "?"
    amp = ""
    fi = 0
    for key,val in p2.items():
        fin_url += amp + key + "=" + quote(val)
        amp = "&"
    # First, getting the total count
    count_url = fin_url + "&returnCountOnly=true"
    if verbose:
        print(count_url)
    response_count = req.get(count_url)
    # If error, try using json instead of geojson
    if 'error' in response_count.json():
        if verbose:
            print('geojson query failed, going to json')
        p2['f'] = 'json'
        fin_url = fin_url.replace('geojson','json')
        count_url = fin_url + "&returnCountOnly=true"
        response_count2 = req.get(count_url)
        count_n = response_count2.json()['count']
    else:
        try:
            count_n = response_count.json()["properties"]["count"]
        except:
            count_n = response_count.json()['count']
    if verbose:
        print(f'Total count to query is {count_n}')
    # Getting initial query
    if p2_orig_f != 'geojson':
        fin_url = fin_url.replace('geojson',p2_orig_f)
    dat_li = []
    if limitSize:
        fin_url_limit = fin_url + f"&resultRecordCount={limitSize}"
    else:
        fin_url_limit = fin_url
    if gpd_query:
        full_response = gpd.read_file(fin_url_limit)
        dat = full_response
    else:
        full_response = req.get(fin_url_limit)
        dat = gpd.read_file(StringIO(full_response.text))
    # If too big, getting subsequent chunks
    chunk = dat.shape[0]
    if chunk == count_n:
        d2 = dat
    else:
        if verbose:
            print(f'The max chunk size is {chunk:,}, total rows are {count_n:,}')
            print(f'Need to do {np.ceil(count_n/chunk):,.0f} total queries')
        offset = chunk
        dat_li = [dat]
        remaining = count_n - chunk
        while remaining > 0:
            if verbose:
                print(f'Remaining {remaining}, Offset {offset}')
            offset_val = f"&cacheHint=true&resultOffset={offset}&resultRecordCount={chunk}"
            off_url = fin_url + offset_val
            if gpd_query:
                part_response = gpd.read_file(off_url)
                dat_li.append(part_response.copy())
            else:
                part_response = req.get(off_url)
                dat_li.append(gpd.read_file(StringIO(part_response.text)))
            offset += chunk
            remaining -= chunk
        d2 = pd.concat(dat_li,ignore_index=True)
    if verbose:
        print(f'Finished queries @ {datetime.now()}')
    # checking to make sure numbers are correct
    if d2.shape[0] != count_n:
        print('Warning! Total count {count_n} is different than queried count {d2.shape[0]}')
    # if geojson, just return
    if p2['f'] == 'geojson':
        return d2
    # if json, can drop geometry column
    elif p2['f'] == 'json':
        if 'geometry' in list(d2):
            return d2.drop(columns='geometry')
        else:
            return d2

# newer Socrata does not have a limit
def query_socrata(base,add_params):
    # Get the total number of items to query
    tot_query = base + add_params + "&$group=&$select=count(*)%20AS%20tot"
    #print(tot_query)
    # the tot query *NEEDS* to be json format
    res_tot = requests.get(tot_query.replace('geojson','json'),verify=False)
    #print(res_tot.text)
    totn = int(res_tot.json()[0]['tot'])
    # with Socrata, can query the whole data
    whole_query = base + add_params + f'&$limit={totn}'
    #print(whole_query)
    res = requests.get(whole_query,verify=False)
    if 'geojson' in whole_query:
        data = gpd.read_file(res.text)
    else:
        data = pd.DataFrame(res.json())
    return data


# TODO
def query_socrata_page(base,add_params,page_limit=1000):
    # Get the total number of items to query
    tot_query = base + add_params + "&$group=&$select=count(*)%20AS%20tot"
    #print(tot_query)
    # the tot query *NEEDS* to be json format
    res_tot = requests.get(tot_query.replace('geojson','json'),verify=False)
    #print(res_tot.text)
    totn = int(res_tot.json()[0]['tot'])
    # with Socrata, can query the whole data
    whole_query = base + add_params + f'&$limit={totn}'
    #print(whole_query)
    res = requests.get(whole_query,verify=False)
    if 'geojson' in whole_query:
        data = gpd.read_file(res.text)
    else:
        data = pd.DataFrame(res.json())
    return data


cary_base = ('https://data.townofcary.org/api/explore/v2.1/catalog/datasets'
             '/cpd-incidents/exports/csv?lang=en&timezone=US%2FEastern'
             '&use_labels=true&delimiter=%2C')

def query_opendata(base=cary_base,add_params='&where=year=2024'):
    # base should be the export endpoint
    return get_csv(base + add_params)

def query_opendata_geo(base):
    res = requests.get(base,verify=False)
    areas = gpd.read_file(StringIO(res.text))
    return areas

def phoenix_query(offset=0,limit=30000):
    url = 'https://www.phoenixopendata.com/api/3/action/datastore_search'
    data = {'resource_id': '0ce3411a-2fc6-4302-a33f-167f68608a20',
            'limit': str(limit),
            'offset': str(offset)}
    #'sort': '_id desc'
    res = requests.get(url,params=data)
    rj = res.json()
    totn = rj['result']['total']
    df = pd.DataFrame(rj['result']['records'])
    return df

def phoenix_max():
    url = 'https://www.phoenixopendata.com/api/3/action/datastore_search'
    data = {'resource_id': '0ce3411a-2fc6-4302-a33f-167f68608a20',
            'limit': '1',
            'offset': '0'}
    res = requests.get(url,params=data)
    rj = res.json()
    totn = rj['result']['total']
    return totn

