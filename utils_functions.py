import pandas as pd
import numpy as np
import requests
import json


## Function for getting postal code, geo coordinates of addresses
def find_postal(add):
    '''With the block number and street name, get the full address of the hdb flat,
    including the postal code, geogaphical coordinates (lat/long)'''
    
    # Do not need to change the URL
    url= "https://developers.onemap.sg/commonapi/search?returnGeom=Y&getAddrDetails=Y&pageNum=1&searchVal="+ add        
    response = requests.get(url)
    try:
        data = json.loads(response.text) 
    except ValueError:
        print('JSONDecodeError')
        pass
    
    return data
    
def find_nearest(house, amenity, radius=2):
    """
    this function finds the nearest locations from the 2nd table from the 1st address
    Both are dataframes with a specific format:
        1st column: any string column ie addresses taken from the "find_postal_address.py"
        2nd column: latitude (float)
        3rd column: longitude (float)
    Column name doesn't matter.
    It also finds the number of amenities within the given radius (default=2)
    """
    from geopy.distance import geodesic

    results = {}
    # first column must be address
    for index,flat in enumerate(house.iloc[:,0]):
        
        # 2nd column must be latitude, 3rd column must be longitude
        flat_loc = (house.iloc[index,1],house.iloc[index,2])
        flat_amenity = ['','',100,0]
        amenity_2km = pd.DataFrame({'lat':[], 'lon':[]})

        for ind, eachloc in enumerate(amenity.iloc[:,0]):
            amenity_loc = (amenity.iloc[ind,1],amenity.iloc[ind,2])
            distance = geodesic(flat_loc,amenity_loc)
            distance = float(str(distance)[:-3]) # convert to float

            if distance <= radius:   # compute number of amenities in 2km radius
                flat_amenity[3] += 1
                amenity_2km = amenity_2km.append(pd.DataFrame({'lat':[amenity_loc[0]], 'lon':[amenity_loc[1]]}))

            if distance < flat_amenity[2]: # find nearest amenity
                flat_amenity[0] = flat
                flat_amenity[1] = eachloc
                flat_amenity[2] = distance

        results[flat] = flat_amenity
    return results, amenity_2km

def dist_from_location(house, location):
    """
    this function finds the distance of a location from the 1st address
    First is a dataframe with a specific format:
        1st column: any string column ie addresses taken from the "find_postal_address.py"
        2nd column: latitude (float)
        3rd column: longitude (float)
    Column name doesn't matter.
    Second is tuple with latitude and longitude of location
    """
    from geopy.distance import geodesic
    results = {}
    # first column must be address
    for index,flat in enumerate(house.iloc[:,0]):
        
        # 2nd column must be latitude, 3rd column must be longitude
        flat_loc = (house.iloc[index,1],house.iloc[index,2])
        flat_amenity = ['',100]
        distance = geodesic(flat_loc,location)
        distance = float(str(distance)[:-3]) # convert to float
        flat_amenity[0] = flat
        flat_amenity[1] = distance
        results[flat] = flat_amenity
    return results

def map(data, lat, lon, zoom, amenities_2km, checker):
    import streamlit as st
    import pydeck as pdk 
    
    if checker[0]: mrt = amenities_2km[0]
    else: mrt = None
    if checker[1]: malls = amenities_2km[1]
    else: malls = None
    if checker[2]: schools = amenities_2km[2]
    else: schools = None
    if checker[3]: parks = amenities_2km[3]
    else: parks = None
    if checker[4]: hawkers = amenities_2km[4]
    else: hawkers = None
    if checker[5]: supermarkets = amenities_2km[5]
    else: supermarkets = None
    if checker[6]: data = None
    
    
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer( # mrt - red
                'ScatterplotLayer',
                data=mrt,
                get_position='[lon, lat]',
                get_color='[255, 0, 0, 160]',
                get_radius=200,
            ),
            pdk.Layer( # malls - orange
                'ScatterplotLayer',
                data=malls,
                get_position='[lon, lat]',
                get_color='[255, 102, 0, 160]',
                get_radius=200,
            ),
            pdk.Layer( # schools - blue
                'ScatterplotLayer',
                data=schools,
                get_position='[lon, lat]',
                get_color='[0, 102, 255, 160]',
                get_radius=300,
            ),
            pdk.Layer( # parks - green
                'ScatterplotLayer',
                data=parks,
                get_position='[lon, lat]',
                get_color='[0, 153, 0, 160]',
                get_radius=100,
            ),
            pdk.Layer( # hawkers - purple
                'ScatterplotLayer',
                data=hawkers,
                get_position='[lon, lat]',
                get_color='[204, 0, 204, 160]',
                get_radius=150,
            ),
            pdk.Layer( # supermarkets - brown
                'ScatterplotLayer',
                data=supermarkets,
                get_position='[lon, lat]',
                get_color='[153, 51, 0, 160]',
                get_radius=150,
            ),
            pdk.Layer( # HDB user - black
                'ScatterplotLayer',
                stroked=True,
                data=amenities_2km[6],
                get_position=["LONGITUDE", "LATITUDE"],
                get_color='[0, 0, 0, 160]',
                line_width_min_pixels=5,
                get_line_color=[0, 0, 0],
                get_radius=100,
            ),
            pdk.Layer( # flats
                "HexagonLayer",
                data=data,
                get_position=["LONGITUDE", "LATITUDE"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
         ),
        ]
    ))
    
    
def _max_width_():
    import streamlit as st
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )