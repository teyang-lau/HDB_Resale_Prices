import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from utils_functions import find_postal, find_nearest, dist_from_location, map, _max_width_
import pickle
import streamlit.components.v1 as components

_max_width_()

st.title('Singapore HDB Resale Prices Prediction')

st.image('./Pictures/HDB.jpg', width=900)

# Create user input sidebar

st.sidebar.header('User Input HDB Features')
flat_address = st.sidebar.text_input("Flat Address or Postal Code", '988B BUANGKOK GREEN') # flat address
town = st.sidebar.selectbox('Town', list(['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                          'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
                                          'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                          'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG',
                                          'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN',
                                          'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS','PUNGGOL']),
                            index=10)
flat_model = st.sidebar.selectbox('Flat Model', list(['Model A', 'Improved', 'Premium Apartment', 'Standard',
                                                           'New Generation', 'Maisonette', 'Apartment', 'Simplified',
                                                           'Model A2', 'DBSS', 'Terrace', 'Adjoined flat', 'Multi Generation',
                                                           '2-room', 'Executive Maisonette', 'Type S1S2']), index=0)
flat_type = st.sidebar.selectbox('Flat Type', list(['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']),
                                 index=2)
floor_area = st.sidebar.slider("Floor Area (sqm)", 34,280,93) # floor area
storey = st.sidebar.selectbox('Storey', list(['01 TO 03','04 TO 06','07 TO 09','10 TO 12','13 TO 15',
                                              '16 TO 18','19 TO 21','22 TO 24','25 TO 27','28 TO 30',
                                              '31 TO 33','34 TO 36','37 TO 39','40 TO 42','43 TO 45',
                                              '46 TO 48','49 TO 51']), index=3)
lease_commence_date = st.sidebar.selectbox('Lease Commencement Date', list(reversed(range(1966, 2017))), index=1)


with st.sidebar.beta_expander("Comparison"):
    st.slider("2nd Floor Area (sqm)", 34,280,93)


## Get flat coordinates
coord = find_postal(flat_address)
#coord
flat_coord = pd.DataFrame({'address':[coord.get('results')[0].get('ADDRESS')],
                           'LATITUDE':[coord.get('results')[0].get('LATITUDE')], 
                           'LONGITUDE':[coord.get('results')[0].get('LONGITUDE')]})

## Load amenities coordinates
@st.cache
def load_data(filepath):
    return pd.read_csv(filepath)

supermarket_coord = load_data('Data/supermarket_coordinates_clean.csv')
school_coord = load_data('Data/school_coordinates_clean.csv')
hawker_coord = load_data('Data/hawker_coordinates_clean.csv')
shop_coord = load_data('Data/shoppingmall_coordinates_clean.csv')
park_coord = load_data('Data/parks_coordinates_clean.csv')
mrt_coord = load_data('Data/MRT_coordinates.csv')[['STN_NAME','Latitude','Longitude']]

## Get nearest and number of amenities in 2km radius
# Supermarkets
nearest_supermarket,supermarkets_2km = find_nearest(flat_coord, supermarket_coord)
flat_supermarket = pd.DataFrame.from_dict(nearest_supermarket).T
flat_supermarket = flat_supermarket.rename(columns={0: 'flat', 1: 'supermarket', 2: 'supermarket_dist',
                                                    3: 'num_supermarket_2km'}).reset_index().drop(['index'], axis=1)

# Primary Schools
nearest_school,schools_2km = find_nearest(flat_coord, school_coord)
flat_school = pd.DataFrame.from_dict(nearest_school).T
flat_school = flat_school.rename(columns={0: 'flat', 1: 'school', 2: 'school_dist',
                                          3: 'num_school_2km'}).reset_index().drop('index', axis=1)

# Hawker Centers
nearest_hawker,hawkers_2km = find_nearest(flat_coord, hawker_coord)
flat_hawker = pd.DataFrame.from_dict(nearest_hawker).T
flat_hawker = flat_hawker.rename(columns={0: 'flat', 1: 'hawker', 2: 'hawker_dist',
                                          3: 'num_hawker_2km'}).reset_index().drop('index', axis=1)

# Shopping Malls
nearest_mall,malls_2km = find_nearest(flat_coord, shop_coord)
flat_mall = pd.DataFrame.from_dict(nearest_mall).T
flat_mall = flat_mall.rename(columns={0: 'flat', 1: 'mall', 2: 'mall_dist',
                                      3: 'num_mall_2km'}).reset_index().drop('index', axis=1)

# Parks
nearest_park,parks_2km = find_nearest(flat_coord, park_coord)
flat_park = pd.DataFrame.from_dict(nearest_park).T
flat_park = flat_park.rename(columns={0: 'flat', 1: 'park', 2: 'park_dist',
                                      3: 'num_park_2km'}).reset_index().drop(['index','park'], axis=1)

# MRT
nearest_mrt,mrt_2km = find_nearest(flat_coord, mrt_coord)
flat_mrt = pd.DataFrame.from_dict(nearest_mrt).T
flat_mrt = flat_mrt.rename(columns={0: 'flat', 1: 'mrt', 2: 'mrt_dist',
                                    3: 'num_mrt_2km'}).reset_index().drop('index', axis=1)

amenities_2km = [mrt_2km, malls_2km, schools_2km, parks_2km, hawkers_2km, supermarkets_2km,
                 flat_coord[['LATITUDE', 'LONGITUDE']].astype(float)]

# Distance from Dhoby Ghaut
dist_dhoby = dist_from_location(flat_coord, (1.299308, 103.845285))
flat_coord['dist_dhoby'] = [list(dist_dhoby.values())[0][1]]

## Concat all dataframes
flat_coord = pd.concat([flat_coord, flat_supermarket.drop(['flat'], axis=1), 
                        flat_school.drop(['flat'], axis=1),
                        flat_hawker.drop(['flat'], axis=1),
                        flat_mall.drop(['flat'], axis=1),
                        flat_park.drop(['flat'], axis=1),
                        flat_mrt.drop(['flat'], axis=1)],
                       axis=1)
# st.dataframe(flat_coord)

# Flat Type
replace_values = {'2 ROOM':0, '3 ROOM':1, '4 ROOM':2, '5 ROOM':3, 'EXECUTIVE':4}
flat_coord['flat_type'] = replace_values.get(flat_type)

# Storey
flat_coord['storey_range'] = list(['01 TO 03','04 TO 06','07 TO 09','10 TO 12','13 TO 15',
                                              '16 TO 18','19 TO 21','22 TO 24','25 TO 27','28 TO 30',
                                              '31 TO 33','34 TO 36','37 TO 39','40 TO 42','43 TO 45',
                                              '46 TO 48','49 TO 51']).index(storey)

# Floor Area
flat_coord['floor_area_sqm'] = floor_area

# Lease commence date
flat_coord['lease_commence_date'] = lease_commence_date

# Region
d_region = {'ANG MO KIO':'North East', 'BEDOK':'East', 'BISHAN':'Central', 'BUKIT BATOK':'West', 'BUKIT MERAH':'Central',
       'BUKIT PANJANG':'West', 'BUKIT TIMAH':'Central', 'CENTRAL AREA':'Central', 'CHOA CHU KANG':'West',
       'CLEMENTI':'West', 'GEYLANG':'Central', 'HOUGANG':'North East', 'JURONG EAST':'West', 'JURONG WEST':'West',
       'KALLANG/WHAMPOA':'Central', 'MARINE PARADE':'Central', 'PASIR RIS':'East', 'PUNGGOL':'North East',
       'QUEENSTOWN':'Central', 'SEMBAWANG':'North', 'SENGKANG':'North East', 'SERANGOON':'North East', 'TAMPINES':'East',
       'TOA PAYOH':'Central', 'WOODLANDS':'North', 'YISHUN':'North'}
region_dummy = {'region_East':[0], 'region_North':[0], 'region_North East':[0], 'region_West':[0]}
region = d_region.get(town)
if region == 'East': region_dummy['region_East'][0] += 1
elif region == 'North': region_dummy['region_North'][0] += 1
elif region == 'North East': region_dummy['region_North East'][0] += 1
elif region == 'West': region_dummy['region_West'][0] += 1
#region_dummy
flat_coord = pd.concat([flat_coord, pd.DataFrame.from_dict(region_dummy)], axis=1)

# Flat Model
replace_values = {'Model A':'model_Model A', 'Simplified':'model_Model A', 'Model A2':'model_Model A', 
                  'Standard':'Standard', 'Improved':'Standard', '2-room':'Standard',
                  'New Generation':'model_New Generation',
                  'Apartment':'model_Apartment', 'Premium Apartment':'model_Apartment',
                  'Maisonette':'model_Maisonette', 'Executive Maisonette':'model_Maisonette', 
                  'Special':'model_Special', 'Terrace':'model_Special', 'Adjoined flat':'model_Special', 
                    'Type S1S2':'model_Special', 'DBSS':'model_Special'}
d = {'model_Apartment':[0], 'model_Maisonette':[0], 'model_Model A':[0], 'model_New Generation':[0], 'model_Special':[0]}
if replace_values.get(flat_model) != 'Standard': d[replace_values.get(flat_model)][0] += 1
#d
df = pd.DataFrame.from_dict(d)
flat_coord = pd.concat([flat_coord, pd.DataFrame.from_dict(d)], axis=1)

flat1 = flat_coord[['flat_type', 'storey_range', 'floor_area_sqm', 'lease_commence_date',
       'school_dist', 'num_school_2km', 'hawker_dist', 'num_hawker_2km',
       'park_dist', 'num_park_2km', 'mall_dist', 'num_mall_2km', 'mrt_dist',
       'num_mrt_2km', 'supermarket_dist', 'num_supermarket_2km', 'dist_dhoby',
       'region_East', 'region_North', 'region_North East', 'region_West',
       'model_Apartment', 'model_Maisonette', 'model_Model A',
       'model_New Generation', 'model_Special']]

#st.dataframe(flat_coord)
#st.dataframe(flat1)

## MAP ==============================================================================================
flats = pd.read_csv('Data/flat_coordinates_clean.csv')[['LATITUDE','LONGITUDE']]
flats = flats.append([flat_coord[['LATITUDE', 'LONGITUDE']].astype(float)]*1000, ignore_index=True)
#st.dataframe(flats)
          
zoom_level = 11
st.write("**HDB Resale Flats In Singapore (with amenities)**")

row1_1, row1_2, row1_3, row1_4 = st.beta_columns(4)
with row1_1:
    show_mrt = st.checkbox('MRT Stations',True)
with row1_2:
    show_malls = st.checkbox('Shopping Malls',True)
with row1_3:  
    show_schools = st.checkbox('Primary Schools',True)
row2_1, row2_2, row2_3, row2_4 = st.beta_columns(4)
with row2_1:
    show_parks = st.checkbox('Parks',True)
with row2_2:
    show_hawkers = st.checkbox('Hawker Centers',True)
with row2_3:
    show_supermarkets = st.checkbox('Supermarkets',True)
with row2_4:
    hide_hdb = st.checkbox('Hide HDBs',False)    
    
checker = [show_mrt,show_malls,show_schools,show_parks,show_hawkers,show_supermarkets,hide_hdb]
map(flats, float(flat_coord.iloc[0]['LATITUDE']), float(flat_coord.iloc[0]['LONGITUDE']),
    zoom_level, amenities_2km, checker)
## ==============================================================================================

## Load Trained Random Forest Model and Predict
rfr = pickle.load(open('hdb_prices_rf_model.pkl', 'rb'))
predict1 = rfr.predict(flat1)[0]

st.header('Predicted HDB Resale Price is **SGD$%s**' % ("{:,}".format(int(predict1))))

## Shap
import shap
shap.initjs()

explainer = shap.TreeExplainer(rfr)
shap_values = explainer.shap_values(flat1)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, width=900)

st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], flat1))
with st.beta_expander("See explanation for understanding SHAP values"):
    st.write("""
             """)
    st.markdown("""SHAP (SHapley Additive exPlanations) values allow us to look at feature importance and was first proposed by [Lundberg and Lee (2006)](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf) for model interpretability of any machine learning model. SHAP values have a few advantages:

1. Directionality — Unlike the feature importance from random forest, SHAP values allows us to see the importance of a feature and the direction in which it influences the outcome 
2. Global interpretability — the collective SHAP values can show how much each predictor contributes to the entire dataset (this is not shown here as it takes a long time for a large dataset)
3. Local interpretability — each observation gets its own SHAP values, allowing us to identify which features are more important for each observation
4. SHAP values can be calculated for any tree-based model, which other methods are not able to do

**Red bars**: Features that push the HDB resale price ***higher*** 

**Blue bars**: Features that pull the HDB resale price ***lower*** 

**Width of bars**: Importance of the feature. The wider it is, the higher impact is has on the price
""")



st.text(""" """)
st.text(""" """)
st.text(""" """)

st.subheader('Amenities Within 2km Radius')
with st.beta_expander("MRT Station"):
    st.subheader('Nearest MRT: **%s** (%0.2fkm)' % (flat_coord.iloc[0]['mrt'], flat_coord.iloc[0]['mrt_dist']))
    st.subheader('Number of MRTs within 2km: **%d**' % (flat_coord.iloc[0]['num_mrt_2km']))
with st.beta_expander("Shopping Mall"):
    st.subheader('Nearest Mall: **%s** (%0.2fkm)' % (flat_coord.iloc[0]['mall'], flat_coord.iloc[0]['mall_dist']))
    st.subheader('Number of Malls within 2km: **%d**' % (flat_coord.iloc[0]['num_mall_2km']))
with st.beta_expander("Primary School"):
    st.subheader('Nearest School: **%s** (%0.2fkm)' % (flat_coord.iloc[0]['school'], flat_coord.iloc[0]['school_dist']))
    st.subheader('Number of Schools within 2km: **%d**' % (flat_coord.iloc[0]['num_school_2km']))   
with st.beta_expander("Park"):
    st.subheader('Nearest Park: %0.2fkm' % (flat_coord.iloc[0]['park_dist']))
    st.subheader('Number of Parks within 2km: **%d**' % (flat_coord.iloc[0]['num_park_2km']))
with st.beta_expander("Hawker Center"):
    st.subheader('Nearest Hawker: **%s** (%0.2fkm)' % (flat_coord.iloc[0]['hawker'], flat_coord.iloc[0]['hawker_dist']))
    st.subheader('Number of Hawker within 2km: **%d**' % (flat_coord.iloc[0]['num_hawker_2km'])) 
with st.beta_expander("Supermarket"):
    st.subheader('Nearest Supermarket: **%s** (%0.2fkm)' % (flat_coord.iloc[0]['supermarket'], flat_coord.iloc[0]['supermarket_dist']))
    st.subheader('Number of Supermarket within 2km: **%d**' % (flat_coord.iloc[0]['num_supermarket_2km']))     

#flat_coord

## TO DO
# Expander for price comparison
# HDB Price through the years map -- https://pydeck.gl/gallery/column_layer.html