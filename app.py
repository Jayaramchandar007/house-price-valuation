import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Valuation & Market Explorer", page_icon="🏠", layout="wide")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def clean_data(raw):
    d=raw.copy()
    d["location"]=d["location"].fillna("Unknown").astype(str).str.strip()
    d["location"]=d["location"].replace("", "Unknown")
    d["area_type"]=d["area_type"].fillna("Unknown").astype(str)
    d["availability"]=d["availability"].fillna("Unknown").astype(str)
    d["bedrooms"]=d["size"].astype(str).str.extract(r"(\d+)")[0]
    d["bedrooms"]=pd.to_numeric(d["bedrooms"], errors="coerce")
    def sqft_num(v):
        if pd.isna(v): return np.nan
        s=str(v).lower().replace(",","").strip()
        # Support simple numeric ranges such as 1200-1500 by averaging
        nums=re.findall(r"\d+(?:\.\d+)?",s)
        if not nums: return np.nan
        vals=[float(x) for x in nums]
        if len(vals)>=2 and ("-" in s or "to" in s):
            return (vals[0]+vals[1])/2
        return vals[0]
    d["area_sqft"]=d["total_sqft"].apply(sqft_num)
    d["bath"]=pd.to_numeric(d["bath"],errors="coerce")
    d["balcony"]=pd.to_numeric(d["balcony"],errors="coerce")
    d["price"]=pd.to_numeric(d["price"],errors="coerce")
    d=d.dropna(subset=["area_sqft","price","bedrooms"])
    d=d[(d["area_sqft"]>100)&(d["area_sqft"]<50000)&(d["price"]>0)]
    d=d.drop_duplicates()
    return d

@st.cache_resource
def train_model(d):
    features=["area_sqft","bedrooms","bath","balcony","location","area_type","availability"]
    numeric=["area_sqft","bedrooms","bath","balcony"]
    categorical=["location","area_type","availability"]
    prep=ColumnTransformer([
        ("num",SimpleImputer(strategy="median"),numeric),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                         ("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)
    ])
    model=Pipeline([("prep",prep),("rf",RandomForestRegressor(n_estimators=180,min_samples_leaf=2,
                max_features=0.8,random_state=42,n_jobs=-1))])
    X=d[features]; y=d["price"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    metrics={"mae":mean_absolute_error(y_test,pred),"rmse":np.sqrt(mean_squared_error(y_test,pred)),
             "r2":r2_score(y_test,pred)}
    return model,metrics,features

st.title("🏠 House Price Valuation & Market Explorer")
st.caption("Explore Bengaluru property prices, estimate a property's value, and compare neighborhoods.")

with st.sidebar:
    st.header("Explore the dataset")
    uploaded=st.file_uploader("Upload another CSV (optional)", type=["csv"])
    st.markdown("---")
    st.caption("Default dataset: Bengaluru house prices")

raw=load_data(uploaded) if uploaded is not None else pd.read_csv("bengaluru_house_prices.csv")
d=clean_data(raw)
model,metrics,features=train_model(d)

st.sidebar.header("Market filters")
locations=sorted(d["location"].unique().tolist())
selected_locations=st.sidebar.multiselect("Location(s)",locations,default=[])
area_min=float(max(100,d["area_sqft"].quantile(0.01)))
area_max=float(d["area_sqft"].quantile(0.99))
area_range=st.sidebar.slider("Area (sq ft)",min_value=int(area_min),max_value=int(area_max),
    value=(int(area_min),int(area_max)),step=50)
bed_opts=sorted(d["bedrooms"].dropna().unique().astype(int).tolist())
selected_beds=st.sidebar.multiselect("Bedrooms",bed_opts,default=[])
price_min=float(d["price"].quantile(0.01)); price_max=float(d["price"].quantile(0.99))
price_range=st.sidebar.slider("Price (₹ lakh)",min_value=int(price_min),max_value=int(price_max),
    value=(int(price_min),int(price_max)),step=5)

filtered=d[(d["area_sqft"].between(area_range[0],area_range[1])) & (d["price"].between(price_range[0],price_range[1]))]
if selected_locations: filtered=filtered[filtered["location"].isin(selected_locations)]
if selected_beds: filtered=filtered[filtered["bedrooms"].isin(selected_beds)]

# KPI row
c1,c2,c3,c4=st.columns(4)
c1.metric("Properties in view",f"{len(filtered):,}")
c2.metric("Average price",f"₹{filtered.price.mean():,.2f} L" if len(filtered) else "—")
c3.metric("Median price",f"₹{filtered.price.median():,.2f} L" if len(filtered) else "—")
c4.metric("Avg. price / sq ft",f"₹{(filtered.price*100000/filtered.area_sqft).mean():,.0f}" if len(filtered) else "—")

st.subheader("Market overview")
if filtered.empty:
    st.warning("No properties match these filters. Adjust the filters to see results.")
else:
    left,right=st.columns(2)
    with left:
        fig=px.histogram(filtered,x="price",nbins=35,title="Property price distribution (₹ lakh)")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=px.scatter(filtered,x="area_sqft",y="price",color="bedrooms",hover_data=["location","bath","balcony"],
                       title="Area vs. price")
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("Neighborhood price comparison")
    compare=d.copy()
    if selected_locations: compare=compare[compare.location.isin(selected_locations)]
    # Top locations by property count in current filter scope, for readable plots
    loc_stats=compare.groupby("location").agg(avg_price=("price","mean"),median_price=("price","median"),
                    listings=("price","size")).reset_index()
    loc_stats=loc_stats.sort_values("listings",ascending=False).head(25)
    fig=px.bar(loc_stats.sort_values("avg_price"),x="avg_price",y="location",orientation="h",
               title="Average house price by neighborhood (top 25 by listing count)",
               labels={"avg_price":"Average price (₹ lakh)","location":"Neighborhood"})
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Neighborhood chart is based on the dataset and listing counts, not live market listings.")

st.subheader("🔮 Property valuation")
with st.form("valuation_form"):
    col1,col2,col3=st.columns(3)
    with col1:
        val_location=st.selectbox("Location",locations,index=0)
        val_area=st.number_input("Area (sq ft)",min_value=100.0,value=1200.0,step=50.0)
    with col2:
        val_bed=st.number_input("Bedrooms",min_value=1,max_value=20,value=2,step=1)
        val_bath=st.number_input("Bathrooms",min_value=1.0,max_value=20.0,value=2.0,step=1.0)
    with col3:
        val_bal=st.number_input("Balconies",min_value=0.0,max_value=10.0,value=1.0,step=1.0)
        val_area_type=st.selectbox("Area type",sorted(d.area_type.unique()))
    val_avail=st.selectbox("Availability",sorted(d.availability.unique()))
    submitted=st.form_submit_button("Estimate house price",type="primary")
if submitted:
    row=pd.DataFrame([{"area_sqft":val_area,"bedrooms":val_bed,"bath":val_bath,"balcony":val_bal,
                       "location":val_location,"area_type":val_area_type,"availability":val_avail}])
    estimate=float(model.predict(row[features])[0])
    st.success(f"Estimated price: ₹{estimate:,.2f} lakh")
    st.write(f"Approximate price per sq ft: ₹{estimate*100000/val_area:,.0f}")
    local=d[d.location==val_location]
    if not local.empty:
        st.write(f"Dataset average for **{val_location}**: ₹{local.price.mean():,.2f} lakh")
    st.caption("This is a model estimate based on historical dataset patterns, not an appraisal or guaranteed market price.")

st.subheader("📋 Filtered property records")
if not filtered.empty:
    show_cols=["location","area_sqft","bedrooms","bath","balcony","area_type","availability","price"]
    st.dataframe(filtered[show_cols].rename(columns={"area_sqft":"Area (sq ft)","bedrooms":"Bedrooms",
        "price":"Price (₹ lakh)"}),use_container_width=True,hide_index=True)
    st.download_button("Download filtered data as CSV",filtered[show_cols].to_csv(index=False).encode("utf-8"),
                       file_name="filtered_house_prices.csv",mime="text/csv")

with st.expander("🤖 Model performance"):
    st.write("Model: Random Forest Regression with numeric imputation and categorical encoding.")
    m1,m2,m3=st.columns(3)
    m1.metric("R² score",f"{metrics['r2']:.3f}")
    m2.metric("MAE",f"₹{metrics['mae']:.2f} lakh")
    m3.metric("RMSE",f"₹{metrics['rmse']:.2f} lakh")
    st.caption("Scores are measured on a held-out test split. Performance can vary; price estimates should be treated as approximate.")

with st.expander("ℹ️ About this project"):
    st.markdown("""- **Dataset:** Bengaluru house-price records supplied for this project.
- **Filters:** Location, area, bedrooms, and price.
- **Predictor:** Random Forest regression.
- **Charts:** Price distribution, area-price relationship, neighborhood comparison.
- **Limitations:** Dataset is historical and may contain outliers or nonstandard area formats. The app is for educational exploration and not financial advice.""")
