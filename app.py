import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Air Quality ML App")

# -------------------------
# Helpers & Caching
# -------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        st.info("Please upload a CSV file (Kaggle 'Air Quality Data in India' recommended).")
        return None
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data
def preprocess(df):
    df = df.copy()
    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # --- START FIX ---
    # Check for 'City' column, and try to find a replacement if missing
    if "City" not in df.columns:
        st.warning("Column 'City' not found. Attempting to find a replacement...")
        city_col_candidate = None
        
        # Look for common alternatives
        for col in df.columns:
            if col.lower() in ["city", "location", "station", "station name"]:
                city_col_candidate = col
                break
        
        if city_col_candidate:
            st.success(f"Found '{city_col_candidate}'. Renaming to 'City' for analysis.")
            df = df.rename(columns={city_col_candidate: "City"})
        else:
            # If no replacement found, stop the app
            st.error("Fatal Error: Could not find a 'City' column in the uploaded file.")
            st.subheader("Please ensure your CSV has a column for the city name (e.g., 'City', 'Location', 'Station').")
            st.stop() # Stops the app from running further
    # --- END FIX ---

    # Required columns: City, Date. If Date not parsed, try parse.
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        st.warning("No 'Date' column found. Make sure your CSV has a 'Date' column.")
        df["Date"] = pd.NaT

    # Pollutant columns (best-effort detection)
    possible_pollutants = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]
    pollutant_cols = [c for c in possible_pollutants if c in df.columns]

    # Fill numeric NaNs with column medians
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        median = df[c].median(skipna=True)
        if np.isfinite(median):
            df[c] = df[c].fillna(median)

    # If AQI missing, compute simple proxy as mean of main pollutants
    main_poll = [p for p in ["PM2.5","PM10","NO2","CO","SO2","O3"] if p in df.columns]
    if "AQI" not in df.columns or df["AQI"].isnull().sum() > 0:
        if main_poll:
            df["AQI"] = df[main_poll].mean(axis=1).round(0)
        else:
            df["AQI"] = np.nan

    # If AQI_Bucket missing, create basic bucket from AQI
    def aqi_bucket(a):
        try:
            a = float(a)
        except:
            return "Unknown"
        if a <= 50: return "Good"
        if a <= 100: return "Moderate"
        if a <= 200: return "Unhealthy"
        if a <= 300: return "VeryUnhealthy"
        return "Hazardous"

    if "AQI_Bucket" not in df.columns or df["AQI_Bucket"].isnull().sum() > 0:
        df["AQI_Bucket"] = df["AQI"].apply(aqi_bucket)

    # Add Year and Month fields for OLAP
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df, pollutant_cols

@st.cache_data
def compute_city_aggregates(df, pollutant_cols):
    # City-month average AQI for OLAP visuals
    olap = df.groupby(["City","Year","Month"], as_index=False)["AQI"].mean().rename(columns={"AQI":"Avg_AQI"})
    # City-level pollutant means
    city_means = df.groupby("City", as_index=False)[pollutant_cols].mean() if pollutant_cols else pd.DataFrame()
    return olap, city_means

@st.cache_data
def run_kmeans(city_means_original, k=4): # <-- Renamed input
    if city_means_original.empty:
        return None
    
    # --- START FIX ---
    # Create a copy to avoid mutating the original DataFrame
    city_means = city_means_original.copy()
    # --- END FIX ---
    
    X = city_means.select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    city_means["Cluster"] = labels
    return city_means, km, scaler

@st.cache_data
def train_classifier(df, pollutant_cols):
    # Only use rows with a target
    dfc = df.dropna(subset=["AQI_Bucket"])
    if not pollutant_cols:
        return None
    X = dfc[pollutant_cols].fillna(0)
    y = dfc["AQI_Bucket"].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, target_names=le.classes_, zero_division=0)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return clf, le, acc, report, cm

# -------------------------
# UI - Sidebar: Upload + options
# -------------------------
st.sidebar.title("Data / Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Kaggle dataset recommended)", type=["csv"])
st.sidebar.markdown("If you don't upload, the app will not have data to analyze.")

k_clusters = st.sidebar.slider("K (clusters for KMeans)", 2, 8, 4)
train_model = st.sidebar.checkbox("Train Classification Model", value=True)

# -------------------------
# Load & preprocess
# -------------------------
df = load_data(uploaded_file)
if df is None:
    st.stop()

df, pollutant_cols = preprocess(df)
olap_df, city_means = compute_city_aggregates(df, pollutant_cols)

# -------------------------
# Top: Title & KPIs
# -------------------------
st.title("🌆 Air Quality — Interactive ML App")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Records", int(len(df)))
with col2:
    st.metric("Cities", int(df["City"].nunique()))
with col3:
    # --- START FIX ---
    # Original logic was failing if max() returned a non-number type
    max_year = df["Year"].max()
    
    if pd.isna(max_year):
        latest = "n/a"
    else:
        try:
            latest = int(max_year)
        except (ValueError, TypeError):
            latest = "n/a" # Failsafe
            
    st.metric("Latest Year in Data", latest)
    # --- END FIX ---
with col4:
    avg_aqi = float(df["AQI"].mean())
    st.metric("Overall Avg AQI", f"{avg_aqi:.1f}")


# -------------------------
# Main Tabs: Overview / Clustering / Classification / Predict
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Clustering", "Classification", "Predict"])

# -------------------------
# TAB 1: Overview
# -------------------------
with tab1:
    st.header("Overview & OLAP")
    c1, c2 = st.columns([3,1])
    with c1:
        # City & Year filter
        city_choice = st.selectbox("Choose city (for trend)", options=sorted(df["City"].unique()))
        year_choice = st.selectbox("Choose year", options=sorted(df["Year"].dropna().unique()), index=0)
        filt = df[(df["City"]==city_choice) & (df["Year"]==year_choice)]
        if filt.empty:
            st.warning("No records for selected city/year.")
        else:
            # line chart of AQI over time
            fig = px.line(filt.sort_values("Date"), x="Date", y="AQI", title=f"AQI Trend — {city_choice} ({year_choice})")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Month heatmap for selected city
        heat_df = olap_df[olap_df["City"] == city_choice]
        
        if not heat_df.empty:
            # --- START FIX ---
            # Aggregate by City and Month, averaging across the 'Year' dimension
            heat_agg = heat_df.groupby(["City", "Month"], as_index=False)["Avg_AQI"].mean()
            
            # This pivot will now work, as (City, Month) is unique
            heat_pivot = heat_agg.pivot(index="City", columns="Month", values="Avg_AQI")
            # --- END FIX ---
            
            # plot using seaborn
            fig2, ax2 = plt.subplots(figsize=(4,3))
            sns.heatmap(heat_pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", cbar=False, ax=ax2)
            ax2.set_title("Avg AQI by Month")
            st.pyplot(fig2)
        else:
            st.info("No OLAP monthly data for this city.")

    st.markdown("---")
    st.subheader("Top polluted cities (by overall Avg AQI)")
    topN = st.slider("Top N cities", 5, 30, 10)
    city_avg = df.groupby("City", as_index=False)["AQI"].mean().sort_values("AQI", ascending=False).head(topN)
    fig3 = px.bar(city_avg, x="City", y="AQI", color="AQI", color_continuous_scale="OrRd", title="Top polluted cities")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("Pollutant correlation")
    if pollutant_cols:
        corr = df[pollutant_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(8,4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)
    else:
        st.info("No pollutant columns detected for correlation.")

# -------------------------
# TAB 2: Clustering
# -------------------------
with tab2:
    st.header("Clustering: Group cities by pollutant profile")
    if city_means.empty:
        st.info("No city-level pollutant data available for clustering.")
    else:
        st.write("Using these pollutant features:", city_means.select_dtypes(include=[np.number]).columns.tolist())
        km_res = run_kmeans(city_means, k=k_clusters)
        if km_res is None:
            st.error("Clustering failed.")
        else:
            city_clusters, km_model, scaler = km_res
            st.dataframe(city_clusters.sort_values("Cluster").reset_index(drop=True).head(50))

            # PCA for 2D scatter (simple)
            from sklearn.decomposition import PCA
            Xnum = city_means.select_dtypes(include=[np.number]).fillna(0)
            Xs = scaler.transform(Xnum)
            pca = PCA(2, random_state=42)
            coords = pca.fit_transform(Xs)
            plotdf = city_means.copy()
            plotdf["pc1"] = coords[:,0]
            plotdf["pc2"] = coords[:,1]
            plotdf["Cluster"] = city_clusters["Cluster"]

            fig = px.scatter(plotdf, x="pc1", y="pc2", color=plotdf["Cluster"].astype(str),
                             hover_data=["City"], title="City clusters (PCA projection)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("Cluster counts:")
            st.write(plotdf["Cluster"].value_counts().sort_index())

# -------------------------
# TAB 3: Classification
# -------------------------
with tab3:
    st.header("Classification: Predict AQI Bucket from pollutant values")
    st.write("Features used:", pollutant_cols)
    if not train_model:
        st.info("Enable 'Train Classification Model' in the sidebar to train.")
    else:
        clf_res = train_classifier(df, pollutant_cols)
        if clf_res is None:
            st.info("Not enough data or pollutant columns to train a classifier.")
        else:
            clf, le, acc, report, cm = clf_res
            st.subheader(f"Model accuracy: {acc:.3f}")
            st.text("Classification report:")
            st.text(report)
            st.subheader("Confusion matrix")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax_cm, xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            st.pyplot(fig_cm)

# -------------------------
# TAB 4: Predict (live)
# -------------------------
with tab4:
    st.header("Live prediction — enter pollutant levels")
    st.write("This uses the trained RandomForest model (if available).")

    if pollutant_cols:
        user_vals = {}
        # Use columns for sliders, makes it cleaner
        slider_cols = st.columns(3)
        col_idx = 0
        
        for p in pollutant_cols:
            with slider_cols[col_idx % 3]:
                # Get reasonable min/max/median from the data
                p_median = df[p].median(skipna=True)
                p_max = df[p].max(skipna=True)
                
                # Failsafe if data is all NaN
                if pd.isna(p_median): p_median = 0.0
                if pd.isna(p_max): p_max = 100.0
                
                # Ensure max is greater than median and provides range
                max_val = float(max((p_median * 2) + 1, p_max))
                
                user_vals[p] = st.slider(p, 0.0, max_val, float(p_median))
            col_idx += 1

        st.markdown("---")

        # Try to get the trained model
        if train_model:
            res = train_classifier(df, pollutant_cols)
            if res is None:
                st.error("Model not trained (insufficient data).")
            else:
                clf, le, acc, report, cm = res
                
                # Ensure columns are in the correct order for prediction
                user_df = pd.DataFrame([user_vals])
                user_df = user_df[pollutant_cols] # Re-order to match training
                
                # --- START FIX: SHOW PROBABILITIES ---
                
                # 1. Get probabilities for ALL classes
                pred_probs = clf.predict_proba(user_df.fillna(0))[0]
                class_names = le.classes_
                
                # 2. Get the winning prediction
                pred_idx = np.argmax(pred_probs)
                pred_label = class_names[pred_idx]
                
                # 3. Create a DataFrame for the plot
                prob_df = pd.DataFrame({
                    "Category": class_names,
                    "Probability": pred_probs
                })

                # 4. Display the winner in a large font
                st.subheader("Predicted AQI Category")
                st.success(f"**{pred_label}**")
                
                st.markdown("---")
                
                # 5. Display the bar chart of all probabilities
                st.subheader("Prediction Probabilities")
                st.write("This chart shows the model's confidence for each possible category based on your inputs.")
                
                fig = px.bar(
                    prob_df.sort_values("Probability"), 
                    x="Probability", 
                    y="Category", 
                    orientation='h',
                    text_auto='.1%', # Show percentage on bars
                    title="Model Confidence"
                )
                fig.update_layout(xaxis_title="Probability", yaxis_title="AQI Category")
                st.plotly_chart(fig, use_container_width=True)
                
                # --- END FIX ---
                
                st.write("Model accuracy (on test set):", f"{acc:.3f}")
        else:
            st.info("Enable 'Train Classification Model' in the sidebar to enable live predictions.")
    else:
        st.info("No pollutant columns detected in the dataset. Upload a suitable dataset with pollutant columns.")