import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from utils import saveSession, getSession
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

st.set_page_config(page_title="SHAP Dashboard",
                   page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: SHAP Dashboard")

if getSession("uploaded_file") != False:
    st.caption("**Dataset:** *{}*".format(getSession("uploaded_file")))


@st.cache_resource(show_spinner=False)
def load_shap(idx_instance=0):

    target = getSession("target_feature")

    data_csv = getSession("data_csv")
    data_csv = data_csv.drop(columns=[target])

    kmeans = getSession("kmeans")
    background_dataset = kmeans.data  # <-- background dataset

    instance = data_csv.iloc[idx_instance:idx_instance+1]

    model_linear_regression = getSession('model_linear_regression')
    model_knn = getSession('model_knn')
    model_svr = getSession('model_svr')

    models = [model_linear_regression, model_knn, model_svr]

    shap_output_many = []
    shap_output_instance = []
    shap_explainers = []

    for model in models:
        explainer = shap.KernelExplainer(
            model.predict, background_dataset)
        shap_explainers.append(explainer)

        shap_output_instance.append(explainer(instance))
        shap_output_many.append(explainer(data_csv))

    return shap_explainers, shap_output_many, shap_output_instance


if not getSession('confirmInit'):
    st.error("Please complete data initialization first at home page")
else:
    st.subheader("Models")

    # SHAP SECTION
    with st.spinner("Calculating Shapley Value... This may take a while"):
        shap_explainers, shap_output_many, shap_output_instance = load_shap()
    ######

    tab1, tab2, tab3 = st.tabs(
        ["Linear Regression", "K Nearest Neighbor", "Support Vector Regression"])
    tabs = [tab1, tab2, tab3]

    for i in range(len(tabs)):
        tab = tabs[i]
        with tab:
            col1, col2 = st.columns([3, 1])

            with col1:

                beeswarm_plot = shap.plots.beeswarm(
                    shap_output_many[i], show=False)

                st.subheader("SHAP Feature Contribution across Dataset")
                st.pyplot(plt.gcf())
                plt.clf()

            with col2:
                st.subheader("SMALL COL")

    # with st.spinner("Loading Dashboard..."):
    #     st.success("Data Initialized")
    #     time.sleep(5)
    #     st.success("Dashboard Loaded")
