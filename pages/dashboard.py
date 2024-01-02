import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from utils import saveSession, getSession
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

st.set_page_config(page_title="SHAP Dashboard",
                   page_icon=":bar_chart:", layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title(":bar_chart: SHAP Dashboard")

if getSession("uploaded_file") != False:
    st.caption("**Dataset:** *{}*".format(getSession("uploaded_file")))


@st.cache_resource(show_spinner=False)
def load_shap(many=False, idx_instance=0):
    # many -> means shap explainer for many instance
    # idx_instance = idx instance to be explainer

    target = getSession("target_feature")

    data_csv = getSession("data_csv")
    data_csv = data_csv.drop(columns=[target])

    kmeans = getSession("kmeans")
    background_dataset = kmeans.data  # <-- background dataset

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

        if many == True:
            shap_output_many.append(explainer(data_csv))
        else:
            idx_instance = 0 if idx_instance == False else idx_instance
            instance = data_csv.iloc[idx_instance:idx_instance+1]
            shap_output_instance.append(explainer(instance))

    return shap_explainers, shap_output_many, shap_output_instance


if not getSession('confirmInit'):
    st.error("Please complete data initialization first at home page")
else:
    st.sidebar.button("Reset Everything", type="secondary",
                      key="sidebar_reset", on_click=lambda: st.session_state.clear())

    st.subheader("Models")

    # SHAP SECTION
    with st.spinner("Calculating Shapley Value... This may take a while"):
        shap_explainers, shap_output_many, shap_output_instance = load_shap(
            many=True)
    ######

    tab1, tab2, tab3 = st.tabs(
        ["Linear Regression", "K Nearest Neighbor", "Support Vector Regression"])
    tabs = [tab1, tab2, tab3]

    x_train = getSession("x_train")

    data_csv = getSession("data_csv")
    target_feature = getSession("target_feature")

    data_csv_notarget = getSession("data_csv").drop(columns=[target_feature])

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
                st.subheader("SHAP Explanation Output")
                st.caption("Total Data : {}".format(
                    len(data_csv)))
                st.markdown(
                    "**Mean Base Value:** *{}*".format(np.mean(shap_output_many[i].base_values)))
                shap_overall_sum = pd.DataFrame(
                    {'Features': x_train.columns, 'SHAP Value': np.sum(shap_output_many[i].values, axis=0)})
                shap_overall_sum = shap_overall_sum
                st.markdown("Overall **sum** of SHAP feature contributions: ")
                st.write(shap_overall_sum)

            st.subheader("SHAP Feature Dependency Option")
            col1, col2 = st.columns([1, 2])

            # get max SHAP value for each feature
            shap_max_value = np.max(shap_output_many[i].values, axis=0)

            # Initialize
            dependence_plot_area = st.empty()

            with col1:
                st.write("Select a features to be plotted")
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    main_feature = st.selectbox(
                        '1st Feature', [x for x in x_train.columns], index=0, key="main_feature_dplot_tab{}".format(i))
                    st.caption("Main Feature to be plotted as the main y-axis")
                with col_b:
                    second_feature = st.selectbox(
                        '2nd Feature', [x for x in x_train.columns], index=int(np.argmax(shap_max_value)), key="second_feature_dplot_tab{}".format(i))
                    st.caption(
                        "Secondary or interaction feature to be plotted against main feature")

                st.caption(
                    "Changes are applied immediately upon selection")
            with col2:
                main_feature = getSession('main_feature_dplot_tab{}'.format(i))
                second_feature = getSession(
                    'second_feature_dplot_tab{}'.format(i))

                shap.dependence_plot(ind=main_feature, interaction_index=second_feature,
                                     shap_values=shap_output_many[i].values, features=data_csv_notarget, feature_names=data_csv_notarget.columns, show=False)

                st.pyplot(plt.gcf())
                plt.clf()

            st.subheader("Instance Explanation")
            st.caption("SHAP Visual Explanation based on single instance")

            d_col1, d_col2 = st.columns([2, 1])

            with d_col1:
                idx_instance = getSession("instance_idx_tab{}".format(i))
                idx_instance = 0 if idx_instance == False else idx_instance

                explainer = shap_explainers[i]
                shap_output_instance = explainer(
                    data_csv_notarget.iloc[idx_instance:idx_instance+1])

                waterfall_plot = shap.waterfall_plot(
                    shap_output_instance[0], show=False)
                st.pyplot(waterfall_plot)
                plt.clf()

            with d_col2:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("Pick an instance to be explained")
                    st.number_input('Instance Index',
                                    min_value=0, max_value=len(data_csv)-1, value=0, key="instance_idx_tab{}".format(i))
                with col2:
                    idx_instance = getSession("instance_idx_tab{}".format(i))
                    st.write("Instance Data")
                    st.write(data_csv.iloc[idx_instance:idx_instance+1].T)

            idx_instance = getSession("instance_idx_tab{}".format(i))
            idx_instance = 0 if idx_instance == False else idx_instance

            force_plot = shap.force_plot(
                shap_output_instance, matplotlib=True, show=False)
            st.pyplot(force_plot)
            plt.clf()

            decision_plot = shap.decision_plot(
                base_value=shap_output_instance.base_values[0], shap_values=shap_output_instance.values[0], features=data_csv_notarget.iloc[idx_instance], highlight=0, show=False)
            st.pyplot(plt.gcf())
            plt.clf()

    # with st.spinner("Loading Dashboard..."):
    #     st.success("Data Initialized")
    #     time.sleep(5)
    #     st.success("Dashboard Loaded")
