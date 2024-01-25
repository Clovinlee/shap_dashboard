import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from numpy.random import default_rng

from utils import getSession
shap.initjs()

st.set_page_config(page_title="SHAP Dashboard",
                   page_icon=":bar_chart:", layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title(":bar_chart: SHAP Dashboard")

st.caption("**SHAP Version:**  *{}*".format(shap.__version__))

if getSession("uploaded_file") != False:
    st.caption("**Dataset:** *{}*".format(getSession("uploaded_file")))

# NOTES
# Each model HAS different pipelines, and model_standard_scaler
# but all model has same x and x_train(scaled and non differs)


@st.cache_resource(show_spinner=False)
def load_shap_many():
    # many -> means shap explainer for many instance
    # idx_instance = idx instance to be explainer

    # Initialize Session Variables
    x_data = getSession("x_data")
    x_data_scaled = getSession("x_data_scaled")

    background_datasets = getSession("background_datasets")

    pipeline_linear_regression = getSession('pipeline_linear_regression')
    pipeline_knn = getSession('pipeline_knn')
    pipeline_rf = getSession('pipeline_rf')

    models_standard_scaler = getSession("models_standard_scaler")
    pipelines = [pipeline_linear_regression, pipeline_knn, pipeline_rf]
    # End

    # linreg, knn, fr
    shap_output_many = []
    shap_explainers = []

    for idx, pipeline in enumerate(pipelines):
        shap_standard_scaler = models_standard_scaler[idx]

        x_shap = x_data if shap_standard_scaler == False else x_data_scaled

        background_dataset = background_datasets[idx]

        explainer = shap.KernelExplainer(
            model=pipeline.predict, data=background_dataset, feature_names=x_shap.columns)

        shap_explainers.append(explainer)

        shap_output = explainer(x_shap)

        # revert the data to its original state
        if (shap_standard_scaler):
            shap_output.data = x_data.values

        shap_output_many.append(shap_output)

    return shap_explainers, shap_output_many


@st.cache_resource(show_spinner=False)
def load_shap_instance(_explainer, instance, tab_title, standard_scaler=False):
    shap_output_instance = _explainer(instance)

    # Convert back to original value
    if (standard_scaler):  # means the shap output need to be converted to original value as standard scaler
        x_data = getSession("x_data")
        shap_output_instance.data = x_data.values

    return shap_output_instance


def load_shap_plots(_shap_output_instance, instance, tab_title, plot_name):
    if plot_name == "waterfall":
        return load_shap_waterfall_plot(_shap_output_instance, instance, tab_title)
    elif plot_name == "force" or plot_name == "force_plot":
        return load_shap_force_plot(_shap_output_instance, instance, tab_title)
    elif plot_name == "decision" or plot_name == "decision_plot":
        return load_shap_decision_plot(_shap_output_instance, instance, tab_title)
    else:
        raise ("Plot name not found : {}".format(plot_name))


# @st.cache_data(show_spinner=False) -> This is on purpose because cache dont work well with dependency plot!
def load_shap_dependence_plot(main_feature, interaction_feature, shap_values, features):
    return shap.dependence_plot(ind=main_feature, interaction_index=interaction_feature,
                                shap_values=shap_values, features=features, feature_names=features.columns, show=False)


@st.cache_data(show_spinner=False)
def load_shap_waterfall_plot(_shap_output_instance, instance, tab_title):
    waterfall_plot = shap.waterfall_plot(
        _shap_output_instance[0], show=False)
    return waterfall_plot


@st.cache_data(show_spinner=False)
def load_shap_stacked_force_plot(_shap_output_many, tab_title, x_tab, qty):
    random_state = getSession("random_state")
    rng = default_rng(seed=random_state)
    rng_numbers = rng.choice(len(x_data), size=qty, replace=False)

    _shap_output_many = _shap_output_many[rng_numbers]

    stacked_force_plot = shap.force_plot(_shap_output_many, show=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{stacked_force_plot.html()}</body>"
    plt.clf()
    return shap_html


@st.cache_data(show_spinner=False)
def load_shap_force_plot(_shap_output_instance, instance, tab_title):
    # Note: instane and tab_title does NOTHING here
    # it only help for streamlit cache data to remember which state are saved for each tab

    force_plot = shap.force_plot(
        _shap_output_instance, matplotlib=True, show=False)
    return force_plot

# @st.cache_data(show_spinner=False) -> This is on purpose because cache dont work well with decision plot!


def load_shap_decision_plot(_shap_output_instance, instance, tab_title):
    decision_plot = shap.decision_plot(
        base_value=_shap_output_instance.base_values[0],
        shap_values=_shap_output_instance.values[0],
        features=instance,
        highlight=0,
        show=False)
    return decision_plot


x_data = getSession("x")
x_data_scaled = getSession("x_scaled")
x_train = getSession("x_train")
x_test = getSession("x_test")
# models_standard_scaler (initialized above)


def resetEverything():
    load_shap_many.clear()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()


if not getSession('confirmInit'):
    st.error("Please complete data initialization first at home page")
else:
    st.sidebar.button("Reset Everything", type="secondary",
                      key="sidebar_reset", on_click=resetEverything)
    st.sidebar.button("Clear Cache", type="secondary",
                      key="sidebar_clear_cache", on_click=lambda: st.cache_data.clear())

    st.subheader("Models")

    # Initialization of variables
    shap_explainers = []
    shap_output_many = []

    # SHAP SECTION
    with st.spinner("Calculating Shapley Value... This may take a while"):
        shap_explainers, shap_output_many = load_shap_many()
    ######

    # INITIALIZE TABS DATA FROM SESSION
    x_train = getSession("x_train")
    x_train_scaled = getSession("x_train_scaled")

    x_data = getSession("x_data")
    x_data_scaled = getSession("x_data_scaled")

    models_standard_scaler = getSession("models_standard_scaler")

    background_datasets = getSession("background_datasets")
    # end

    tab_titles = ["Linear Regression",
                  "K Nearest Neighbor", "Random Forest Regression"]

    tab1, tab2, tab3 = st.tabs(
        tab_titles)
    tabs = [tab1, tab2, tab3]

    for i in range(len(tabs)):
        tab = tabs[i]
        tab_standard_scaler = models_standard_scaler[i]
        x_tab = x_data_scaled if tab_standard_scaler else x_data

        tab_title = tab_titles[i]

        with tab:
            col1, col2 = st.columns([3, 1])

            with col1:

                beeswarm_plot = shap.plots.beeswarm(
                    shap_output_many[i], show=False)

                st.subheader("SHAP Feature Contribution across Dataset")
                st.pyplot(beeswarm_plot)
                plt.clf()

            with col2:
                st.subheader("SHAP Explanation Output")
                st.caption("Total Data : {}".format(
                    len(x_data)))
                st.markdown(
                    "**Mean Base Value:** *{}*".format(np.mean(shap_output_many[i].base_values)))

                shap_overall_sum = pd.DataFrame(
                    {'Features': x_tab.columns, 'SHAP Value': np.sum(shap_output_many[i].values, axis=0)})
                shap_overall_sum = shap_overall_sum
                st.markdown("Overall **sum** of SHAP feature contributions: ")
                st.write(shap_overall_sum)

            st.subheader("SHAP Feature Dependency Option")
            st.caption(
                "SHAP Visual Explanation based on two feature interaction")

            col1, col2 = st.columns([1, 2])

            # get max SHAP value for each feature
            shap_max_value = np.max(shap_output_many[i].values, axis=0)

            with col1:
                st.write("Select a features to be plotted")
                with st.form(key="shap_feature_dependency_form_tab{}".format(i)):
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        main_feature = st.selectbox(
                            '1st Feature', [x for x in x_tab.columns], index=0, key="main_feature_dplot_tab{}".format(i))
                        st.caption(
                            "Main Feature to be plotted as the main y-axis")
                    with col_b:
                        second_feature = st.selectbox(
                            '2nd Feature', [x for x in x_tab.columns], index=int(np.argmax(shap_max_value)), key="second_feature_dplot_tab{}".format(i))
                        st.caption(
                            "Secondary or interaction feature to be plotted against main feature")

                    st.form_submit_button("Plot", type="primary")

            with col2:
                main_feature = getSession(
                    'main_feature_dplot_tab{}'.format(i))
                second_feature = getSession(
                    'second_feature_dplot_tab{}'.format(i))

                # shap.dependence_plot(ind=main_feature, interaction_index=second_feature,
                #                      shap_values=shap_output_many[i].values, features=x_tab, feature_names=x_tab.columns, show=False)
                st.pyplot(load_shap_dependence_plot(
                    main_feature, second_feature, shap_output_many[i].values, x_tab))
                plt.clf()

            st.subheader("SHAP Stacked Force Plot")
            st.caption(
                "Shap Multiple Visual Explanation based on the force of each features")

            col1, col2 = st.columns([1, 4])
            with col1:
                with st.form(key="shap_stacked_force_form_tab{}".format(i)):
                    st.write("Select number of random instance to be plotted")
                    qty_stacked_force_plot = st.number_input('Number of Instance', min_value=1, max_value=len(
                        x_tab)-1, value=len(x_tab)//10 if len(x_tab)//10 <= 100 else 100, key="qty_stacked_force_plot{}".format(i))
                    st.form_submit_button("Plot", type="primary")
            with col2:
                stacked_force_plot = load_shap_stacked_force_plot(
                    shap_output_many[i], tab_title, x_tab, qty_stacked_force_plot)
                components.html(stacked_force_plot, height=400)

                # st.write(load_shap_stacked_force_plot(
                #     shap_output_many[i], tab_title, x_tab, qty=qty_stacked_force_plot))
                # shap.save_html("plots/stacked_force_plot{}.html".format(
                #     i), stacked_force_plot)
                # html_plot = open("plots/stacked_force_plot{}.html".format(
                #     i), 'r', encoding='utf-8')

            # ################## #
            # SINGLE EXPLANATION #
            # ################## #
            st.subheader("Instance Explanation")
            st.caption("SHAP Visual Explanation based on single instance")

            d_col1, d_col2 = st.columns([2, 1])

            with d_col1:
                idx_instance = getSession("instance_idx_tab{}".format(i))
                idx_instance = 0 if idx_instance == False else idx_instance

                instance_tab = x_tab[idx_instance:idx_instance+1]
                explainer_tab = shap_explainers[i]

                # shap_output_instance = explainer(
                #     x_tab.iloc[idx_instance:idx_instance+1])

                shap_output_instance = load_shap_instance(
                    explainer_tab, instance_tab, tab_title, standard_scaler=tab_standard_scaler)

                waterfall_plot = load_shap_plots(
                    shap_output_instance, instance_tab, tab_title, "waterfall")
                st.pyplot(waterfall_plot)
                plt.clf()

            with d_col2:
                with st.form(key="instance_idx_form_tab{}".format(i)):
                    st.write("Pick an instance to be explained")
                    st.number_input('Instance Index',
                                    min_value=0, max_value=len(x_tab)-1, value=0, key="instance_idx_tab{}".format(i))
                    st.form_submit_button("Plot", type="primary")

                idx_instance = getSession(
                    "instance_idx_tab{}".format(i))
                st.write(
                    "Instance Data of ID: ***{}***".format(getSession("instance_idx_tab"+str(i))))

                st.write(x_data[idx_instance:idx_instance+1])

            idx_instance = getSession("instance_idx_tab{}".format(i))
            idx_instance = 0 if idx_instance == False else idx_instance

            force_plot = load_shap_plots(
                shap_output_instance[0], instance_tab, tab_title, "force")
            st.pyplot(force_plot)
            plt.clf()

            decision_plot = load_shap_plots(
                shap_output_instance, instance_tab, tab_title, "decision")
            st.pyplot(plt.gcf())
            plt.clf()

    # with st.spinner("Loading Dashboard..."):
    #     st.success("Data Initialized")
    #     time.sleep(5)
    #     st.success("Dashboard Loaded")
