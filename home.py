import streamlit as st
import pandas as pd
from utils import saveSession, getSession
from sklearn.model_selection import train_test_split
from st_pages import show_pages_from_config

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from itertools import permutations
from itertools import combinations

import shap

show_pages_from_config()

st.set_option('deprecation.showPyplotGlobalUse', False)

# warnings.filterwarnings('ignore')

st.title("Home Page")
# st.markdown(
#     "<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)
st.markdown("Welcome to the **dashboard**")

st.subheader("Upload Dataset", divider="grey")
uploaded_file = st.file_uploader("Choose a csv file", type="csv")

if 'clicked' not in st.session_state:
    st.session_state["clicked"] = {"confirmTarget": False, "initData": False,
                                   "click_generate_instance_tab0": True,
                                   "click_generate_instance_tab1": True,
                                   "click_generate_instance_tab2": True}


def clicked(button):
    if (button == "initData"):
        saveSession({"confirmProgressInit": True})
    st.session_state.clicked[button] = True


def updateLoadState(idx: int, loadModelState: list):
    loadModelState[idx] = True
    for i in loadModelState:
        if (not i):
            return False
    saveSession({"confirmInit": True, "confirmProgressInit": False})
    return True


@st.cache_resource(show_spinner=False)
def loadModelLinear(x_train, y_train, x_test, y_test):
    model_linear_regression = LinearRegression()
    model_linear_regression.fit(x_train, y_train)
    y_pred = model_linear_regression.predict(x_test)

    # score error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model_linear_regression, mse, r2


@st.cache_resource(show_spinner=False)
def loadModelKNN(x_train, y_train, x_test, y_test):
    max_iter = len(y_test)
    k_neighbor_score = []

    for k in range(1, max_iter):
        knn_model = KNeighborsRegressor(n_neighbors=k)
        knn_model.fit(x_train, y_train)

        score_knn = knn_model.score(x_test, y_test)
        k_neighbor_score.append({k: score_knn})

    max_k_neighbor = max(
        k_neighbor_score, key=lambda x: list(x.values())[0])
    k_neighbor = list(max_k_neighbor.keys())[0]

    knn_model = KNeighborsRegressor(n_neighbors=k_neighbor)
    knn_model.fit(x_train, y_train)

    y_pred = knn_model.predict(x_test)

    # Score error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return knn_model, k_neighbor, mse, r2


@st.cache_resource(show_spinner=False)
def loadModelSVR(x_train, y_train, x_test, y_test):

    svr_model = SVR()
    svr_model.fit(x_train, y_train)

    y_pred = svr_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return svr_model, mse, r2


# LinReg, KNN, SVR
loadModelState = [False, False, False]


@st.cache_data(show_spinner=False)
def readData(uploaded_file):
    return pd.read_csv(uploaded_file)


if (uploaded_file):
    data_csv = readData(uploaded_file)
    st.session_state["data_csv"] = data_csv
    st.session_state["uploaded_file"] = uploaded_file.name

    st.subheader("Data Information", divider="grey")
    st.caption(
        "Display Dataset General Information such as Shape, Columns, Describe, etc")

    with st.expander("Show Data Information"):
        st.write("Data Shape:", data_csv.shape)
        st.write("Data Columns:", data_csv.columns)
        st.write("Data Types:", data_csv.dtypes)
        st.write("Data Missing Values:", data_csv.isnull().sum())
        st.subheader("Data Describe")
        st.write(data_csv.describe())
        st.subheader("Data Header")
        st.write(data_csv.head())

    st.subheader("Target Feature Selection :red[*]", divider="grey")
    st.caption(
        "Select target feature to be predicted in the dataset")
    sbTarget = st.selectbox(
        'Feature Names', [x for x in data_csv.columns], index=len(data_csv.columns)-1)
    st.markdown('Target Feature: ***{}***'.format(sbTarget))

    # BUTTON PROCEED
    st.button("Proceed", type="primary",
              on_click=clicked, args=["confirmTarget"])

    if getSession("clicked")["confirmTarget"]:
        saveSession({"target_feature": sbTarget})

        # Begin Initialization
        st.subheader("Data Initialization :red[*]", divider="grey")

        x = data_csv.drop(sbTarget, axis=1)
        y = data_csv[sbTarget]

        col_init1, col_init2, col_init3 = st.columns(3)

        with col_init1:
            st.number_input(
                'Test Size Ration', key="test_size", min_value=0.1, max_value=0.9, value=0.2, step=0.1)
            st.caption(
                "Test and Train Size Ratio to be Divided for Train and Testing")
        with col_init2:
            st.number_input(
                'Random State', key='random_state', min_value=0, max_value=100, value=42, step=1)
            st.caption("State of Randomness for Reproducibility")
        with col_init3:
            st.number_input(
                'K Means', key="k_means", min_value=1, value=2, step=1)
            st.caption(
                "Used for Clustering into K data for SHAP Background Dataset")

        col1, col2, a, b = st.columns(4)

        with col1:
            st.button(
                "Begin Initialization", type="primary", on_click=clicked, args=["initData"])
        with col2:
            st.button("Reset", on_click=lambda: st.session_state.clear())

        if getSession("clicked")["initData"]:

            with st.spinner('Initializing the Data, please wait...'):
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=getSession("test_size"), random_state=getSession("random_state"))
                kmeans = shap.kmeans(x, getSession("k_means"))

                saveSession({"x_train": x_train, "x_test": x_test,
                            "y_train": y_train, "y_test": y_test, "kmeans": kmeans})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Linear Regression", divider='grey')

                with st.spinner('Getting Linear Regression Model Ready, Please Wait...'):

                    model_linear_regression, mse, r2 = loadModelLinear(
                        x_train, y_train, x_test, y_test)

                    # save model
                    saveSession(
                        {"model_linear_regression": model_linear_regression})

                updateLoadState(0, loadModelState)

                with st.expander("Show Model Information"):
                    st.markdown("**MSE :** *{}*".format(mse))
                    st.markdown("**r2 :** *{}*".format(r2))
                st.markdown(
                    "*Linear Regression Model Initialization Complete*")

            with col2:
                st.subheader("KNN", divider="grey")
                with st.spinner('Getting KNN Model Ready, Please Wait...'):

                    knn_model, k_neighbor, mse, r2 = loadModelKNN(
                        x_train, y_train, x_test, y_test)

                    # save model
                    saveSession({"model_knn": knn_model})

                updateLoadState(1, loadModelState)

                with st.expander("Show Model Information"):
                    st.markdown("**Best K :** *{}*".format(k_neighbor))
                    st.markdown("**MSE :** *{}*".format(mse))
                    st.markdown("**r2 :** *{}*".format(r2))
                st.markdown("*KNN Model Initialization Complete*")

            with col3:
                st.subheader("SVR", divider="grey")
                with st.spinner('Getting SVR Model Ready, Please Wait...'):

                    svr_model, mse, r2 = loadModelSVR(
                        x_train, y_train, x_test, y_test)

                    saveSession({"model_svr": svr_model})

                updateLoadState(2, loadModelState)

                with st.expander("Show Model Information"):
                    st.markdown("**MSE :** *{}*".format(mse))
                    st.markdown("**r2 :** *{}*".format(r2))
                st.markdown(
                    "*SVR Model Initialization Complete*")


# sidebar
if (getSession("confirmInit")):
    st.sidebar.success(
        "Inizialization completed, please select the dashboard page above ")
    st.sidebar.button("Reset Everything", type="secondary",
                      key="sidebar_reset", on_click=lambda: st.session_state.clear())
elif (getSession("confirmProgressInit")):
    st.sidebar.warning("Data Initialization in Progress")
else:
    st.sidebar.error("Please begin the initialization first")
# sidebar end
