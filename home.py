import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

from utils import saveSession, getSession
from sklearn.model_selection import train_test_split
from st_pages import show_pages_from_config

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import shap
import matplotlib.pyplot as plt

show_pages_from_config()

st.set_option('deprecation.showPyplotGlobalUse', False)

# warnings.filterwarnings('ignore')

st.title("Home Page")
# st.markdown(
#     "<style>.appview-container .main .block-container{padding-left:1.5rem; padding-right:1.5rem}</style>", unsafe_allow_html=True)
st.markdown("Welcome to the **dashboard**")

st.subheader("Upload Dataset **:red[*]**", divider="grey")
uploaded_file = st.file_uploader(
    "Choose a csv file", type="csv", on_change=lambda: st.session_state.clear())

# First Initialization when project run
if 'clicked' not in st.session_state:
    st.session_state["clicked"] = {"confirmTarget": False, "initData": False, "initModel": False, "confirmDropped": False,
                                   "click_generate_instance_tab0": True,
                                   "click_generate_instance_tab1": True,
                                   "click_generate_instance_tab2": True}


def clicked(button):
    if (button == "initModel"):
        saveSession({"confirmProgressInit": True})
    st.session_state.clicked[button] = True


# LinReg, KNN, SVR
loadModelState = [False, False, False]


def updateLoadState(idx: int):
    loadModelState[idx] = True

    # if there are still FALSE in load data, return false
    if not all(loadModelState):
        return False

    saveSession({"confirmInit": True, "confirmProgressInit": False})
    return True


@st.cache_resource(show_spinner=False)
def loadModel(_model, x_train, y_train, category_columns, x_columns, modelName):

    # boolean_columns = x_train.columns[x_train.dtypes == 'bool'].tolist()

    x_train = x_train.values

    # convert into category column type
    # for col in category_columns:
    #     data_csv[col] = data_csv[col].astype('category')

    indices_category_columns = np.where(
        np.isin(x_columns, category_columns))[0]

    # indices_boolean_columns = np.where(np.isin(x_columns, boolean_columns))[0]

    pipeline = getPipeline(_model, indices_category_columns)

    pipeline.fit(x_train, y_train)

    return pipeline


def getPipeline(_model, indices_category_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), indices_category_columns),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(
        steps=[('preprocessor', preprocessor), ('model', _model)])

    return pipeline


def testModel(x_test, y_test, pipeline):
    x_test = x_test.values
    y_pred = pipeline.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,
                  y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    plt.clf()
    plt.scatter(y_test, y_pred, color='steelblue',
                label='Actual vs. Predicted', alpha=0.4)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             linestyle='--', color='red', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values ({})'.format(
        type(pipeline.named_steps['model']).__name__))
    plt.legend()

    plt.gcf().text(0.92, 0.85, "MSE : {}".format(mse), fontsize=8)
    plt.gcf().text(0.92, 0.81, "MAE : {}".format(mae), fontsize=8)
    plt.gcf().text(0.92, 0.77, "R2 : {}".format(r2), fontsize=8)

    return mae, mse, r2, plt.gcf()


def getKNN_k(x_train, y_train, x_test, y_test, x_columns, category_columns, min_iter=2, max_iter=15):

    k_neighbor_score = []

    for k in range(min_iter, max_iter):
        knn_model = KNeighborsRegressor(n_neighbors=k)

        pipeline = loadModel(knn_model, x_train, y_train,
                             category_columns, x_columns, modelName="KNN"+str(k))

        score_knn = pipeline.score(x_test.values, y_test)
        k_neighbor_score.append({k: score_knn})

    max_dict = max(k_neighbor_score, key=lambda x: list(x.values())[0])
    k_neighbor = list(max_dict.keys())[0]

    return k_neighbor, k_neighbor_score


@st.cache_data(show_spinner=False)
def readData(uploaded_file):
    return pd.read_csv(uploaded_file)


if (uploaded_file):
    data_csv = readData(uploaded_file)
    st.session_state["data_csv"] = data_csv
    st.session_state["uploaded_file"] = uploaded_file.name

    # st.subheader("Data Information", divider="grey")
    st.subheader("Data Information", divider="grey")
    st.caption(
        "Display Dataset General Information such as Shape, Columns, Describe, etc")

    with st.expander("Show Data Information"):
        st.write("Data Shape:", data_csv.shape)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Data Columns:", data_csv.columns)

        with col2:
            st.write("Data Types:", data_csv.dtypes)

        with col3:
            st.write("Data Missing Values:", data_csv.isnull().sum())

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Describe")
            st.write(data_csv.describe())

        with col2:
            st.subheader("Data Header")
            st.write(data_csv.head())

        st.subheader("Data Correlation")
        use_annotation = st.toggle(
            "Use Annotation", value=False, key="use_annotation")
        st.pyplot(sns.heatmap(data_csv.drop(data_csv.select_dtypes(
            include=['object']).columns, axis=1).corr(), annot=use_annotation).figure)

        st.markdown(''' <a target="_self" href="#data-information">
                        Back to Top
                </a>''', unsafe_allow_html=True)
        st.write("")

    st.subheader("Feature Selection :red[*]", divider="grey")

    sbDropped = []
    dataColumns = [data_csv.columns[idx]
                   for idx in range(len(data_csv.columns))]

    with st.form("form_dropped_features"):
        st.caption(
            "Select features to be dropped in the dataset")
        sbDropped = st.multiselect(
            'Dropped Features', dataColumns, key="dropped_features")
        st.form_submit_button("Proceed", type="primary",
                              on_click=clicked, args=["confirmDropped"])
    if (len(sbDropped) > len(dataColumns)-2):
        st.caption(
            ":red[**WARNING!**] *At least 2 or more feature should be in the data!*")

    if getSession("clicked")["confirmDropped"] and len(sbDropped) <= len(dataColumns)-2:
        columnOptions = [
            col for col in data_csv.columns if col not in sbDropped]

        with st.form(key="form_feature_selection"):
            col1, col2 = st.columns(2)
            with col1:
                st.caption(
                    "Select target feature to be predicted in the dataset")

                sbTarget = st.selectbox(
                    'Feature Name', columnOptions, index=len(columnOptions)-1, key="target_feature", )

            with col2:
                st.caption(
                    "Select all category features in the dataset")
                sbCategories = st.multiselect(
                    'Categories Feature', columnOptions, key="category_features")

            # BUTTON PROCEED
            st.form_submit_button("Proceed", type="primary",
                                  on_click=clicked, args=["confirmTarget"])

    # ####################  #
    # DATA INITIALIZATION   #
    # ####################  #
    if getSession("clicked")["confirmTarget"]:
        # saveSession({"target_feature": sbTarget})
        # saveSession({"category_feature": sbCategories})

        # Begin Initialization
        st.subheader("Data Initialization :red[*]", divider="grey")

        with st.form(key="form_data_initialization"):

            col_init1, col_init2, col_init3 = st.columns(3)

            with col_init1:
                st.number_input(
                    'Test Size Ration', key="test_size", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
                st.caption(
                    "Test and Train Size Ratio to be Divided for Train and Testing")

                default_subsampling_value = 0
                if (len(data_csv) > 10000):
                    default_subsampling_value = 8000

                sbSubsampling = st.number_input(
                    'Subsampling', key="num_subsampling", min_value=0, max_value=len(data_csv), value=default_subsampling_value, step=1)
                st.caption(
                    "Subsampling data for faster processing (optional). **Leave at 0 for no subsampling**")

            with col_init2:
                sbModeBackground = st.selectbox(
                    'Select background dataset mode', ('K Sample', 'K Means'), key="background_mode")
                st.caption(
                    ":red[**WARNING!**] *KMeans* only work with numerical data only!")

                st.number_input(
                    'K Value', key="k_value", min_value=1, value=2, step=1)
                st.caption(
                    "Used for getting K data for SHAP Background Dataset using selected method")

            with col_init3:
                st.number_input(
                    'Random State', key='random_state', min_value=0, max_value=100, value=42, step=1)
                st.caption("State of Randomness for Reproducibility")

            st.form_submit_button(
                "Proceed", type="primary", on_click=clicked, args=["initData"])

        if (sbCategories != [] and sbModeBackground == "K Means"):
            st.caption(
                ":red[**There are category feature present. Background dataset mode need to be set to K sample!**]")

        # #################### #
        # MODEL INITIALIZATION
        # #################### #

    if getSession("clicked")["initData"]:
        st.subheader("Model Initialization :red[*]", divider="grey")

        with st.form(key="form_model_initialization"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("#### Linear Regression")
                linreg_fit_intercept = st.checkbox(
                    "Fit Intercept", value=True, key="linreg_fit_intercept")
                st.caption(
                    "Whether to calculate the intercept for this model.")

            with col2:
                st.write("#### KNN")
                knn_auto_n = False
                knn_inp_k = st.number_input(
                    'K Neighbor', key="knn_inp_k", min_value=2, value=5)
                st.caption("*or*")
                knn_auto_n = knn_auto_n = st.checkbox(
                    "Auto K", value=True, key="knn_auto_n")
                st.caption(
                    "Automatically find the most optimal N for KNN model using KNN Regressor")

            with col3:
                rf_max_features_option = ["None", "sqrt", "log2"]
                st.write("#### Random Forest")
                rf_max_feature = st.selectbox("Max Features", rf_max_features_option,
                                              key="rf_max_features_option", index=1)
                st.caption(
                    "The number of features to consider when looking for best split")

                rf_n_estimator = st.number_input(
                    'Number of Estimators', key="rf_n_estimators", min_value=10, value=100)
                st.caption("The number of trees in the forest")

            st.subheader("", divider='grey')

            col1, col2, col3 = st.columns(3)
            with col1:
                linreg_standard_scaler = st.checkbox(
                    "Standard Scaler", value=False, key="linreg_standard_scaler")
                st.caption(
                    "Whether to train linear regression with standard scaled **x.**")
            with col2:
                knn_standard_scaler = st.checkbox(
                    "Standard Scaler", value=False, key="knn_standard_scaler")
                st.caption(
                    "Whether to train KNN with standard scaled **x.**")
            with col3:
                rf_standard_scaler = st.checkbox(
                    "Standard Scaler", value=False, key="rf_standard_scaler")
                st.caption(
                    "Whether to train random forest regressor with standard scaled **x.**")

            col1, col2, a, b = st.columns(4)

            st.form_submit_button(
                "Begin Initialization", type="primary", on_click=clicked, args=["initModel"])

    if getSession("clicked")["initModel"]:
        with st.spinner('Initializing the Data, please wait...'):

            # get variables from session
            num_subsampling = getSession("num_subsampling")
            background_mode = getSession("background_mode")
            k_value = getSession("k_value")
            random_state = getSession("random_state")
            target_feature = sbTarget
            category_features = sbCategories

            dropped_features = sbDropped
            test_size = getSession("test_size")

            # drop unused features
            data_csv = data_csv.drop(dropped_features, axis=1)

            # subsampling the dataset
            if (num_subsampling != 0):
                data_csv = shap.sample(
                    data_csv, num_subsampling, random_state=random_state)
                data_csv.reset_index(drop=True, inplace=True)

            # convert into category column type
            for col in category_features:
                mapping_dict = {
                    value: idx+1 for idx, value in enumerate(data_csv[col].squeeze().unique())}
                data_csv[col] = data_csv[col].map(mapping_dict)
                data_csv[col] = data_csv[col].astype('category')

            # get x, y(target)
            x = data_csv.drop(sbTarget, axis=1)
            y = data_csv[sbTarget]

            numeric_features = [
                col for col in x.columns if col not in category_features]

            saveSession({"numeric_features": numeric_features,
                        "data_csv": data_csv, })

            # makes the first column to be category column, then rest for numeric
            x = pd.concat([x[category_features], x[numeric_features]], axis=1)

            # train test split
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=random_state)

            # reset index
            x_train.reset_index(drop=True, inplace=True)
            x_test.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)

            # Split into category and numeric column for merging purpose
            x_train_category = x_train[category_features]
            x_test_category = x_test[category_features]

            x_train_numeric = x_train[numeric_features]
            x_test_numeric = x_test[numeric_features]

            models_standard_scaler = [
                linreg_standard_scaler, knn_standard_scaler, rf_standard_scaler]

            # standard scaler (category and numeric)
            if (any(models_standard_scaler)):

                # fit standard scaler
                scaler = StandardScaler()
                scaler.fit(x_train_numeric)

                # also transform x
                x_scaled = pd.concat([x[category_features], pd.DataFrame(
                    scaler.transform(x[numeric_features]), columns=numeric_features)], axis=1)

                # transform standard scaler
                x_train_numeric_scaled = pd.DataFrame(scaler.transform(
                    x_train_numeric), columns=x_train_numeric.columns)
                x_test_numeric_scaled = pd.DataFrame(scaler.transform(
                    x_test_numeric), columns=x_test_numeric.columns)

                # merge back
                x_train_scaled = pd.concat(
                    [x_train_category, x_train_numeric_scaled], axis=1)

                x_test_scaled = pd.concat(
                    [x_test_category, x_test_numeric_scaled], axis=1)

                # saves into session
                saveSession({"x_train_scaled": x_train_scaled,
                            "x_test_scaled": x_test_scaled, "x_scaled": x_scaled,
                             "scaler": scaler,
                             'x_data_scaled': x_scaled,
                             })

            # calculate background datasets
            background_datasets = []
            for idx, scaled in enumerate(models_standard_scaler):
                x_background = x_scaled if scaled else x
                if category_features == []:
                    background_datasets.append(
                        shap.kmeans(x_background, k_value).data)
                else:
                    background_datasets.append(
                        shap.sample(x_background, k_value, random_state=random_state))

            # saves into session
            saveSession({"x_train": x_train, "x_test": x_test,
                        "y_train": y_train, "y_test": y_test,
                         "background_datasets": background_datasets,
                         "x_data": x, "y_data": y,
                         "models_standard_scaler": models_standard_scaler, })

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Linear Regression", divider='grey')

            with st.spinner('Getting Linear Regression Model Ready, Please Wait...'):

                linreg_x_train = x_train_scaled if linreg_standard_scaler else x_train
                linreg_x_test = x_test_scaled if linreg_standard_scaler else x_test

                pipeline_linear_regression = loadModel(
                    LinearRegression(fit_intercept=linreg_fit_intercept), linreg_x_train, y_train, category_features, list(x.columns), modelName="Linear Regression")

                mae, mse, r2, plot_linear_regression = testModel(
                    linreg_x_test, y_test, pipeline_linear_regression)

                # save model
                saveSession(
                    {"pipeline_linear_regression": pipeline_linear_regression})

            updateLoadState(0)

            with st.expander("Show Model Information"):
                st.markdown("**MSE :** *{:.3f}*".format(mse))
                st.markdown("**MAE :** *{:.3f}*".format(mae))
                st.markdown("**r2 :** *{:.3f}*".format(r2))
                st.pyplot(plot_linear_regression)
            st.markdown(
                "*Linear Regression Model Initialization Complete*")

        with col2:
            st.subheader("KNN", divider="grey")
            with st.spinner('Getting KNN Model Ready, Please Wait...'):

                knn_x_train = x_train_scaled if knn_standard_scaler else x_train
                knn_x_test = x_test_scaled if knn_standard_scaler else x_test

                k_neighbor = knn_inp_k

                if (knn_auto_n):
                    k_neighbor, k_neighbor_score = getKNN_k(
                        knn_x_train, y_train, knn_x_test,  y_test, list(x.columns), category_features)

                pipeline_knn = loadModel(
                    KNeighborsRegressor(n_neighbors=k_neighbor), knn_x_train, y_train, category_features, list(x.columns), modelName="KNN")

                mae, mse, r2, plot_knn = testModel(
                    knn_x_test, y_test, pipeline_knn)

                # save model
                saveSession(
                    {"pipeline_knn": pipeline_knn})

            updateLoadState(1)

            with st.expander("Show Model Information"):
                st.markdown("**Best K :** *{}*".format(k_neighbor))
                st.markdown("**MSE :** *{:.3f}*".format(mse))
                st.markdown("**MAE :** *{:.3f}*".format(mae))
                st.markdown("**r2 :** *{:.3f}*".format(r2))
                st.pyplot(plot_knn)
            st.markdown(
                "*KNN Model Initialization Complete*")

        with col3:
            st.subheader("Random Forest", divider="grey")
            with st.spinner('Getting Random Forest Model Ready, Please Wait...'):

                rf_x_train = x_train_scaled if rf_standard_scaler else x_train
                rf_x_test = x_test_scaled if rf_standard_scaler else x_test
                st.cache_data.clear()
                st.cache_resource.clear()

                pipeline_rf = loadModel(
                    RandomForestRegressor(max_features=1 if rf_max_feature == "None" else rf_max_feature, n_estimators=rf_n_estimator, random_state=random_state), rf_x_train, y_train, category_features, list(x.columns), modelName="Random Forest")

                mae, mse, r2, plot_rf = testModel(
                    rf_x_test, y_test, pipeline_rf)

                # save model
                saveSession(
                    {"pipeline_rf": pipeline_rf})

            updateLoadState(2)

            with st.expander("Show Model Information"):
                st.markdown("**MSE :** *{:.3f}*".format(mse))
                st.markdown("**MAE :** *{:.3f}*".format(mae))
                st.markdown("**r2 :** *{:.3f}*".format(r2))
                st.pyplot(plot_rf)
            st.markdown(
                "*Random Forest Model Initialization Complete*")


def resetEverything():
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()


if (getSession("confirmInit")):
    st.sidebar.success(
        "Inizialization completed, please select the dashboard page above ")
elif getSession("confirmProgressInit"):
    st.sidebar.warning("Initialization in progress please wait")
else:
    st.sidebar.error("Please begin the initialization first")

st.sidebar.button("Reset Everything", type="secondary",
                  key="sidebar_reset", on_click=resetEverything)
