import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import shap
from joblib import load
#st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# California House Price Prediction App

This app predicts the **California House Price**!
""")
st.write('---')

# Loads the California House Price Dataset
california = datasets.fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
Y = pd.DataFrame(california.target, columns=["Value"])

# Sidebar
st.sidebar.header('Specify Input Parameters')
MedInc = st.sidebar.slider('MedInc', float(X.MedInc.min()), float(X.MedInc.max()), float(X.MedInc.mean()))
HouseAge = st.sidebar.slider('HouseAge', float(X.HouseAge.min()), float(X.HouseAge.max()), float(X.HouseAge.mean()))
AveRooms = st.sidebar.slider('AveRooms', float(X.AveRooms.min()), float(X.AveRooms.max()), float(X.AveRooms.mean()))
AveBedrms = st.sidebar.slider('AveBedrms', float(X.AveBedrms.min()), float(X.AveBedrms.max()), float(X.AveBedrms.mean()))
Population = st.sidebar.slider('Population', float(X.Population.min()), float(X.Population.max()), float(X.Population.mean()))
AveOccup = st.sidebar.slider('AveOccup', float(X.AveOccup.min()), float(X.AveOccup.max()), float(X.AveOccup.mean()))
Latitude = st.sidebar.slider('Latitude', float(X.Latitude.min()), float(X.Latitude.max()), float(X.Latitude.mean()))
Longitude = st.sidebar.slider('Longitude', float(X.Longitude.min()), float(X.Longitude.max()), float(X.Longitude.mean()))

data = {'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
        }
features = pd.DataFrame(data, index=[0])

# Main Panel
st.header('Specified Input parameters')
st.write(features)
st.write('---')

# Load the pre-trained model
model = load('model.joblib')

# Apply the model to make predictions
prediction = model.predict(features)
st.header('Prediction of Value')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, features)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, features, plot_type="bar")
st.pyplot(bbox_inches='tight')