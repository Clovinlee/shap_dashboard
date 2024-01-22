# Chrisanto's SHAP Dashboard
*made using from scratch python, streamlit, and various other library*

## Project Background
This project is part of my university thesis on the *Study of explainable artificial intelligence analysis using the model-agnostic SHAP method for regression*. The project takes a CSV dataset to display multiple SHAP visualizations as the final output for a regression problem, utilizing the SHAP kernel explainer. Users can modify certain parameters of the model and SHAP to observe how these changes affect the output. Currently, the models used are limited to three: Linear Regression, K Nearest Neighbor, and Random Forest Regressor.

## How to Install
```python
pip install streamlit
pip install shap
```

## How to use
Simply run this in the project folder terminal
```python
streamlit run home.py
```

## What's next
- D̶a̶t̶a̶ ̶I̶n̶f̶o̶r̶m̶a̶t̶i̶o̶n̶,̶ ̶a̶d̶d̶ ̶D̶F̶ ̶c̶o̶r̶r̶e̶l̶a̶t̶i̶o̶n̶ ̶h̶e̶a̶t̶m̶a̶p̶ ̶s̶n̶s̶
- S̶e̶l̶e̶c̶t̶ ̶c̶a̶t̶e̶g̶o̶r̶y̶ ̶f̶e̶a̶t̶u̶r̶e̶
- A̶d̶d̶ ̶o̶p̶t̶i̶o̶n̶ ̶f̶o̶r̶ ̶s̶t̶a̶n̶d̶a̶r̶d̶ ̶s̶c̶a̶l̶e̶r̶ ̶d̶a̶t̶a̶
- ̶A̶d̶d̶ ̶o̶p̶t̶i̶o̶n̶ ̶t̶o̶ ̶l̶i̶m̶i̶t̶ ̶d̶a̶t̶a̶ ̶s̶a̶m̶p̶l̶e̶
- ̶I̶g̶n̶o̶r̶e̶ ̶s̶e̶l̶e̶c̶t̶e̶d̶ ̶c̶o̶l̶u̶m̶n̶
- ̶O̶p̶t̶i̶o̶n̶ ̶f̶o̶r̶ ̶r̶a̶n̶d̶o̶m̶ ̶s̶a̶m̶p̶l̶e̶ ̶a̶s̶ ̶b̶a̶c̶k̶g̶r̶o̶u̶n̶d̶ ̶d̶a̶t̶a̶s̶e̶t̶ ̶
- For Standard scaler, also add map to original value at the end
