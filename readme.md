# Chrisanto's SHAP Dashboard
*made using from scratch python, streamlit, and various other library*

## Project Background
This project is part of my university thesis on the *Study of explainable artificial intelligence analysis using the model-agnostic SHAP method for regression*. The project takes a CSV dataset to display multiple SHAP visualizations as the final output for a regression problem, utilizing the SHAP kernel explainer. Users can modify certain parameters of the model and SHAP to observe how these changes affect the output. Currently, the models used are limited to three: Linear Regression, K Nearest Neighbor, and Random Forest Regressor.

## How to Install
```python
pip install streamlit
pip install shap
```

## How to run
Simply run this in the project folder terminal
```python
streamlit run home.py
```

## What's next
- D̶a̶t̶a̶ ̶I̶n̶f̶o̶r̶m̶a̶t̶i̶o̶n̶,̶ ̶a̶d̶d̶ ̶D̶F̶ ̶c̶o̶r̶r̶e̶l̶a̶t̶i̶o̶n̶ ̶h̶e̶a̶t̶m̶a̶p̶ ̶s̶n̶s̶
- ~~Select droppable feature~~
- ~~Select category feature~~
- ~~Add option for standard scaler data~~ 
- ~~Add Option to limit data sample~~
- ~~Ignore Selected Column~~
- ~~Option for random sample for background dataset~~
- ~~For Standard scaler, also add map to original value at the end~~
