import streamlit as st


def saveSession(d: dict):
    for k, v in d.items():
        st.session_state[k] = v


def getSession(keys):
    dictReturn = {}

    if (type(keys) == str):
        if (keys in st.session_state):
            return st.session_state[keys]
        else:
            return False

    for key in keys:
        if (key in st.session_state):
            dictReturn[key] = st.session_state[key]
    if (dictReturn == {}):
        return False
    return dictReturn
