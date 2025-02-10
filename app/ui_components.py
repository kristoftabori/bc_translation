import streamlit as st
from app.config import MAX_CORRECTION_ITERATION


def setup_sidebar():
    with st.sidebar as my_sidebar:
        st.text_input(
            "Enter Your OpenAI API key below",
            type="password",
            key="openai_api_key"
        )
        with st.expander("Models"):
            openai_model_name = st.radio(
                "Which model to use?",
                [
                    "gpt-4o",
                    "gpt-4o-mini"
                ],
                key="openai_model_name",
            )
        with st.expander("Agent Parameters"):
            max_correction_runs = st.number_input(
                "Max Correction Runs",
                min_value=0,
                max_value=10,
                value=MAX_CORRECTION_ITERATION
            )
    return my_sidebar, openai_model_name, max_correction_runs


def show_translation_status(message, state="running"):
    return st.status(message, expanded=True, state=state)


def display_results(results):
    for result in results:
        st.code(result)
