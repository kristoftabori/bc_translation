import streamlit as st


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
    return my_sidebar, openai_model_name


def show_translation_status(message, state="running"):
    return st.status(message, expanded=True, state=state)


def display_results(results):
    for result in results:
        st.code(result)
