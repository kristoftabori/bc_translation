import streamlit as st
from openai import APIConnectionError, OpenAIError
from app.batch_processing import translate_dataframe, translate_row
import os
from config import LANGUAGES
from app.ui_components import show_translation_status, setup_sidebar, display_results
from app.langchain_integrations import build_correction_chain, build_correction_graph, build_translation_chain
import pandas as pd
from evals import placeholder_identity
from PIL import Image
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))

im = Image.open("app/assets/g_translate_ic_icon.png")
st.set_page_config(
    page_title="SafeTranslate",
    page_icon=im,
    layout="wide",
)


def setup_environment():
    api_key = setup_sidebar()
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    return bool(api_key)


def individual():
    with st.form(key='translation_form'):
        languages = st.multiselect(
            "In what languages shall I translate the text?",
            LANGUAGES,
            placeholder="Choose the target languages...",
            key='language_select'
        )
        txt = st.text_area(
            "Put the text here to get translated",
            placeholder="Write here something I shall translate...",
            key='text_input'
        )

        submit_button = st.form_submit_button(label='Translate', type='primary')

    if submit_button:
        if not (txt and languages):
            return

        if not st.session_state.openai_api_key:
            st.toast("You need to set your API key in the sidebar!", icon="⚠️")
            return

        try:
            with show_translation_status("Translating...") as status:
                st.write("Extracting placeholders...")
                st.write("Translate to languages...")
                translation_chain = build_translation_chain(st.session_state.openai_model_name)
                correction_chain = build_correction_chain(st.session_state.openai_model_name)
                processing_graph = build_correction_graph(st.session_state.MAX_CORRECTION_ITERATION)

                results = []
                for language in languages:
                    st.write(f"Translation into {language} in progress...")
                    result = translate_row(
                        txt,
                        language,
                        translation_chain,
                        correction_chain,
                        processing_graph
                    )
                    assert len(result) > 1
                    results.append(result)

                display_results(results)
                status.update(label="Translation complete!", state="complete")
                st.toast("Everything translated!", icon="🎉")

        except APIConnectionError as e:
            status.update(label="Wrong API key entered!", state="error")
            st.error(e.message, icon="🚨")
        except OpenAIError as e:
            status.update(label="Invalid API key!", state="error")
            st.error(e.message, icon="🚨")
        except Exception as e:
            status.update(label="Translation Failed!", state="error")
            st.error(str(e), icon="🚨")


def highlight_row(row):
    styles = []
    primary_color = st.get_option("theme.primaryColor")
    r = int(primary_color[1:3], 16)
    g = int(primary_color[3:5], 16)
    b = int(primary_color[5:7], 16)
    primary_color_rgba = f"rgba({r}, {g}, {b}, 0.25)"

    for col in row.index:
        if placeholder_identity(row["English"], row[col]) == 0.0:
            styles.append(f"background-color: {primary_color_rgba};")
        else:
            styles.append("")
    return styles


def batch():
    with st.form(key='batch_translation_form'):
        batch_languages = st.multiselect(
            "What languages shall I translate the text?",
            LANGUAGES,
            placeholder="Choose the target languages...",
            key='batch_language_select'
        )
        uploaded_file = st.file_uploader("Choose a CSV to upload")
        upload_submit_button = st.form_submit_button(label='Translate', type='primary')
    if upload_submit_button:
        if not (uploaded_file and batch_languages):
            return
        if not st.session_state.openai_api_key:
            st.toast("You need to set your API key in the sidebar!", icon="⚠️")
            return

        try:
            with show_translation_status("Translating...") as batch_status:
                st.write("Translate to languages...")
                translation_chain = build_translation_chain(st.session_state.openai_model_name)
                correction_chain = build_correction_chain(st.session_state.openai_model_name)
                processing_graph = build_correction_graph(st.session_state.MAX_CORRECTION_ITERATION)
                uploaded_df = pd.read_csv(uploaded_file, header=None)
                results = []
                for language in batch_languages:
                    st.write(f"Translating into {language} in progress...")
                    result = translate_dataframe(
                        uploaded_df,
                        [language],
                        translation_chain,
                        correction_chain,
                        processing_graph
                    )
                    results.append(result[language])
                st.write("stitching together final DataFrame...")
                final_df = pd.concat(
                    [
                        pd.DataFrame({"English": uploaded_df.iloc[:, 0]}),
                        *results
                    ],
                    axis=1
                )
                styled_final_df = final_df.style.apply(highlight_row, axis=1)
                st.dataframe(styled_final_df, hide_index=True)
                batch_status.update(label="Translation complete!", state="complete")
                st.toast("Everything translated!", icon="🎉")

        except APIConnectionError as e:
            batch_status.update(label="Wrong API key entered!", state="error")
            st.error(e.message, icon="🚨")
        except OpenAIError as e:
            batch_status.update(label="Invalid API key!", state="error")
            st.error(e.message, icon="🚨")
        except Exception as e:
            batch_status.update(label="Translation Failed!", state="error")
            st.error(str(e) + "1", icon="🚨")


def main():
    st.title("Translation with Placeholders")
    st.text(
        "This app will translate the text in the textbox "
        "to the target languages respecting and not touching the placeholders."
    )
    _, _, max_correction_runs = setup_sidebar()
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    individual_tab, batch_tab = st.tabs(["Interactive", "CSV"])

    if "MAX_CORRECTION_ITERATION" not in st.session_state:
        st.session_state["MAX_CORRECTION_ITERATION"] = max_correction_runs

    if max_correction_runs != st.session_state["MAX_CORRECTION_ITERATION"]:
        st.session_state["MAX_CORRECTION_ITERATION"] = max_correction_runs

    with individual_tab:
        individual()
    with batch_tab:
        batch()


if __name__ == "__main__":
    main()
