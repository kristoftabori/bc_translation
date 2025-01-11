import streamlit as st
import time
from openai import APIConnectionError, OpenAIError
from placeholder_parsing import extract_placeholders
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

st.title("Translation with Placeholders")
st.text("This is app will translate the text in the textbox to the target languages respecting and not touching the placeholders.")

with st.sidebar:
    st.text_input("Enter Your OpenAI API key below", type="password", key="openai_api_key")
os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

languages = st.multiselect(
    "What languages shall I translate the text?",
    [
        "Hungarian",
        "Spanish",
        "French",
        "German",
        "Japanese",
        "Arabic",
        "Hindi",
        "Portugese",
    ]
)
txt = st.text_area(
    "Put the text here to get translated",
    placeholder="Write here something I shall translate...",
)
small_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
big_llm = ChatOpenAI(temperature=0, model="gpt-4o")

output_parser = StrOutputParser()
prompt_template_location = "app/prompts/translation_prompt.md"

with open(prompt_template_location, "r") as file_handler:
    translation_str = file_handler.read()
    
translation_prompt = ChatPromptTemplate.from_template(
    template=translation_str,
)

translation_chain = translation_prompt | big_llm | output_parser
if txt and languages:
    if len(st.session_state.openai_api_key) == 0:
        st.toast("You need to set your API key in the sidebar!", icon="⚠️")
    else:
        with st.status("Translating...", expanded=True) as status:
            st.write("Extracting placeholders...")
            placeholders = extract_placeholders(txt)
            time.sleep(2)
            st.write("Translate to languages...")
            results = []
            try:
                for language in languages:
                    my_input = {
                        "language": language,
                        "source_text": txt,
                        "placeholders": placeholders
                    }
                    st.write(f"Translation into {language} in progress...")
                    results.append(translation_chain.invoke(my_input))
                    translation = "dummy things {text}".format(text=", ".join(placeholders))
                for result in results:
                    st.code(result)
                status.update(
                    label="Translation complete!", state="complete", expanded=True
                )
                st.toast("Everything translated!", icon="🎉")
            except APIConnectionError as apierror:
                status.update(
                    label="Wrong API key entered!", state="error", expanded=True
                )
                st.error(apierror.message, icon="🚨")
            except OpenAIError as openaierror:
                status.update(
                    label="Invalid API key!", state="error", expanded=True
                )
                st.error(openaierror.message, icon="🚨")
            except Exception as e:
                status.update(
                    label="Translation Failed!", state="error", expanded=True
                )
