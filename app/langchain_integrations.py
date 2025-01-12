from config import PROMPT_TEMPLATE_LOCATION
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

with open(PROMPT_TEMPLATE_LOCATION, "r") as file_handler:
    TRANSLATION_STR = file_handler.read()

TRANSLATION_PROMPT = ChatPromptTemplate.from_template(
    template=TRANSLATION_STR,
)


def build_translation_chain(model_name: str) -> RunnableSerializable[dict, str]:
    translate_llm = ChatOpenAI(temperature=0, model=model_name)
    translation_prompt = TRANSLATION_PROMPT
    output_parser = StrOutputParser()
    print(model_name)

    return translation_prompt | translate_llm | output_parser
