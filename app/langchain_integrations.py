from app.config import PROMPT_TEMPLATE_LOCATION, CORRECTION_PROMPT_TEMPLATE_LOCATION, MAX_CORRECTION_ITERATION
from app.placeholder_parsing import extract_placeholders
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from operator import add


with open(PROMPT_TEMPLATE_LOCATION, "r") as file_handler:
    TRANSLATION_STR = file_handler.read()

TRANSLATION_PROMPT = ChatPromptTemplate.from_template(
    template=TRANSLATION_STR,
)

with open(CORRECTION_PROMPT_TEMPLATE_LOCATION, "r") as file_handler:
    CORRECTION_PROMPT_STR = file_handler.read()

CORRECTION_PROMPT = ChatPromptTemplate.from_template(
    template=CORRECTION_PROMPT_STR,
)


def build_simple_chain(model_name: str, prompt_template: ChatPromptTemplate) -> RunnableSerializable[dict, str]:
    translate_llm = ChatOpenAI(temperature=0, model=model_name)
    translation_prompt = prompt_template
    output_parser = StrOutputParser()

    return translation_prompt | translate_llm | output_parser


def build_translation_chain(model_name: str) -> RunnableSerializable[dict, str]:
    return build_simple_chain(model_name, TRANSLATION_PROMPT)


def build_correction_chain(model_name: str) -> RunnableSerializable[dict, str]:
    return build_simple_chain(model_name, CORRECTION_PROMPT)


class TranslationState(TypedDict):
    translations: Annotated[List[str], add]
    placeholders: List[str]
    iteration_count: int
    original_sentence: str
    target_language: str
    translation_chain: RunnableSerializable[dict, str]
    correction_chain: RunnableSerializable[dict, str]


def initial_translate(state: TranslationState) -> dict:
    translation_result = state["translation_chain"].invoke(dict(
        source_text=state["original_sentence"],
        language=state["target_language"],
        placeholders=state["placeholders"]
    ))

    return {
        "translations": [translation_result]
    }


def are_placeholders_maintained(state: TranslationState) -> str:
    assert "translations" in state
    assert len(state["translations"]) > 0

    new_placeholders = extract_placeholders(state["translations"][-1])

    if set(state["placeholders"]) == set(new_placeholders):
        return "placeholder maintained"
    if state["iteration_count"] > MAX_CORRECTION_ITERATION:
        return "too many iterations"
    return "problem with placeholders"


def correct_mistake(state: TranslationState) -> dict:
    assert "translations" in state
    assert len(state["translations"]) > 0

    correction_result = state["correction_chain"].invoke(dict(
        source_text=state["original_sentence"],
        language=state["target_language"],
        placeholders=state["placeholders"],
        counter_examples="\n\n".join("translations")
    ))

    return {
        "translations": [correction_result],
        "iteration_count": state["iteration_count"] + 1
    }


def build_correction_graph() -> CompiledStateGraph:
    workflow = StateGraph(TranslationState)

    workflow.add_node("initial_translate", initial_translate)
    workflow.add_node("correct_mistake", correct_mistake)

    workflow.set_entry_point("initial_translate")
    workflow.add_conditional_edges(
        "initial_translate",
        are_placeholders_maintained,
        {
            "placeholder maintained": END,
            "too many iterations": END,
            "problem with placeholders": "correct_mistake"
        }
    )
    workflow.add_conditional_edges(
        "correct_mistake",
        are_placeholders_maintained,
        {
            "placeholder maintained": END,
            "too many iterations": END,
            "problem with placeholders": "correct_mistake"
        }
    )

    return workflow.compile()
