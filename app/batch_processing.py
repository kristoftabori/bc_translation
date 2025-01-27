from placeholder_parsing import extract_placeholders
import pandas as pd
from typing import List
from langchain.chains.base import Chain
from langgraph.graph.state import CompiledStateGraph


def translate_row(
    source_text: str,
    language: str,
    translate_chain: Chain,
    correction_chain: Chain,
    processing_graph:  CompiledStateGraph
) -> str:
    placeholders = extract_placeholders(source_text)
    initial_state = {
        "placeholders": placeholders,
        "iteration_count": 0,
        "original_sentence": source_text,
        "target_language": language,
        "translation_chain": translate_chain,
        "correction_chain": correction_chain,
    }

    result = processing_graph.invoke(initial_state)["translations"][-1]

    if not isinstance(result, str):
        raise ValueError(f"Expected string from translate_chain.invoke, got: {type(result)}")

    return result


def translate_dataframe(
    df: pd.DataFrame,
    languages: List[str],
    translation_chain: Chain,
    correction_chain: Chain,
    processing_graph:  CompiledStateGraph
) -> pd.DataFrame:
    translated_df = pd.DataFrame({"English": df.iloc[:, 0]})
    for language in languages:
        translated_df[language] = translated_df["English"].apply(
            lambda row: translate_row(row, language, translation_chain, correction_chain, processing_graph)
        )

    return translated_df
