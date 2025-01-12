from placeholder_parsing import extract_placeholders
import pandas as pd
from typing import List
from langchain.chains.base import Chain


def translate_row(source_text: str, language: str, translate_chain: Chain) -> str:
    placeholders = extract_placeholders(source_text)

    result = translate_chain.invoke({
        "source_text": source_text,
        "language": language,
        "placeholders": placeholders
    })

    if not isinstance(result, str):
        raise ValueError(f"Expected string from translate_chain.invoke, got: {type(result)}")

    return result


def translate_dataframe(df: pd.DataFrame, languages: List[str], translation_chain: Chain) -> pd.DataFrame:
    translated_df = pd.DataFrame({"English": df.iloc[:, 0]})
    for language in languages:
        translated_df[language] = translated_df["English"].apply(
            lambda row: translate_row(row, language, translation_chain)
        )

    return translated_df
