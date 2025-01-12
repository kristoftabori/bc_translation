from app.placeholder_parsing import extract_placeholders
from langchain_openai import OpenAIEmbeddings
from langchain.evaluation.schema import EvaluatorType, StringEvaluator
from langchain.chains.base import Chain
from langchain.evaluation import load_evaluator
import pandas as pd
import numpy as np
from typing import Callable, List
import re
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))


def placeholder_identity(original: str, translated: str) -> float:
    original_placeholders = extract_placeholders(original)
    translated_placeholders = extract_placeholders(translated)

    return float(set(original_placeholders) == set(translated_placeholders))


def openai_embeddings_similarity(original: str, translated: str) -> dict:
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    evaluator = load_evaluator(
        EvaluatorType.EMBEDDING_DISTANCE,
        embeddings=embedding_model
    )

    if not hasattr(evaluator, "evaluate_strings"):
        raise TypeError(f"Unexpected evaluator type: {type(evaluator)}")

    if isinstance(evaluator, StringEvaluator):
        return evaluator.evaluate_strings(
            prediction=translated,
            reference=original
        )
    elif isinstance(evaluator, Chain):
        raise TypeError("Evaluator is a Chain, which does not support evaluate_strings.")
    else:
        raise TypeError(f"Unexpected evaluator type: {type(evaluator)}")


def eval_row(
    row: pd.Series,
    language: str,
    evaluation_func: Callable[[str, str], float],
    default_language: str = "English"
) -> float:
    return evaluation_func(row[default_language], row[language])


def row_terms(row: pd.Series, col_name: str) -> List[str]:
    pattern = r"\[.*?\]|\S+"
    unnecessary = ".;,?!"
    return [str(elem).strip(unnecessary).lower() for elem in re.findall(pattern, str(row[col_name]))]


def get_df_terms(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    new_df = df.copy()
    new_df[f"{str(col_name)}_terms"] = new_df.apply(
        lambda row: row_terms(row, col_name), axis=1
    )
    return new_df


def calculate_term_freq_in_doc(term: str, row: pd.Series, col_name: str) -> float:
    return sum([int(element == term) for element in row[f"{str(col_name)}_terms"]])


def calcualte_doc_length(row: pd.Series, col_name: str) -> int:
    return len(row[f"{str(col_name)}_terms"])


def docs_contain(term: str, df: pd.DataFrame, col_name: str) -> int:
    return df[f"{str(col_name)}_terms"].apply(
        lambda row: term.lower() in row
    ).sum().astype("int")


def term_idf(term: str, df: pd.DataFrame, col_name: str) -> float:
    doc_num = df.shape[0]
    docs_with_term = docs_contain(str(term).lower(), df, col_name)
    if docs_with_term == 0:
        raise ZeroDivisionError(f"No documents contain the term '{term}' in column '{col_name}'")

    return np.log(doc_num/docs_with_term)


def calcualte_term_bm25_for_row(
    term: str,
    row: pd.Series,
    k: float,
    b: float,
    df: pd.DataFrame,
    col_name: str
) -> float:
    idf = term_idf(term, df, col_name)
    freq = calculate_term_freq_in_doc(str(term).lower(), row, col_name)
    doc_length = calcualte_doc_length(row, col_name)
    avg_doc_length = df.apply(
        lambda row: calcualte_doc_length(row, col_name), axis=1
    ).mean()

    return idf*(freq*(k + 1.0))/(freq + k*(1.0 - b + b*doc_length/avg_doc_length))


def calcualte_distribution_for_row(row: pd.Series, k: float, b: float, df: pd.DataFrame, col_name: str) -> List[float]:
    term_distribution = []
    for term in row[f"{str(col_name)}_terms"]:
        term_distribution.append(
            calcualte_term_bm25_for_row(
                str(term).lower(),
                row,
                k,
                b,
                df,
                col_name
            )
        )
    return sorted(term_distribution)


def calculate_max_distance(first_list: List[float], second_list: List[float]) -> float:
    max_distance = 0.0
    for target in first_list + second_list:
        first_cdf = sum(elem <= target for elem in first_list)/len(first_list)
        second_cdf = sum(elem <= target for elem in second_list)/len(second_list)
        if np.abs(first_cdf - second_cdf) > max_distance:
            max_distance = np.abs(first_cdf - second_cdf)
    return max_distance


def bm25_term_similarity(
    k: float,
    b: float,
    df: pd.DataFrame,
    col_name_original: str,
    col_name_translated: str
) -> float:
    assert col_name_original in df.columns, f"Column {col_name_original} not found in DataFrame"
    assert col_name_translated in df.columns, f"Column {col_name_translated} not found in DataFrame"
    assert k > 0, "Parameter k must be positive"
    assert b > 0, "Parameter b must be positive"

    if df.empty:
        raise ValueError("DataFrame is empty or contains no rows")

    terms_df = get_df_terms(df, col_name_original)
    terms_df = get_df_terms(terms_df, col_name_translated)
    o_bm_name = f"{col_name_original}_bm25s"
    t_bm_name = f"{col_name_translated}_bm25s"
    o_bm_avg_name = f"{col_name_original}_bm25s_avg"
    t_bm_avg_name = f"{col_name_translated}_bm25s_avg"
    terms_df[o_bm_name] = terms_df.apply(
        lambda row: calcualte_distribution_for_row(
            row,
            k,
            b,
            terms_df,
            col_name_original
        ),
        axis=1
    )
    terms_df[t_bm_name] = terms_df.apply(
        lambda row: calcualte_distribution_for_row(
            row,
            k,
            b,
            terms_df,
            col_name_translated
        ),
        axis=1
    )
    terms_df[o_bm_avg_name] = terms_df.apply(
        lambda row: np.average(row[f"{col_name_original}_bm25s"]), axis=1
    )
    terms_df[t_bm_avg_name] = terms_df.apply(
        lambda row: np.average(row[t_bm_name]), axis=1
    )
    terms_df[o_bm_name] = terms_df.apply(
        lambda row: [elem - row[o_bm_avg_name] for elem in row[o_bm_name]],
        axis=1
    )
    terms_df[t_bm_name] = terms_df.apply(
        lambda row: [elem - row[t_bm_avg_name] for elem in row[t_bm_name]],
        axis=1
    )

    return 1.0 - terms_df.apply(
        lambda row: calculate_max_distance(
            row[f"{col_name_original}_bm25s"],
            row[f"{col_name_translated}_bm25s"]
        ), axis=1
    ).mean()
