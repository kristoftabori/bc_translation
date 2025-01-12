import pytest
from unittest.mock import patch, Mock
from langchain.evaluation.schema import StringEvaluator
import pandas as pd
import numpy as np
from app.evals import (
    placeholder_identity,
    openai_embeddings_similarity,
    calculate_term_freq_in_doc,
    calcualte_doc_length,
    docs_contain,
    term_idf,
    calcualte_term_bm25_for_row,
    calcualte_distribution_for_row,
    calculate_max_distance,
    bm25_term_similarity,
)


# Tests for placeholder_identity
def test_placeholder_identity_exact_match():
    original = "Hello [name], welcome to [place]!"
    translated = "Hola [name], bienvenido a [place]!"
    assert placeholder_identity(original, translated) == 1.0


def test_placeholder_identity_different_placeholders():
    original = "Hello [name], welcome to [place]!"
    translated = "Hola [nombre], bienvenido a [lugar]!"
    assert placeholder_identity(original, translated) == 0.0


def test_placeholder_identity_missing_placeholder():
    original = "Hello [name], welcome to [place]!"
    translated = "Hola [name]!"
    assert placeholder_identity(original, translated) == 0.0


def test_placeholder_identity_no_placeholders():
    original = "Hello, welcome!"
    translated = "Hola, bienvenido!"
    assert placeholder_identity(original, translated) == 1.0


def test_placeholder_identity_empty_strings():
    assert placeholder_identity("", "") == 1.0


# Tests for openai_embeddings_similarity
@pytest.fixture
def mock_evaluator():
    # Mock StringEvaluator directly
    evaluator = Mock(spec=StringEvaluator)
    evaluator.evaluate_strings.return_value = {"score": 0.95}
    return evaluator


@pytest.fixture
def mock_embeddings():
    return Mock()


@patch("app.evals.OpenAIEmbeddings")
@patch("app.evals.load_evaluator")
def test_openai_embeddings_similarity_high_similarity(
    mock_load_evaluator,
    mock_openai_embeddings,
    mock_evaluator,
    mock_embeddings
):
    mock_openai_embeddings.return_value = mock_embeddings
    mock_load_evaluator.return_value = mock_evaluator

    original = "Hello world!"
    translated = "Hello world!"
    result = openai_embeddings_similarity(original, translated)

    assert result == {"score": 0.95}
    mock_load_evaluator.assert_called_once_with("embedding_distance", embeddings=mock_embeddings)
    mock_evaluator.evaluate_strings.assert_called_once_with(prediction=translated, reference=original)


@patch("app.evals.OpenAIEmbeddings")
@patch("app.evals.load_evaluator")
def test_openai_embeddings_similarity_different_texts(
    mock_load_evaluator,
    mock_openai_embeddings,
    mock_evaluator,
    mock_embeddings
):
    mock_openai_embeddings.return_value = mock_embeddings
    mock_load_evaluator.return_value = mock_evaluator
    mock_evaluator.evaluate_strings.return_value = {"score": 0.5}

    original = "Hello world!"
    translated = "Completely different text"
    result = openai_embeddings_similarity(original, translated)

    assert result == {"score": 0.5}


@patch("app.evals.OpenAIEmbeddings")
@patch("app.evals.load_evaluator")
def test_openai_embeddings_similarity_empty_strings(
    mock_load_evaluator,
    mock_openai_embeddings,
    mock_evaluator,
    mock_embeddings
):
    mock_openai_embeddings.return_value = mock_embeddings
    mock_load_evaluator.return_value = mock_evaluator

    result = openai_embeddings_similarity("", "")
    print(result)

    mock_evaluator.evaluate_strings.assert_called_once_with(prediction="", reference="")


# Sample test data
@pytest.fixture
def test_df():
    data = {
        "0_terms": [
            ["apple", "banana", "cherry"],
            ["banana", "cherry"],
            ["apple", "cherry", "cherry"]
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_row(test_df):
    return test_df.iloc[0]


def test_calculate_term_freq_in_doc(test_row):
    assert calculate_term_freq_in_doc("apple", test_row, 0) == 1
    assert calculate_term_freq_in_doc("cherry", test_row, 0) == 1
    assert calculate_term_freq_in_doc("orange", test_row, 0) == 0


def test_calculate_doc_length(test_row):
    assert calcualte_doc_length(test_row, 0) == 3


def test_docs_contain(test_df):
    assert docs_contain("apple", test_df, 0) == 2
    assert docs_contain("banana", test_df, 0) == 2
    assert docs_contain("cherry", test_df, 0) == 3
    assert docs_contain("orange", test_df, 0) == 0


def test_term_idf(test_df):
    # Test with existing terms
    assert np.isclose(term_idf("apple", test_df, 0), np.log(3/2))
    assert np.isclose(term_idf("cherry", test_df, 0), np.log(3/3))
    # Test with non-existing term
    with pytest.raises(ZeroDivisionError, match="No documents contain the term 'orange' in column '0'"):
        term_idf("orange", test_df, 0)


def test_calculate_term_bm25_for_row(test_df, test_row):
    k = 1.5
    b = 0.75
    result = calcualte_term_bm25_for_row("apple", test_row, k, b, test_df, 0)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_distribution_for_row(test_df, test_row):
    k = 1.5
    b = 0.75
    result = calcualte_distribution_for_row(test_row, k, b, test_df, 0)
    assert isinstance(result, list)
    assert len(result) == 3  # Length should match number of terms
    assert all(isinstance(x, float) for x in result)
    # Test if sorted
    assert result == sorted(result)


def test_calculate_max_distance():
    # Test with equal distributions
    assert calculate_max_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    # Test with different distributions
    first = [1.0, 2.0, 3.0]
    second = [2.0, 3.0, 4.0]
    result = calculate_max_distance(first, second)
    assert isinstance(result, float)
    assert result > 0

    # Test with empty lists
    with pytest.raises(ZeroDivisionError):
        calculate_max_distance([], [1.0, 2.0])


# Edge cases
def test_edge_cases(test_df):
    # Empty DataFrame
    empty_df = pd.DataFrame({"0_terms": []})
    with pytest.raises(ZeroDivisionError):
        term_idf("test", empty_df, 0)

    # Non-existent column
    with pytest.raises(KeyError):
        docs_contain("test", test_df, "invalid_col")


@patch("app.evals.get_df_terms", side_effect=lambda df, col: df)
@patch("app.evals.calcualte_distribution_for_row", side_effect=lambda row, k, b, df, col: [0.5, 0.3, 0.2])
@patch("app.evals.calculate_max_distance", side_effect=lambda x, y: abs(np.mean(x) - np.mean(y)))
def test_bm25_term_similarity_valid(mock_get_df_terms, mock_calc_bm25, mock_max_distance):
    df = pd.DataFrame({
        "original_terms": [["apple", "banana"], ["cherry"]],
        "translated_terms": [["manzana", "plátano"], ["cereza"]]
    })
    similarity = bm25_term_similarity(1.2, 0.75, df, "original_terms", "translated_terms")
    assert similarity >= 0 and similarity <= 1, "Similarity should be between 0 and 1"
    mock_get_df_terms.assert_called()
    mock_calc_bm25.assert_called()
    mock_max_distance.assert_called()


def test_bm25_term_similarity_missing_columns():
    df = pd.DataFrame({
        "wrong_column": [["apple", "banana"]],
    })
    with pytest.raises(AssertionError, match="Column original_terms not found in DataFrame"):
        bm25_term_similarity(1.2, 0.75, df, "original_terms", "translated_terms")


def test_bm25_term_similarity_invalid_k_b():
    df = pd.DataFrame({
        "original_terms": [["apple", "banana"]],
        "translated_terms": [["manzana", "plátano"]]
    })
    with pytest.raises(AssertionError, match="Parameter k must be positive"):
        bm25_term_similarity(-1, 0.75, df, "original_terms", "translated_terms")

    with pytest.raises(AssertionError, match="Parameter b must be positive"):
        bm25_term_similarity(1.2, -0.5, df, "original_terms", "translated_terms")


def test_bm25_term_similarity_empty_dataframe():
    df = pd.DataFrame(columns=["original_terms", "translated_terms"])
    with pytest.raises(ValueError, match="DataFrame is empty or contains no rows"):
        bm25_term_similarity(1.2, 0.75, df, "original_terms", "translated_terms")


@patch("app.evals.get_df_terms", side_effect=lambda df, col: df)
@patch("app.evals.calcualte_distribution_for_row", side_effect=lambda row, k, b, df, col: [0.0, 0.0, 0.0])
@patch("app.evals.calculate_max_distance", side_effect=lambda x, y: 1.0)
def test_bm25_term_similarity_no_similarity(mock_get_df_terms, mock_calc_bm25, mock_max_distance):
    df = pd.DataFrame({
        "original_terms": [["apple", "banana"]],
        "translated_terms": [["grape", "orange"]]
    })
    similarity = bm25_term_similarity(1.2, 0.75, df, "original_terms", "translated_terms")
    assert similarity == 0, "Similarity should be 0 for completely dissimilar terms"
