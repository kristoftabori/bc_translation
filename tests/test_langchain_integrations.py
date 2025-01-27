import pytest
from unittest.mock import patch, MagicMock
from app.langchain_integrations import (
    build_simple_chain,
    build_translation_chain,
    build_correction_chain,
    initial_translate,
    are_placeholders_maintained,
    correct_mistake,
    build_correction_graph,
    TranslationState,
)


@pytest.fixture
def mock_prompt_templates():
    # Mock the file reading for templates
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.side_effect = [
            "Translation prompt template",
            "Correction prompt template",
        ]
        yield


@pytest.fixture
def mock_chains():
    # Mock the ChatPromptTemplate and RunnableSerializable
    with patch("app.langchain_integrations.ChatPromptTemplate.from_template") as mock_from_template:
        mock_from_template.side_effect = lambda template: MagicMock(template=template)

    with patch("app.langchain_integrations.ChatOpenAI") as mock_llm:
        mock_llm.return_value = MagicMock()

    with patch("app.langchain_integrations.StrOutputParser") as mock_parser:
        mock_parser.return_value = MagicMock()


@pytest.fixture
def mock_state():
    return TranslationState(
        translations=[],
        placeholders=["placeholder1", "placeholder2"],
        iteration_count=0,
        original_sentence="This is a test [placeholder1] and [placeholder2].",
        target_language="French",
        translation_chain=MagicMock(),
        correction_chain=MagicMock(),
    )


def test_build_simple_chain(mock_prompt_templates, mock_chains):
    chain = build_simple_chain("gpt-3.5-turbo", MagicMock())
    assert chain is not None


def test_build_translation_chain(mock_prompt_templates, mock_chains):
    chain = build_translation_chain("gpt-3.5-turbo")
    assert chain is not None


def test_build_correction_chain(mock_prompt_templates, mock_chains):
    chain = build_correction_chain("gpt-3.5-turbo")
    assert chain is not None


def test_initial_translate(mock_state):
    mock_state["translation_chain"].invoke.return_value = "Ceci est un test [placeholder1] et [placeholder2]."
    new_state = initial_translate(mock_state)
    assert "translations" in new_state
    assert len(new_state["translations"]) == 1
    assert new_state["translations"][0] == "Ceci est un test [placeholder1] et [placeholder2]."


def test_are_placeholders_maintained(mock_state):
    mock_state["translations"] = ["Ceci est un test [placeholder1] et [placeholder2]."]
    result = are_placeholders_maintained(mock_state)
    assert result == "placeholder maintained"

    mock_state["translations"] = ["Ceci est un test [placeholder1]."]
    result = are_placeholders_maintained(mock_state)
    assert result == "problem with placeholders"

    mock_state["iteration_count"] = 10
    result = are_placeholders_maintained(mock_state)
    assert result == "too many iterations"


def test_correct_mistake(mock_state):
    mock_state["translations"] = ["Ceci est un test [placeholder1]."]
    mock_state["correction_chain"].invoke.return_value = "Ceci est un test [placeholder1] et [placeholder2]."
    new_state = correct_mistake(mock_state)
    assert "translations" in new_state
    assert len(new_state["translations"]) == 1
    assert new_state["translations"][0] == "Ceci est un test [placeholder1] et [placeholder2]."
    assert new_state["iteration_count"] == 1


def test_correction_graph(mock_state):
    mock_state["translation_chain"].invoke.return_value = "Ceci est un test [placeholder1]."
    mock_state["correction_chain"].invoke.return_value = "Ceci est un test [placeholder1] et [placeholder2]."

    graph = build_correction_graph()
    final_state = graph.invoke(mock_state)

    assert "translations" in final_state
    assert len(final_state["translations"]) == 2
    assert final_state["translations"][0] == "Ceci est un test [placeholder1]."
    assert final_state["translations"][1] == "Ceci est un test [placeholder1] et [placeholder2]."
    assert final_state["iteration_count"] == 1
    assert mock_state["correction_chain"].invoke.call_count == 1
    assert mock_state["translation_chain"].invoke.call_count == 1


def test_build_correction_graph():
    graph = build_correction_graph()
    assert graph is not None
