"""
Sanity checks on agent_pipeline.constants.

These guard against the class of bug this project actually hit once already:
a typo'd environment variable name that silently breaks provider detection.
"""

from agent_pipeline import constants


def test_default_topics_are_a_non_empty_pool_of_distinct_strings() -> None:
    assert len(constants.DEFAULT_TOPICS) > 1
    assert all(topic.strip() for topic in constants.DEFAULT_TOPICS)
    assert len(set(constants.DEFAULT_TOPICS)) == len(constants.DEFAULT_TOPICS)


def test_default_model_is_a_non_empty_string() -> None:
    assert constants.DEFAULT_OPENAI_MODEL.strip()


def test_agent_names_are_distinct_and_non_empty() -> None:
    names = [constants.RESEARCHER_NAME, constants.WRITER_NAME, constants.EDITOR_NAME]
    assert all(name.strip() for name in names)
    assert len(set(names)) == len(names)


def test_agent_instructions_are_non_empty() -> None:
    for instructions in (
        constants.RESEARCHER_INSTRUCTIONS,
        constants.WRITER_INSTRUCTIONS,
        constants.EDITOR_INSTRUCTIONS,
    ):
        assert instructions.strip()


def test_openai_key_prefix_matches_real_openai_format() -> None:
    assert constants.OPENAI_KEY_PREFIX == "sk-"


def test_env_var_names_match_agent_framework_conventions() -> None:
    # These must exactly match what agent_framework/OpenAIChatClient reads from
    # the environment; a typo here would silently break provider auto-detection.
    assert constants.ENV_OPENAI_API_KEY == "OPENAI_API_KEY"
    assert constants.ENV_OPENAI_CHAT_MODEL == "OPENAI_CHAT_MODEL"
    assert constants.ENV_AZURE_OPENAI_ENDPOINT == "AZURE_OPENAI_ENDPOINT"
    assert constants.ENV_AZURE_OPENAI_CHAT_MODEL == "AZURE_OPENAI_CHAT_MODEL"
    assert constants.ENV_AZURE_OPENAI_API_KEY == "AZURE_OPENAI_API_KEY"
    assert constants.ENV_AZURE_OPENAI_API_VERSION == "AZURE_OPENAI_API_VERSION"


def test_auth_error_markers_are_lowercase_and_non_empty() -> None:
    # looks_like_auth_error() lowercases the exception text before comparing,
    # so an uppercase marker here would silently never match.
    assert len(constants.AUTH_ERROR_MARKERS) > 0
    assert all(marker == marker.lower() for marker in constants.AUTH_ERROR_MARKERS)
