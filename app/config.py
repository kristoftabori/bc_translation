# app/config.py
LANGUAGES = [
    "Hungarian",
    "Spanish",
    "French",
    "German",
    "Japanese",
    "Arabic",
    "Hindi",
    "Portugese",
    "Italian",
]
# PROMPT_TEMPLATE_LOCATION = "app/prompts/translation_prompt.md"
PROMPT_TEMPLATE_LOCATION = "app/prompts/translation_prompt_v1.md"
CORRECTION_PROMPT_TEMPLATE_LOCATION = "app/prompts/correction_prompt.md"
MAX_CORRECTION_ITERATION = 5
MODEL_CONFIG = {
    "small": "gpt-4o-mini",
    "large": "gpt-4o"
}
