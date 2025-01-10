from typing import List
import re


def extract_placeholders(text: str, placeholder_style: str = r"\[(.*?)\]") -> List[str]:
    return list(set(re.findall(placeholder_style, text)))
