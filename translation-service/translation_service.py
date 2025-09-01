"""
service for translation
"""

from typing import Literal, Optional
import re
import torch
from fastmcp import FastMCP
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

MODEL_FOLDER ="/models/ml_full"
DEVICE = 0 if torch.cuda.is_available() else -1
MCP_PORT = 8002

app = FastMCP("translation-service")

tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER, use_fast = False, local_files_only = True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_FOLDER, local_files_only = True)
translator = pipeline("text2text-generation", model = model, tokenizer = tokenizer, device = DEVICE)

CYRILLIC_RE = re.compile(r"[а-щыьэюяіїєґА-ЩЫЬЭЮЯІЇЄҐ]")

def is_english(sentence: str) -> bool:
    """
    check if the message is in english
    """

    return (
        not re.compile(r"[а-щыьэюяіїєґА-ЩЫЬЭЮЯІЇЄҐ]").search(sentence)
        and re.search(r"[A-Za-z]", sentence) is not None
    )

def pick_response_language(english: str, ukrainian: str, real: Optional[str]) -> str:
    """
    pick the right language for response
    """

    if real == "uk":
        return ukrainian
    if real == "en":
        return english
    return english

@app.tool()
def translate(user_input: str, max_new_tokens: int = 128, language: Optional[Literal["uk","en"]] = None) -> str:
    """
    USE THIS TOOL WHEN: the user explicitly asks to translate / перекласти (keywords:
    "переклади", "перекладіть", "перекласти", "translate", "translate to Ukrainian",
    "укр:", "з англійської:", "to ukrainian:").
    DIRECTION: English -> Ukrainian only (input must be in English). Output is finalized text for the user.
    CONTRACT FOR CALLERS (important!):
    - Always pass `language="uk"` if the user's instruction (UI) is in Ukrainian, otherwise "en".
    - The returned string is FINAL user-facing text. DO NOT modify it in the caller.
    - On success, it includes a localized heading:
        'Ось ваш переклад за допомогою тули translate:'  /  'Here is your translation using tool translate:'
    followed by the model output exactly as generated (no edits).
    """

    sequence = (user_input or "").strip()

    if not sequence:
        return ""

    if len(sequence) > 128:
        return pick_response_language(
            "Input too long (max 128 characters).",
            "Вхідний текст занадто довгий (макс. 128 символів).",
            language
        )

    if not is_english(user_input):
        return pick_response_language(
            "Only English input is supported.",
            "Підтримується лише англійська як вхідний текст.",
            language,
        )

    result = translator(
        f"translate English to Ukrainian: {sequence}",
        max_new_tokens = max_new_tokens, num_beams = 4, do_sample = False
    )[0]["generated_text"]

    heading = "Ось ваш переклад за допомогою тули translate:" if language == "uk" else "Here is your translation using tool translate:"

    return f"{heading}\n{result}"

if __name__ == "__main__":
    app.run("http", host = "0.0.0.0", port = MCP_PORT)
