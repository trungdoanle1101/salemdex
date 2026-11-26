"""Splitting text into words: the default tokenizer of the embedders that need one.

It suits languages that separate words with spaces, Vietnamese included (each syllable a
word). It doesn't suit those written without them (Chinese, Japanese, Thai), where a
clause comes back as one word: pass those a tokenizer of their own.
"""

import re
import unicodedata


def split_words(text: str) -> list[str]:
    """Return the words of `text`, normalised to NFC and case-folded.

    A word is a run of letters, digits or underscores (`\\w+`); punctuation and spaces
    separate words and are dropped. Diacritics stay: "lao" and "lão" are two words.

    Args:
        text: The text to split.

    Returns:
        The words, in order, repeats kept; empty for a text without any.

    Example:
        split_words("Điều 12: Nghỉ phép") == ["điều", "12", "nghỉ", "phép"]
    """
    # NFC first: a Vietnamese letter such as "ạ" can be one character or a letter
    # plus a combining mark, and both must hash alike
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
