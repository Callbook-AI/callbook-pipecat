import re
from typing import Iterable, Pattern, Union

Regex = Union[str, Pattern[str]]

def regex_list_matches(text: str, patterns: Iterable[Regex]) -> bool:
    """
    Return True as soon as one pattern in *patterns* is found in *text*.

    Parameters
    ----------
    text : str
        The text to be searched.
    patterns : Iterable[Union[str, re.Pattern]]
        Regex patterns.  These may be pre-compiled (recommended) or plain strings.

    Notes
    -----
    • Empty / None pattern lists return False.  
    • The search is Unicode-aware.  
    • If a pattern is a plain string it is compiled once, on demand, and cached
      so that repeated calls stay fast.
    """
    if not text or not patterns:
        return False

    # Cache compiled expressions on the function object itself
    _cache = getattr(regex_list_matches, "_cache", {})
    setattr(regex_list_matches, "_cache", _cache)

    for pat in patterns:
        # Compile only if we’ve never seen this exact pattern object / string
        if isinstance(pat, str):
            regex = _cache.get(pat)
            if regex is None:
                regex = re.compile(pat, re.UNICODE)
                _cache[pat] = regex
        else:                     # already an re.Pattern
            regex = pat

        if regex.search(text):
            return True

    return False




def _test_regex_list_matches() -> None:
    """
    Minimal self-contained test-suite (4 cases) for `regex_list_matches`.

    • Two cases use the Italian “sì” pattern you provided.
    • Two cases use a second simple pattern for “hello”.
    """
    # pattern from your SENSIBLE_AND_FAST_RESPONSE_WORDS
    PATTERN_1 = r"(?i)\bs(?:i|[ìí])\b(?=[.,!?]|$)"
    # a second, independent example
    PATTERN_2 = r"\bhello\b"

    # --- case 1: pattern 1 should match -----------------------------
    assert regex_list_matches("Sì!", [PATTERN_1]) is True,  "Case 1 failed"

    # --- case 2: pattern 1 should NOT match -------------------------
    assert regex_list_matches("disintegrare", [PATTERN_1]) is False, "Case 2 failed"

    # --- case 3: pattern 1 should NOT match -------------------------
    assert regex_list_matches("Si Que", [PATTERN_1]) is False, "Case 2 failed"

    # --- case 4: pattern 2 should match -----------------------------
    assert regex_list_matches("hello world", [PATTERN_2]) is True,   "Case 3 failed"

    # --- case 5: pattern 2 should NOT match -------------------------
    assert regex_list_matches("wellhellothere", [PATTERN_2]) is False, "Case 4 failed"


    print("All regex_list_matches tests passed.")


if __name__ == '__main__':
    _test_regex_list_matches()