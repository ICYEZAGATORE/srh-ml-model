"""
safety_filter.py — Safety and content moderation layer for the SRH pipeline.
Applied to both incoming queries and outgoing LLM responses.
"""

from dataclasses import dataclass

BLOCKED_PHRASES_EN = [
    "have sex with someone who is asleep",
    "force someone to have sex",
    "force them to have sex",
    "drug someone",
    "rape",
    "self harm",
    "kill myself",
    "commit suicide",
    "how to get pregnant without consent",
]

BLOCKED_PHRASES_RW = [
    "kumvisha umuntu nabi",
    "kwica",
]

UNSAFE_RESPONSE_KEYWORDS = [
    "kill", "murder", "rape", "force them", "drug them",
    "exploit", "assault them",
]

FALLBACK_EN = (
    "I'm not able to answer that question here. "
    "Please speak to a trusted health worker or contact a helpline. "
    "In Rwanda, you can reach Isange One Stop Centres for support."
)

FALLBACK_RW = (
    "Ntashobora gusubiza icyo kibazo hano. "
    "Baza umujyanama w'ubuzima ukenya cyangwa hamagara inzego z'ubufasha. "
    "Mu Rwanda, ushobora kugana Isange One Stop Centres kugirango ubone inkunga."
)


@dataclass
class SafetyResult:
    is_safe: bool
    matched_phrase: str | None
    fallback_message: str | None


def check_query(query: str, lang: str = "en") -> SafetyResult:
    """
    Check whether a user query is safe to process.

    Args:
        query: The user's raw query string.
        lang:  Detected language code ("en" or "rw").

    Returns:
        SafetyResult with is_safe=True if the query is acceptable.
    """
    q_lower = query.lower()
    phrases = BLOCKED_PHRASES_EN + (BLOCKED_PHRASES_RW if lang == "rw" else [])
    for phrase in phrases:
        if phrase in q_lower:
            fallback = FALLBACK_RW if lang == "rw" else FALLBACK_EN
            return SafetyResult(is_safe=False, matched_phrase=phrase, fallback_message=fallback)
    return SafetyResult(is_safe=True, matched_phrase=None, fallback_message=None)


def check_response(response: str, lang: str = "en") -> SafetyResult:
    """
    Check whether a generated LLM response is safe to return to the user.

    Args:
        response: The LLM-generated response string.
        lang:     Language of the response.

    Returns:
        SafetyResult with is_safe=True if the response is acceptable.
    """
    r_lower = response.lower()
    for kw in UNSAFE_RESPONSE_KEYWORDS:
        if kw in r_lower:
            fallback = FALLBACK_RW if lang == "rw" else FALLBACK_EN
            return SafetyResult(is_safe=False, matched_phrase=kw, fallback_message=fallback)
    return SafetyResult(is_safe=True, matched_phrase=None, fallback_message=None)
