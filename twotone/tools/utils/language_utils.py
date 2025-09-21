
import pycountry


def unify_lang(lang: str) -> str:
    lang_info = pycountry.languages.get(alpha_2 = lang) if len(lang) == 2 else pycountry.languages.get(alpha_3 = lang)
    if lang_info:
        return lang_info.alpha_3
    else:
        # try to look for in biliographic set
        lang_info = pycountry.languages.get(bibliographic = lang)
        if lang_info:
            return lang_info.alpha_3
        else:
            raise ValueError(f"Invalid or unsupported language code: {lang}")


def language_name(lang: str | None) -> str:
    """Return a human friendly language name for ``lang`` code.

    If ``lang`` is ``None`` or not recognized, ``"Unknown"`` is returned.
    """

    if not lang:
        return "Unknown"

    try:
        lang_code = unify_lang(lang)
    except ValueError:
        return lang

    lang_info = pycountry.languages.get(alpha_3=lang_code)
    return lang_info.name if lang_info else lang
