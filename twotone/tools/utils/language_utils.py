
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
