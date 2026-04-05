import unittest

from parameterized import parameterized

from twotone.tools.language_fixer import LanguageFixerTool


def _make_tool() -> LanguageFixerTool:
    return LanguageFixerTool()


class GuessLanguageFromKeywordContextTest(unittest.TestCase):
    """Tests for LanguageFixerTool._guess_language_from_keyword_context."""

    @parameterized.expand([
        ("dubbing_en",         "dubbing en",         "eng"),
        ("dubbed_eng",         "dubbed eng",         "eng"),
        ("dub_pl",             "dub pl",             "pol"),
        ("lektor_pl",          "lektor pl",          "pol"),
        ("voice_de",           "voice de",            "deu"),
        ("audio_eng",          "audio eng",           "eng"),
        ("reversed_en_dub",    "en dub",              "eng"),
        ("reversed_pl_dubbing","pl dubbing",          "pol"),
        ("case_insensitive",   "DUBBING EN",          "eng"),
        ("separator_dash",     "dub-en",              "eng"),
        ("separator_space",     "dub en",              "eng"),
        ("glued_endub",        "endub",               "eng"),
        ("glued_pldub",        "pldub",               "pol"),
        ("glued_pldubbing",    "pldubbing",           "pol"),
        ("glued_endubbed",     "endubbed",            "eng"),
    ])
    def test_keyword_match(self, _name: str, text: str, expected: str) -> None:
        tool = _make_tool()
        self.assertEqual(tool._guess_language_from_keyword_context(text), expected)

    @parameterized.expand([
        ("empty_string",   ""),
        ("no_keyword",     "some random text"),
        ("only_keyword",   "dubbing"),
        ("invalid_code",   "dubbing xyz"),
    ])
    def test_no_match(self, _name: str, text: str) -> None:
        tool = _make_tool()
        self.assertIsNone(tool._guess_language_from_keyword_context(text))


class GuessLanguageFromLabelTokensTest(unittest.TestCase):
    """Tests for LanguageFixerTool._guess_language_from_label_tokens (static)."""

    @parameterized.expand([
        ("full_name_english",  "English",  "eng"),
        ("full_name_polish",   "Polish",   "pol"),
        ("full_name_german",   "German",   "deu"),
        ("full_name_french",   "French",   "fra"),
        ("three_letter_eng",   "eng",      "eng"),
        ("three_letter_pol",   "pol",      "pol"),
    ])
    def test_label_match(self, _name: str, label: str, expected: str) -> None:
        self.assertEqual(LanguageFixerTool._guess_language_from_label_tokens(label), expected)

    @parameterized.expand([
        ("empty",           ""),
        ("single_char",     "x"),
        ("two_letter_code", "en"),     # 2-letter codes are intentionally skipped
        ("gibberish",       "xyzxyz"),
        ("number_only",     "12345"),
    ])
    def test_no_match(self, _name: str, label: str) -> None:
        self.assertIsNone(LanguageFixerTool._guess_language_from_label_tokens(label))


class GuessAudioLanguageTest(unittest.TestCase):
    """Tests for LanguageFixerTool._guess_audio_language — combines keyword + label."""

    def test_keyword_takes_priority(self) -> None:
        tool = _make_tool()
        # "dubbing pl" should match keyword context (pol),
        # even though "dubbing" alone might tokenize differently
        self.assertEqual(tool._guess_audio_language("dubbing pl"), "pol")

    def test_falls_back_to_label_tokens(self) -> None:
        tool = _make_tool()
        self.assertEqual(tool._guess_audio_language("English"), "eng")

    def test_empty_returns_none(self) -> None:
        tool = _make_tool()
        self.assertIsNone(tool._guess_audio_language(""))


class GuessAudioLanguageFromFilenameTest(unittest.TestCase):
    """Tests for LanguageFixerTool._guess_audio_language_from_filename."""

    def test_filename_with_keyword(self) -> None:
        tool = _make_tool()
        self.assertEqual(tool._guess_audio_language_from_filename("/path/to/Movie dubbing en.mkv"), "eng")

    def test_filename_with_glued_pattern(self) -> None:
        tool = _make_tool()
        self.assertEqual(tool._guess_audio_language_from_filename("/path/to/Movie pldub.mkv"), "pol")

    def test_filename_no_match(self) -> None:
        tool = _make_tool()
        self.assertIsNone(tool._guess_audio_language_from_filename("/path/to/Movie.mkv"))


if __name__ == "__main__":
    unittest.main()
