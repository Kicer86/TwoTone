import unittest

from twotone.tools.language_fixer import LanguageFixerTool


class GuessAudioLanguageUnitTest(unittest.TestCase):
    """Unit tests for LanguageFixerTool._guess_audio_language() and helpers."""

    def setUp(self):
        self.tool = LanguageFixerTool()

    # --- _guess_language_from_keyword_context ---

    def test_keyword_dub_english(self):
        result = self.tool._guess_language_from_keyword_context("dubbed en")
        self.assertEqual(result, "eng")

    def test_keyword_dubbing_polish(self):
        result = self.tool._guess_language_from_keyword_context("dubbing pl")
        self.assertEqual(result, "pol")

    def test_keyword_lektor_pol(self):
        result = self.tool._guess_language_from_keyword_context("lektor pl")
        self.assertEqual(result, "pol")

    def test_keyword_audio_then_lang(self):
        result = self.tool._guess_language_from_keyword_context("audio de")
        self.assertEqual(result, "deu")

    def test_keyword_lang_then_dub(self):
        result = self.tool._guess_language_from_keyword_context("en dub")
        self.assertEqual(result, "eng")

    def test_glued_endub(self):
        result = self.tool._guess_language_from_keyword_context("endub")
        self.assertEqual(result, "eng")

    def test_glued_pldubbing(self):
        result = self.tool._guess_language_from_keyword_context("pldubbing")
        self.assertEqual(result, "pol")

    def test_no_keyword_returns_none(self):
        result = self.tool._guess_language_from_keyword_context("just some text")
        self.assertIsNone(result)

    def test_empty_string(self):
        result = self.tool._guess_language_from_keyword_context("")
        self.assertIsNone(result)

    # --- _guess_language_from_label_tokens ---

    def test_label_full_language_name(self):
        result = self.tool._guess_language_from_label_tokens("English")
        self.assertEqual(result, "eng")

    def test_label_full_name_polish(self):
        result = self.tool._guess_language_from_label_tokens("Polish")
        self.assertEqual(result, "pol")

    def test_label_three_letter_code(self):
        result = self.tool._guess_language_from_label_tokens("deu")
        self.assertEqual(result, "deu")

    def test_label_two_letter_code_skipped(self):
        """Two-letter codes are skipped to avoid false positives."""
        result = self.tool._guess_language_from_label_tokens("en")
        self.assertIsNone(result)

    def test_label_unknown_token(self):
        result = self.tool._guess_language_from_label_tokens("xyz123")
        self.assertIsNone(result)

    def test_label_empty(self):
        result = self.tool._guess_language_from_label_tokens("")
        self.assertIsNone(result)

    # --- _guess_audio_language (combined) ---

    def test_combined_keyword_takes_priority(self):
        """Keyword context should be preferred over label token matching."""
        result = self.tool._guess_audio_language("English dub pl")
        # "dub pl" keyword match should win
        self.assertEqual(result, "pol")

    def test_combined_falls_back_to_label(self):
        result = self.tool._guess_audio_language("English")
        self.assertEqual(result, "eng")

    def test_combined_empty(self):
        result = self.tool._guess_audio_language("")
        self.assertIsNone(result)

    # --- _guess_audio_language_from_filename ---

    def test_filename_endub(self):
        result = self.tool._guess_audio_language_from_filename("/path/to/sample_endub.mp4")
        self.assertEqual(result, "eng")

    def test_filename_no_keyword(self):
        result = self.tool._guess_audio_language_from_filename("/path/to/movie.mp4")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
