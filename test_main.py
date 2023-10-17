import unittest
from unittest.mock import patch

from main import main


class TestMainFunction(unittest.TestCase):
    @patch("main.fetch_and_summarize")
    def test_main(self, mock_fetch_and_summarize):
        mock_fetch_and_summarize.return_value = (
            {
                "link": "example.com",
                "snippet": "Example Snippet",
                "score": 90,
                "ranking_score": 0.9,
                "summary": "Example Summary",
            },
            "Example Reason",
        )

        with patch("builtins.print") as mock_print:
            main(5, "example query", "example goal")

            expected_calls = [
                (("example.com",),),
                (("Example Snippet",),),
                (("score: 90",),),
                (("ranking score: 0.9",),),
                (("Ai content:",),),
                (("summary: Example Summary",),),
                (("reason: Example Reason",),),
            ]
            mock_print.assert_has_calls(expected_calls)


if __name__ == "__main__":
    unittest.main()
