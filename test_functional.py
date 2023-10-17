import unittest

from main import main


class TestMainFunction(unittest.TestCase):
    def test_main_function(self):
        n_results = 5
        search_query = "large language models NLP benefits"
        search_objective = "To find empirical evidence supporting the performance benefits of large language models in NLP tasks like text summarization, translation, and question-answering."

        expected_keys = ["link", "snippet", "score", "ranking_score", "ai_content"]

        result_dict = main(n_results, search_query, search_objective)

        # Check if the result_dict has all the expected keys
        self.assertTrue(all(key in result_dict for key in expected_keys))

        # Additional checks can be added based on what you expect the function to return
        ai_content = result_dict.get("ai_content")
        self.assertIsNotNone(ai_content)
        summary = None
        if ai_content:
            summary = ai_content.get("summary")
        self.assertIsNotNone(summary)


if __name__ == "__main__":
    unittest.main()
