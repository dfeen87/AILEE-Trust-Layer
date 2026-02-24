import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import ailee_ai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ailee_ai import TrustScorer, Lineage, AILEE_AI

class TestTrustScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = TrustScorer()

    def test_confidence_score(self):
        text = "Short answer"
        score = self.scorer._compute_confidence(text)
        self.assertLess(score, 0.5)

        text = "Long answer " * 50
        score = self.scorer._compute_confidence(text)
        self.assertGreater(score, 0.5)

        text = "Maybe it is possibly correct"
        score = self.scorer._compute_confidence(text)
        self.assertLess(score, 1.0)

    def test_safety_score(self):
        text = "This is a safe response."
        score = self.scorer._compute_safety(text)
        self.assertEqual(score, 1.0)

        text = "An error occurred and failed."
        score = self.scorer._compute_safety(text)
        self.assertLess(score, 1.0)

    def test_consistency_score(self):
        text = "The sky is blue."
        peers = ["The sky is blue.", "The sky is blue."]
        score = self.scorer._compute_consistency(text, peers)
        self.assertEqual(score, 1.0)

        peers = ["The grass is green.", "The water is wet."]
        score = self.scorer._compute_consistency(text, peers)
        self.assertLess(score, 0.5)

class TestLineage(unittest.TestCase):
    def test_deterministic_hash(self):
        query = "test query"
        outputs = [
            {"model": "B", "answer": "b"},
            {"model": "A", "answer": "a"}
        ]
        final = "final"

        lineage1 = Lineage.build(query, outputs, final)
        lineage2 = Lineage.build(query, outputs, final)

        self.assertEqual(lineage1['verification_hash'], lineage2['verification_hash'])
        self.assertEqual(lineage1['models'], ["A", "B"])

class TestAILEE_AI(unittest.TestCase):
    @patch('ailee_ai.DDGS')
    @patch('ailee_ai.openai')
    @patch('ailee_ai.anthropic')
    @patch('ailee_ai.genai')
    @patch('ailee_ai.requests.post')
    def test_process_flow_with_mocks(self, mock_post, mock_genai, mock_anthropic, mock_openai, mock_ddgs):
        # Setup mocks

        # Mock Search
        mock_ddgs_instance = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = [
            {'title': 'Test Source', 'href': 'http://test.com', 'body': 'Test content'}
        ]

        # Mock LLMs
        # OpenAI
        mock_openai_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_openai_client
        mock_openai_response = MagicMock()
        mock_openai_response.choices[0].message.content = "OpenAI Answer"
        mock_openai_client.chat.completions.create.return_value = mock_openai_response

        # Anthropic
        # We need to set the API key env var for the client to initialize?
        # Actually AILEE_AI init checks for env var, if not present it skips client init.
        # So we should patch os.environ or set the client manually.

        # Mock Trust API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "final_answer": "Trusted Answer",
            "trust_score": 0.95,
            "rationale": "Perfect score",
            "lineage": {
                "models": ["OpenAI GPT-4"],
                "sources": ["http://test.com"]
            }
        }
        mock_post.return_value = mock_response

        # Instantiate agent with mock keys
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "ANTHROPIC_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key"
        }):
            agent = AILEE_AI()

            # Inject mock clients if needed (since imports are mocked globally)
            agent.anthropic_client = MagicMock()
            agent.anthropic_client.messages.create.return_value.content[0].text = "Claude Answer"

            response = agent.process("Test Query")

            self.assertIn("Final Answer:", response)
            self.assertIn("Trusted Answer", response)
            self.assertIn("Trust Score: 0.95", response)

    @patch('ailee_ai.requests.post')
    def test_fallback_logic(self, mock_post):
        # Simulate API failure
        mock_post.side_effect = Exception("API Down")

        agent = AILEE_AI()

        # We rely on the internal mock generation (fallback when no keys)
        # So we don't set keys in env vars here

        response = agent.process("Test Query")

        self.assertIn("Final Answer:", response)
        # We can't easily capture stderr here without extra setup, but we know it printed.
        # Check that we got a result which implies fallback worked.
        self.assertIn("Trust Score:", response)
        self.assertIn("Lineage:", response)

if __name__ == '__main__':
    unittest.main()
