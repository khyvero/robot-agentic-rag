"""
Unit tests for Procedural Retrieval Service
"""

import pytest
from unittest.mock import Mock
from core.services.procedural_retrieval import ProceduralRetrievalService
from core.agents.types import RetrievalInput, RetrievalResult


class TestProceduralRetrievalService:
    """Test suite for Procedural Retrieval Service"""

    @pytest.fixture
    def mock_kb(self):
        # create a mock knowledgebase
        kb = Mock()
        return kb

    @pytest.fixture
    def retrieval_service(self, mock_kb):
        # create a proceduralretrievalservice with mock kb
        return ProceduralRetrievalService(mock_kb)

    def test_basic_retrieval(self, retrieval_service, mock_kb):
        # test basic procedural retrieval
        # Mock KB to return procedural APIs
        mock_kb.query_procedural.return_value = """
pick(target_obj_name: str)
Description: Picks up an object by name

place(destination_obj_name: str)
Description: Places object at destination
        """.strip()

        input_data = RetrievalInput(
            user_prompt="Pick up blood",
            intent_text=None,
            extracted_actions=["pick"],
            min_results=3
        )

        result = retrieval_service.retrieve(input_data)

        assert result.apis_retrieved > 0
        assert "pick" in result.procedural_context
        assert mock_kb.query_procedural.called

    def test_multi_query_retrieval(self, retrieval_service, mock_kb):
        # test multi-query retrieval strategy
        # Mock KB to return different results for each query
        mock_kb.query_procedural.side_effect = [
            "API 1: pick",
            "API 2: place",
            "API 3: pour"
        ]

        input_data = RetrievalInput(
            user_prompt="Pick up blood and pour it",
            intent_text="Recipe: Pick and pour",
            extracted_actions=["pick", "pour"],
            min_results=3
        )

        result = retrieval_service.retrieve(input_data)

        # Should have called query_procedural multiple times
        assert mock_kb.query_procedural.call_count >= 2
        assert result.apis_retrieved > 0

    def test_build_queries(self, retrieval_service):
        # test query building
        input_data = RetrievalInput(
            user_prompt="Pick up blood",
            intent_text="Recipe: Pick blood sample",
            extracted_actions=["pick", "place"],
            min_results=3
        )

        queries = retrieval_service._build_queries(input_data)

        assert len(queries) >= 2
        assert "Pick up blood" in queries[0]
        # Should have combined query with intent
        assert any("Recipe" in q for q in queries)
        # Should have action query
        assert any("pick" in q.lower() for q in queries)

    def test_build_queries_without_intent(self, retrieval_service):
        # test query building without intent
        input_data = RetrievalInput(
            user_prompt="Do something new",
            intent_text=None,
            extracted_actions=["pick"],
            min_results=3
        )

        queries = retrieval_service._build_queries(input_data)

        assert len(queries) >= 1
        assert "Do something new" in queries[0]

    def test_build_queries_without_actions(self, retrieval_service):
        # test query building without extracted actions
        input_data = RetrievalInput(
            user_prompt="Test prompt",
            intent_text="Test intent",
            extracted_actions=[],
            min_results=3
        )

        queries = retrieval_service._build_queries(input_data)

        # Should still have at least user prompt
        assert len(queries) >= 1

    def test_parse_apis_from_text(self, retrieval_service):
        # test parsing unique apis from text
        text = """
API 1: pick(obj)
Description: Picks object

API 2: place(obj)
Description: Places object
        """.strip()

        apis = retrieval_service._parse_apis_from_text(text)

        assert len(apis) > 0
        # Should have unique entries
        assert len(apis) == len(set(apis))

    def test_format_context(self, retrieval_service):
        # test formatting context string
        apis = {
            "pick(obj) - Picks object",
            "place(obj) - Places object",
            "pour(obj) - Pours liquid"
        }

        context = retrieval_service._format_context(apis)

        assert "API 1:" in context
        assert "pick" in context
        assert "place" in context
        assert "pour" in context

    def test_format_context_empty(self, retrieval_service):
        # test formatting empty context
        apis = set()

        context = retrieval_service._format_context(apis)

        assert "No procedural APIs found" in context

    def test_adaptive_expansion(self, retrieval_service, mock_kb):
        # test adaptive expansion when initial results are insufficient
        # First calls return limited results
        mock_kb.query_procedural.side_effect = [
            "API 1",  # First query
            "API 2",  # Second query (insufficient, only 2 total)
            "API 3\n\nAPI 4\n\nAPI 5"  # Expansion query
        ]

        input_data = RetrievalInput(
            user_prompt="Complex task",
            intent_text=None,
            extracted_actions=["pick"],
            min_results=3
        )

        result = retrieval_service.retrieve(input_data)

        # Should have triggered expansion
        assert mock_kb.query_procedural.call_count >= 3

    def test_retrieval_with_all_params(self, retrieval_service, mock_kb):
        # test retrieval with all parameters provided
        mock_kb.query_procedural.return_value = "pick(obj)\nplace(obj)\npour(obj)"

        input_data = RetrievalInput(
            user_prompt="Pick and pour blood",
            intent_text="Recipe: 1. Pick blood, 2. Pour into beaker",
            extracted_actions=["pick", "pour"],
            min_results=2
        )

        result = retrieval_service.retrieve(input_data)

        assert result.apis_retrieved >= 1
        assert "pick" in result.procedural_context.lower() or "pour" in result.procedural_context.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])