# procedural retrieval service
from typing import List, Set
from core.agents.types import RetrievalInput, RetrievalResult
from core.knowledge_base import KnowledgeBase
from config.config import Config


class ProceduralRetrievalService:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def retrieve(self, input_data: RetrievalInput) -> RetrievalResult:
        print(" [Procedural Retrieval] Starting multi-query retrieval...")

        # Build multiple query strategies
        queries = self._build_queries(input_data)

        # Execute queries and collect unique results
        retrieved_apis = set()

        for i, query in enumerate(queries, 1):
            print(f" [Procedural Retrieval] Query {i}/{len(queries)}: '{query[:50]}...'")

            # Query the procedural knowledge base
            result_text = self.kb.query_procedural(query, n_results=input_data.min_results)

            # Parse and extract unique APIs (simple approach: split by function signatures)
            apis = self._parse_apis_from_text(result_text)
            retrieved_apis.update(apis)

            print(f"   > Found {len(apis)} APIs (total unique: {len(retrieved_apis)})")

        # If we haven't met minimum requirements, try expanding
        if len(retrieved_apis) < input_data.min_results and queries:
            print(f" [Procedural Retrieval] Expanding search (current: {len(retrieved_apis)})...")

            # Use first query with larger n_results
            expanded_text = self.kb.query_procedural(queries[0], n_results=20)
            expanded_apis = self._parse_apis_from_text(expanded_text)
            retrieved_apis.update(expanded_apis)

            print(f"   > After expansion: {len(retrieved_apis)} APIs")

        # Format results
        procedural_context = self._format_context(retrieved_apis)

        print(f" [Procedural Retrieval] Retrieved {len(retrieved_apis)} unique APIs")

        return RetrievalResult(
            procedural_context=procedural_context,
            apis_retrieved=len(retrieved_apis)
        )

    def _build_queries(self, input_data: RetrievalInput) -> List[str]:
        queries = []

        # Query 1: User prompt (always included)
        queries.append(input_data.user_prompt)

        # Query 2: User prompt + intent (if available)
        if input_data.intent_text:
            combined_query = f"{input_data.user_prompt} {input_data.intent_text}"
            queries.append(combined_query)

        # Query 3: Extracted actions (if available)
        if input_data.extracted_actions:
            # Join actions into a query
            actions_query = " ".join(input_data.extracted_actions)
            queries.append(actions_query)

            # Query 4: Expanded action keywords
            # This would require loading action keywords, simplified for now
            # Could be enhanced to load from procedural_api.json

        return queries

    def _parse_apis_from_text(self, text: str) -> Set[str]:
        apis = set()

        if text and text.strip():
            # Split by common delimiters (double newlines, or sections)
            sections = text.split("\n\n")
            for section in sections:
                if section.strip():
                    apis.add(section.strip())

        return apis

    def _format_context(self, apis: Set[str]) -> str:
        if not apis:
            return "No procedural APIs found."

        # Join all APIs with clear separation
        formatted_sections = []

        for i, api in enumerate(apis, 1):
            formatted_sections.append(f"API {i}:\n{api}")

        return "\n\n".join(formatted_sections)