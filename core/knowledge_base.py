# knowledge base with chromadb
import chromadb
import json
import os
from typing import List, Dict

class KnowledgeBase:
    def __init__(self):
        self.client = chromadb.Client()

        # Define collection names
        self.PROCEDURAL_COLLECTION = "procedural_collection"
        self.DECLARATIVE_COLLECTION = "declarative_collection"
        self.UNIFIED_COLLECTION = "unified_collection"

        # ALWAYS delete existing collections (no conditional)
        # This ensures fresh data on every system start
        try:
            self.client.delete_collection(self.PROCEDURAL_COLLECTION)
            self.client.delete_collection(self.DECLARATIVE_COLLECTION)
            self.client.delete_collection("declarative_intent")
            self.client.delete_collection("declarative_semantic")
            self.client.delete_collection(self.UNIFIED_COLLECTION)
            print(" [KnowledgeBase] Cleared existing collections")
        except Exception as e:
            # Collections don't exist yet (first run), which is fine
            pass

        # Create fresh collections
        self.procedural = self.client.get_or_create_collection(name=self.PROCEDURAL_COLLECTION)

        # Declarative now uses TWO collections for multi-stage matching
        self.declarative_intent = self.client.get_or_create_collection(name="declarative_intent")
        self.declarative_semantic = self.client.get_or_create_collection(name="declarative_semantic")
        self.declarative = self.declarative_semantic  # Backward compatibility

        self.unified = self.client.get_or_create_collection(name=self.UNIFIED_COLLECTION)

        # Load data from JSON files
        self._load_data()
        print(" [KnowledgeBase] Fresh data loaded successfully")

    def _load_data(self):
        print(" [KnowledgeBase] Loading data into ChromaDB...")

        # Store raw declarative data for hybrid matching
        self.declarative_data = []

        # 1. Load Procedural Knowledge
        proc_path = "data/knowledge/procedural_api.json"
        if os.path.exists(proc_path):
            with open(proc_path, "r") as f:
                data = json.load(f)
                self._add_to_collection(self.procedural, data, prefix="proc")
                self._add_to_collection(self.unified, data, prefix="proc")
        else:
            print(f"Warning: {proc_path} not found.")

        # 2. Load Declarative Knowledge with Multi-Stage Embeddings
        decl_path = "data/knowledge/declarative_tasks.json"
        if os.path.exists(decl_path):
            with open(decl_path, "r") as f:
                data = json.load(f)
                self.declarative_data = data  # Store raw data
                self._add_declarative_multistage(data)
                self._add_to_collection(self.unified, data, prefix="decl")
        else:
            print(f"Warning: {decl_path} not found.")

    def _add_to_collection(self, collection, data_list: List[Dict], prefix: str):
        documents = []
        ids = []
        metadatas = []

        for i, item in enumerate(data_list):
            # Convert item to string representation for embedding
            doc_str = json.dumps(item)
            documents.append(doc_str)
            ids.append(f"{prefix}_{i}")
            metadatas.append({"source": prefix})

        if documents:
            collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def _create_semantic_representations(self, recipe: Dict) -> tuple[str, str]:
        # Skip comment entries
        if "_comment" in recipe:
            return None, None

        name = recipe.get("mission_name", "Unknown")
        keywords = recipe.get("intent_keywords", [])
        steps = recipe.get("logic_steps", [])

        # Stage 1: Intent-focused (keywords + name only, no JSON noise)
        intent_text = f"{name}. Keywords: {', '.join(keywords)}"
        # Example: "Pour. Keywords: pour, transfer liquid, decant, add"

        # Stage 2: Full semantic (add process description)
        # Take first 3 steps to keep it concise
        step_summary = " → ".join(steps[:3]) if steps else "No steps defined"
        semantic_text = f"Task: {name}. Purpose: {', '.join(keywords)}. Process: {step_summary}"
        # Example: "Task: Pour. Purpose: pour, transfer liquid, decant. Process: Pick up source → Pour into destination → Place source"

        return intent_text, semantic_text

    def _add_declarative_multistage(self, data_list: List[Dict]):
        intent_docs = []
        intent_ids = []
        intent_metadatas = []

        semantic_docs = []
        semantic_ids = []
        semantic_metadatas = []

        for i, recipe in enumerate(data_list):
            # Create semantic representations
            intent_text, semantic_text = self._create_semantic_representations(recipe)

            # Skip comment entries
            if intent_text is None:
                continue

            # Store original recipe JSON for retrieval
            recipe_json = json.dumps(recipe)

            # Add to intent collection (keyword-focused)
            intent_docs.append(intent_text)
            intent_ids.append(f"intent_{i}")
            intent_metadatas.append({"source": "decl", "recipe_json": recipe_json, "recipe_id": i})

            # Add to semantic collection (full context)
            semantic_docs.append(semantic_text)
            semantic_ids.append(f"semantic_{i}")
            semantic_metadatas.append({"source": "decl", "recipe_json": recipe_json, "recipe_id": i})

        # Add to collections
        if intent_docs:
            self.declarative_intent.add(
                documents=intent_docs,
                ids=intent_ids,
                metadatas=intent_metadatas
            )
            print(f" [KB] Added {len(intent_docs)} recipes to intent collection")

        if semantic_docs:
            self.declarative_semantic.add(
                documents=semantic_docs,
                ids=semantic_ids,
                metadatas=semantic_metadatas
            )
            print(f" [KB] Added {len(semantic_docs)} recipes to semantic collection")

    def query_unified(self, query_text: str, n_results: int = 3) -> str:
        results = self.unified.query(query_texts=[query_text], n_results=n_results)
        
        if not results['documents']:
            return ""
            
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        
        formatted_results = []
        for doc, meta in zip(docs, metas):
            source_label = "[Tool]" if meta['source'] == 'proc' else "[Mission]"
            formatted_results.append(f"{source_label} {doc}")
            
        return "\n\n".join(formatted_results)


    def query_declarative(self, query_text: str, n_results: int = 1) -> tuple[str, float]:
        print(f" [KB Multi-Stage] Query: '{query_text[:50]}...'")

        # Stage 1: Intent Keywords Query (fast, keyword-focused)
        print(" [KB] Stage 1: Querying intent keywords...")
        intent_results = self.declarative_intent.query(
            query_texts=[query_text],
            n_results=3,
            include=["metadatas", "distances"]
        )

        best_intent_dist = 100.0
        best_intent_recipe = None

        if intent_results['metadatas'] and intent_results['metadatas'][0]:
            best_intent_dist = intent_results['distances'][0][0]
            best_intent_recipe = intent_results['metadatas'][0][0].get('recipe_json')
            print(f" [KB] Stage 1 result: distance={best_intent_dist:.4f}")

            # If excellent match on intent keywords, return immediately
            if best_intent_dist < 0.8:
                print(f" [KB] ✓ Strong intent match (distance < 0.8), returning immediately")
                return best_intent_recipe, best_intent_dist

        # Stage 2: Full Semantic Query (contextual understanding)
        print(" [KB] Stage 2: Querying full semantic descriptions...")
        semantic_results = self.declarative_semantic.query(
            query_texts=[query_text],
            n_results=3,
            include=["metadatas", "distances"]
        )

        best_semantic_dist = 100.0
        best_semantic_recipe = None

        if semantic_results['metadatas'] and semantic_results['metadatas'][0]:
            best_semantic_dist = semantic_results['distances'][0][0]
            best_semantic_recipe = semantic_results['metadatas'][0][0].get('recipe_json')
            print(f" [KB] Stage 2 result: distance={best_semantic_dist:.4f}")

        # Compare both stages, take best result
        if best_intent_dist < best_semantic_dist:
            print(f" [KB] ✓ Using intent match (distance: {best_intent_dist:.4f})")
            return best_intent_recipe if best_intent_recipe else "", best_intent_dist
        else:
            print(f" [KB] ✓ Using semantic match (distance: {best_semantic_dist:.4f})")
            return best_semantic_recipe if best_semantic_recipe else "", best_semantic_dist

    def query_procedural(self, query_text: str, n_results: int = 1) -> str:
        results = self.procedural.query(query_texts=[query_text], n_results=n_results)
        if not results['documents']: return ""
        return "\n".join(results['documents'][0])

    def get_candidates(self, query_text: str, n_results: int = 3) -> list[tuple[str, float]]:
        results = self.declarative.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "distances"]
        )

        candidates = []
        if results['documents'] and results['documents'][0]:
            # Zip documents and distances together
            for doc, dist in zip(results['documents'][0], results['distances'][0]):
                candidates.append((doc, dist))

        return candidates