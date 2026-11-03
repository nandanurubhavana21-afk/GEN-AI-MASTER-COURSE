"""
test_rag.py - Comprehensive Unit & Integration Tests for RAG Engine.
Validates:
- LangChain text chunking on resumes and JDs
- Local dense embedding generation & L2 normalization
- Cosine similarity computation
- ChromaDB vector indexing and Top-K retrieval
- Complete execute_rag_pipeline orchestration
"""

import unittest
import math
from rag_service import (
    chunk_resume_document,
    chunk_job_description,
    LocalDenseEmbedder,
    cosine_similarity,
    ChromaVectorIndex,
    execute_rag_pipeline,
    TextChunk
)


class TestRAGEngine(unittest.TestCase):

    def setUp(self):
        self.sample_resume = (
            "Alex Smith\n"
            "Email: alex.smith@example.com | Phone: +1-555-0199\n\n"
            "PROFESSIONAL SUMMARY\n"
            "Full Stack Engineer with 3+ years building high-throughput Python and React applications.\n\n"
            "TECHNICAL SKILLS\n"
            "Languages: Python, JavaScript, TypeScript, SQL\n"
            "Frameworks: Flask, Django, React, FastAPI\n"
            "Databases: PostgreSQL, Redis, MongoDB\n"
            "Cloud: Docker, Kubernetes, AWS, Git\n\n"
            "WORK EXPERIENCE\n"
            "Software Developer at TechCorp (2022 - Present)\n"
            "• Architected scalable REST microservices using Python and Flask handling 10k req/min.\n"
            "• Implemented PostgreSQL indexing and query optimization reducing latency by 35%.\n"
            "• Deployed containerized applications with Docker and automated CI/CD pipelines.\n\n"
            "PROJECTS\n"
            "AI Resume Screener (2023)\n"
            "• Built a full-stack web platform using Flask and React for automated ATS scoring.\n"
            "• Integrated vector similarity search for semantic candidate matching.\n\n"
            "EDUCATION\n"
            "B.Tech in Computer Science, University of Technology (2018 - 2022)"
        )

        self.sample_jd = (
            "Job Title: Senior Backend Engineer\n"
            "Company: CloudScale Systems\n\n"
            "Responsibilities:\n"
            "• Develop robust backend services and REST APIs in Python.\n"
            "• Maintain scalable database schemas in PostgreSQL and optimize complex queries.\n"
            "• Orchestrate containerized microservices using Docker and Kubernetes.\n\n"
            "Requirements:\n"
            "• 3+ years of professional software engineering experience.\n"
            "• Strong proficiency in Python, Flask, or FastAPI.\n"
            "• Experience with relational databases like PostgreSQL.\n"
            "• Familiarity with containerization (Docker) and cloud deployment."
        )

    def test_chunk_resume_document_sections(self):
        """Verify resume document chunking creates well-formed TextChunk objects with sections."""
        chunks = chunk_resume_document(self.sample_resume)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertIsInstance(c, TextChunk)
            self.assertTrue(c.chunk_id.startswith("resume_c"))
            self.assertEqual(c.doc_type, "resume")
            self.assertGreaterEqual(c.char_length, 15)
            self.assertIn("section", c.metadata)

    def test_chunk_job_description(self):
        """Verify JD chunking categorizes requirements and responsibilities."""
        chunks = chunk_job_description(self.sample_jd)
        self.assertGreater(len(chunks), 1)
        categories = set(c.section for c in chunks)
        self.assertTrue(any(cat in categories for cat in ["responsibilities", "requirements"]))

    def test_local_dense_embedder_normalization(self):
        """Verify local dense embeddings have 768 dimensions and are L2 normalized to unit sphere."""
        embedder = LocalDenseEmbedder(dimension=768)
        vec = embedder.embed_text("Python backend microservices with Flask and PostgreSQL.")
        self.assertEqual(len(vec), 768)
        norm = math.sqrt(sum(v * v for v in vec))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_cosine_similarity(self):
        """Verify cosine similarity calculation between vectors."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0, places=4)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0, places=4)

    def test_chroma_vector_index_add_and_query(self):
        """Verify indexing chunks and semantic query retrieval."""
        index = ChromaVectorIndex(session_id="test_unit_chroma")
        embedder = LocalDenseEmbedder(dimension=768)

        chunk1 = TextChunk("c1", "resume", "skills", "Python Flask REST APIs", 22, {})
        chunk2 = TextChunk("c2", "resume", "skills", "Graphic design and Adobe Photoshop", 34, {})

        embs = embedder.embed_batch([chunk1.text, chunk2.text])
        index.add_chunks([chunk1, chunk2], embs)

        query_vec = embedder.embed_text("Python Flask backend developer")
        results = index.query_similar(query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk.chunk_id, "c1")
        self.assertGreater(score, results[1][1])

    def test_execute_rag_pipeline_end_to_end(self):
        """Verify complete end-to-end RAG pipeline orchestration."""
        result = execute_rag_pipeline(
            resume_text=self.sample_resume,
            job_description=self.sample_jd,
            session_id="test_rag_pipeline"
        )

        self.assertIn("rag_context_prompt", result)
        self.assertIn("rag_metadata", result)
        meta = result["rag_metadata"]

        self.assertGreater(meta["total_resume_chunks"], 0)
        self.assertGreater(meta["total_jd_chunks"], 0)
        self.assertIn("average_semantic_match_percentage", meta)
        self.assertIn("semantic_matches", meta)
        self.assertGreater(len(meta["semantic_matches"]), 0)

        # Check structure of each semantic match
        first_match = meta["semantic_matches"][0]
        self.assertIn("requirement_text", first_match)
        self.assertIn("evidence_text", first_match)
        self.assertIn("similarity_percentage", first_match)
        self.assertIn("match_tier", first_match)


if __name__ == "__main__":
    unittest.main()

