"""
test_auth_db.py - Tests for Database, Authentication, and Report Generator.
"""

import os
import unittest
from werkzeug.security import generate_password_hash, check_password_hash

import database
from database import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    save_analysis,
    get_analysis_by_id,
    list_user_analyses,
    toggle_save_analysis,
    delete_analysis,
    get_admin_statistics
)
from auth import (
    generate_auth_token,
    decode_auth_token,
    hash_password,
    verify_password
)
from report_generator import generate_pdf_report


class TestAuthAndDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database.init_db()

    def test_password_hashing(self):
        """Verify secure password hashing and verification."""
        password = "SecureStudentPass2026!"
        hashed = hash_password(password)
        self.assertNotEqual(password, hashed)
        self.assertTrue(verify_password(password, hashed))
        self.assertFalse(verify_password("WrongPassword", hashed))

    def test_jwt_token_generation_and_decode(self):
        """Verify JWT token encoding and decoding."""
        user_id = 999
        role = "student"
        token = generate_auth_token(user_id, role)
        self.assertIsInstance(token, str)

        payload = decode_auth_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["user_id"], user_id)
        self.assertEqual(payload["role"], role)

    def test_user_creation_and_retrieval(self):
        """Verify database user creation and retrieval by email and id."""
        email = f"test_student_{os.getpid()}@example.com"
        pwd_hash = hash_password("Secret123")
        user = create_user("Test Student", email, pwd_hash, role="student")

        self.assertIsNotNone(user)
        self.assertEqual(user["email"], email)

        fetched = get_user_by_email(email)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["id"], user["id"])

        by_id = get_user_by_id(user["id"])
        self.assertIsNotNone(by_id)
        self.assertEqual(by_id["name"], "Test Student")

    def test_analysis_persistence_and_retrieval(self):
        """Verify saving and retrieving an analysis with structured JSON fields."""
        email = f"analyzer_user_{os.getpid()}@example.com"
        user = create_user("Analysis User", email, hash_password("pass"), role="student")

        analysis_data = {
            "match_score": 88,
            "ats_breakdown": {
                "overall_ats_score": 88,
                "jd_matching_score": 90,
                "technical_skills_score": 85,
                "formatting_feedback": {"readability_score": 92}
            },
            "matching_skills": ["Python", "Flask", "React"],
            "missing_skills": [{"name": "Docker", "importance": "Deployment"}],
            "partially_matching_skills": ["SQL"],
            "additional_skills": ["Git"],
            "resume_strengths": ["Clear structure"],
            "improvement_suggestions": ["Add Docker project"],
            "interview_questions": [{"question": "Explain Flask", "category": "Technical"}],
            "learning_roadmap": [{"step": 1, "skill_name": "Docker", "objective": "Learn containers"}],
            "job_requirements_summary": "Full Stack developer position."
        }

        aid = save_analysis(
            user_id=user["id"],
            resume_filename="resume_test.pdf",
            resume_text="Sample text content for resume",
            job_description="Sample job description requiring Python Flask React Docker",
            analysis_data=analysis_data,
            job_title="Full Stack Engineer",
            company_name="Acme Corp"
        )
        self.assertIsInstance(aid, int)

        # Retrieve and verify parsed JSON fields
        retrieved = get_analysis_by_id(aid)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["match_score"], 88)
        self.assertEqual(retrieved["job_title"], "Full Stack Engineer")
        self.assertEqual(retrieved["ats_breakdown"]["overall_ats_score"], 88)
        self.assertEqual(len(retrieved["matching_skills"]), 3)
        self.assertEqual(retrieved["missing_skills"][0]["name"], "Docker")

        # Test toggle save
        saved_state = toggle_save_analysis(aid, user["id"])
        self.assertTrue(saved_state)

        # List user analyses
        user_list = list_user_analyses(user["id"])
        self.assertEqual(len(user_list), 1)

        # Test PDF Report Generation
        pdf_stream = generate_pdf_report(retrieved)
        self.assertIsNotNone(pdf_stream)
        pdf_bytes = pdf_stream.read()
        self.assertTrue(len(pdf_bytes) > 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

        # Test delete
        deleted = delete_analysis(aid, user["id"])
        self.assertTrue(deleted)
        self.assertIsNone(get_analysis_by_id(aid))

    def test_admin_statistics(self):
        """Verify admin statistics calculation."""
        stats = get_admin_statistics()
        self.assertIn("total_students", stats)
        self.assertIn("total_analyses", stats)
        self.assertIn("recent_activities", stats)
        self.assertIn("daily_trends", stats)


if __name__ == "__main__":
    unittest.main()

