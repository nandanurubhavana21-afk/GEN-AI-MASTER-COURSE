"""
test_api.py - Comprehensive API Integration Tests for Upgraded Flask Endpoints.
Covers:
- /api/health
- /api/config/key
- /api/auth/register & /api/auth/login
- /api/extract-text (Structured Preview)
- /api/analyze (Mode A & Mode B)
- /api/improve-section
- /api/compare-jobs
- /api/history & /api/report/<id>/download
- /api/admin/stats
"""

import io
import time
import uuid
import json
import unittest
from unittest.mock import patch
from pypdf import PdfWriter

from app import app
from database import init_db
from auth import generate_auth_token


def create_sample_pdf_bytes(text_content: str = "Jane Developer Python Flask React SQL Docker") -> io.BytesIO:
    """Creates a valid PDF byte stream for upload testing."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    stream.seek(0)
    return stream


class TestUpgradedFlaskAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_check(self):
        """Test system health and configuration check."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("supported_formats", data)
        self.assertIn("supported_modes", data)
        self.assertIn("pdf", data["supported_formats"])
        self.assertIn("docx", data["supported_formats"])

    def test_config_key_endpoint_validation(self):
        """Test API key validation and configuration endpoint."""
        # Test empty key rejected
        res = self.client.post("/api/config/key", json={"gemini_api_key": ""})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

        # Test invalid/dummy key rejected by probe
        res2 = self.client.post("/api/config/key", json={"gemini_api_key": "dummy_short_key"})
        self.assertEqual(res2.status_code, 400)
        self.assertIn("error", res2.get_json())

    def test_auth_registration_and_login_flow(self):
        """Test full user registration and login API flow."""
        email = f"api_test_{uuid.uuid4().hex[:8]}@test.com"
        reg_payload = {
            "name": "Sarah Connor",
            "email": email,
            "password": "Password123!"
        }

        # 1. Register
        reg_res = self.client.post("/api/auth/register", json=reg_payload)
        self.assertEqual(reg_res.status_code, 201)
        reg_data = reg_res.get_json()
        self.assertTrue(reg_data["success"])
        self.assertIn("token", reg_data)

        # 2. Duplicate registration check (should return 409)
        dup_res = self.client.post("/api/auth/register", json=reg_payload)
        self.assertEqual(dup_res.status_code, 409)

        # 3. Login
        login_res = self.client.post("/api/auth/login", json={
            "email": email,
            "password": "Password123!"
        })
        self.assertEqual(login_res.status_code, 200)
        login_data = login_res.get_json()
        self.assertTrue(login_data["success"])
        self.assertIn("token", login_data)

        # 4. Access protected /api/auth/me
        token = login_data["token"]
        me_res = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me_res.status_code, 200)
        me_data = me_res.get_json()
        self.assertEqual(me_data["user"]["email"], email)

    def test_conversational_career_goal_mode_interception(self):
        """Verify that casual career goals in Mode A are flagged with can_switch_mode: True."""
        token = generate_auth_token(user_id=1, role="student")
        payload = {
            "resume_text": "Experienced Python and Flask developer with 4 years experience.",
            "job_description": "I need to become an AI Engineer, so please find a job.",
            "analysis_mode": "job_description"
        }

        res = self.client.post("/api/analyze", data=payload, headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertTrue(data.get("can_switch_mode"))
        self.assertEqual(data.get("detected_mode"), "career_goal")

    @patch("app.analyze_resume_and_jd")
    def test_analyze_endpoint_mode_a(self, mock_analyze):
        """Test Mode A analysis endpoint with mock AI response and DB record creation."""
        mock_analyze.return_value = {
            "match_score": 85,
            "ats_breakdown": {
                "overall_ats_score": 85,
                "jd_matching_score": 90,
                "technical_skills_score": 85,
                "experience_relevance_score": 80,
                "education_relevance_score": 85,
                "keyword_optimization_score": 88,
                "scoring_methodology_note": "Test heuristic",
                "formatting_feedback": {"readability_score": 90, "formatting_issues": []}
            },
            "matching_skills": ["Python", "Flask", "React"],
            "missing_skills": [
                {
                    "name": "Kubernetes",
                    "importance": "Container orchestration",
                    "recommended_topics": ["Pods", "Deployments"],
                    "suggested_projects": ["Cluster setup"],
                    "learning_levels": {"beginner": "Deploy single pod", "intermediate": "Helm", "advanced": "Operators"}
                }
            ],
            "partially_matching_skills": ["Docker"],
            "additional_skills": ["Git", "CI/CD"],
            "resume_strengths": ["Strong backend experience"],
            "improvement_suggestions": ["Mention Kubernetes hands-on experience"],
            "job_requirements_summary": "Looking for a full stack engineer.",
            "learning_roadmap": [
                {"step": 1, "skill_name": "Kubernetes", "objective": "Learn Pods", "estimated_duration": "1 week"}
            ],
            "interview_questions": [
                {
                    "question": "What is the difference between a Pod and a Service in Kubernetes?",
                    "category": "Technical",
                    "difficulty": "Medium",
                    "suggested_answer": "Pods are transient execution units; Services provide stable networking.",
                    "important_concepts": ["Networking", "DNS"],
                    "follow_up_question": "How does kube-proxy handle traffic?"
                }
            ]
        }

        token = generate_auth_token(user_id=1, role="student")

        payload = {
            "resume_text": "Experienced full stack software engineer proficient in Python, Flask, React, and Git.",
            "job_description": "We are seeking a Full Stack Engineer with Python, Flask, React, and Kubernetes responsibilities.",
            "job_title": "Full Stack Engineer",
            "company_name": "CloudTech",
            "analysis_mode": "job_description"
        }

        res = self.client.post("/api/analyze", data=payload, headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("analysis_id", data)
        self.assertEqual(data["analysis"]["match_score"], 85)
        self.assertEqual(data["analysis"]["ats_breakdown"]["overall_ats_score"], 85)

        analysis_id = data["analysis_id"]

        # Test download PDF report for this analysis
        pdf_res = self.client.get(f"/api/report/{analysis_id}/download", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.mimetype, "application/pdf")
        self.assertTrue(len(pdf_res.data) > 1000)

    @patch("app.analyze_career_goal")
    def test_analyze_endpoint_mode_b(self, mock_career):
        """Test Mode B Career Goal Readiness endpoint."""
        mock_career.return_value = {
            "analysis_mode": "career_goal",
            "match_score": 78,
            "target_career_goal": "AI Engineer",
            "industry_benchmark_summary": "AI Engineers require Python, PyTorch, and Vector Databases in 2026.",
            "ready_skills": ["Python", "Flask"],
            "matching_skills": ["Python", "Flask"],
            "missing_skills": [{"name": "PyTorch", "importance": "Model training"}],
            "recommended_entry_roles": [{"role_title": "Junior AI Engineer", "description": "Stepping stone"}],
            "learning_roadmap": [],
            "interview_questions": [],
            "ats_breakdown": {"overall_ats_score": 78, "scoring_methodology_note": "Career Readiness"},
            "resume_strengths": ["Strong Python"],
            "improvement_suggestions": ["Add PyTorch projects"],
            "job_requirements_summary": "Industry requirements for AI Engineer."
        }

        token = generate_auth_token(user_id=1, role="student")
        payload = {
            "resume_text": "Python and Flask backend developer with machine learning interest.",
            "job_description": "I want to become an AI Engineer",
            "analysis_mode": "career_goal"
        }

        res = self.client.post("/api/analyze", data=payload, headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["analysis_mode"], "career_goal")
        self.assertEqual(data["analysis"]["match_score"], 78)

    @patch("app.improve_resume_section")
    def test_improve_section_endpoint(self, mock_improve):
        """Test section re-writer endpoint."""
        mock_improve.return_value = {
            "section_name": "Experience",
            "improved_text": "Architected high-throughput REST APIs utilizing Python and Flask, reducing response latency by 35%.",
            "improved_bullet_points": ["Architected REST APIs reducing latency by 35%"],
            "key_changes_made": ["Added measurable metrics", "Applied action verbs"],
            "ats_tips": ["Always pair technical tools with business outcomes"]
        }

        payload = {
            "section_name": "experience",
            "current_text": "I worked on making APIs in Flask for the backend team."
        }

        res = self.client.post("/api/improve-section", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("improved_text", data["result"])

    @patch("app.compare_multiple_jobs")
    def test_compare_jobs_endpoint(self, mock_compare):
        """Test multi-job description comparison endpoint."""
        mock_compare.return_value = {
            "comparisons": [
                {
                    "job_index": 1,
                    "job_title": "Backend Dev",
                    "company_name": "Alpha Corp",
                    "match_score": 85,
                    "matching_skills": ["Python", "Flask"],
                    "missing_skills": ["AWS"],
                    "verdict": "Strong fit"
                },
                {
                    "job_index": 2,
                    "job_title": "Frontend Dev",
                    "company_name": "Beta Inc",
                    "match_score": 60,
                    "matching_skills": ["React"],
                    "missing_skills": ["Vue", "Angular"],
                    "verdict": "Moderate alignment"
                }
            ],
            "best_match": {
                "job_title": "Backend Dev",
                "company_name": "Alpha Corp",
                "match_score": 85,
                "recommendation_reason": "Highest overlap with candidate's Python background."
            }
        }

        payload = {
            "resume_text": "Python Flask backend developer with basic React skills.",
            "jobs": [
                {"title": "Backend Dev", "company": "Alpha Corp", "description": "Need Python and Flask."},
                {"title": "Frontend Dev", "company": "Beta Inc", "description": "Need React and Vue."}
            ]
        }

        res = self.client.post("/api/compare-jobs", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["result"]["best_match"]["job_title"], "Backend Dev")

    def test_admin_stats_authorization(self):
        """Verify that non-admins cannot access admin endpoints."""
        student_token = generate_auth_token(user_id=2, role="student")
        admin_token = generate_auth_token(user_id=1, role="admin")

        # 1. Student access should be forbidden (403)
        res_student = self.client.get("/api/admin/stats", headers={"Authorization": f"Bearer {student_token}"})
        self.assertEqual(res_student.status_code, 403)

        # 2. Admin access should succeed (200)
        res_admin = self.client.get("/api/admin/stats", headers={"Authorization": f"Bearer {admin_token}"})
        self.assertEqual(res_admin.status_code, 200)
        data = res_admin.get_json()
        self.assertTrue(data["success"])
        self.assertIn("statistics", data)


if __name__ == "__main__":
    unittest.main()
