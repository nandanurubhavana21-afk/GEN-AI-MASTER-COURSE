"""
database.py - Database Layer for Generative AI Resume & Job Description Analysis Platform.
Uses SQLite by default with an architecture designed to easily migrate to PostgreSQL.
Stores Users, Analyses, Multi-Job Comparisons, and Activity Logs.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from werkzeug.security import generate_password_hash

logger = logging.getLogger(__name__)

if os.getenv("VERCEL"):
    DB_PATH = os.getenv("DATABASE_PATH", "/tmp/resume_analyzer.db")
else:
    DB_PATH = os.getenv("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "resume_analyzer.db"))



def get_db_connection():
    """Returns a connection to the SQLite database with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initializes tables and default administrator account if needed."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'student', -- 'student' or 'admin'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    """)

    # 2. Resume Analyses Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            resume_filename TEXT NOT NULL,
            resume_text TEXT NOT NULL,
            job_title TEXT,
            company_name TEXT,
            job_description TEXT NOT NULL,
            match_score INTEGER NOT NULL,
            ats_breakdown TEXT,            -- JSON: sub-scores & formatting feedback
            matching_skills TEXT,          -- JSON list
            missing_skills TEXT,           -- JSON list of rich objects
            partially_matching_skills TEXT,-- JSON list
            additional_skills TEXT,        -- JSON list
            resume_strengths TEXT,         -- JSON list
            improvement_suggestions TEXT,  -- JSON list
            interview_questions TEXT,      -- JSON list of questions
            learning_roadmap TEXT,         -- JSON learning timeline & milestones
            job_requirements_summary TEXT, -- String
            is_saved INTEGER DEFAULT 0,    -- Bookmark/saved report flag
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
        )
    """)

    # 3. Multi-Job Comparisons Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS multi_job_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            resume_filename TEXT NOT NULL,
            comparisons TEXT NOT NULL,     -- JSON array of comparisons per job
            best_match TEXT,               -- JSON summary of best match
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
        )
    """)

    # 4. Activity Logs Table (for Admin Monitoring)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,          -- e.g. 'REGISTER', 'LOGIN', 'ANALYZE_RESUME', 'DOWNLOAD_REPORT'
            details TEXT,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
        )
    """)

    conn.commit()

    # Create default admin account if not already present
    admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@resumex.ai")
    admin_pass = os.getenv("DEFAULT_ADMIN_PASSWORD", "Admin@123")
    cursor.execute("SELECT id FROM users WHERE email = ?", (admin_email,))
    if not cursor.fetchone():
        hashed = generate_password_hash(admin_pass)
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, 'admin')",
            ("System Administrator", admin_email, hashed)
        )
        conn.commit()
        logger.info(f"Initialized default admin account: {admin_email}")

    # Create default student account for instant evaluation/demo
    student_email = "student@resumex.ai"
    student_pass = "Student@123"
    cursor.execute("SELECT id FROM users WHERE email = ?", (student_email,))
    if not cursor.fetchone():
        s_hashed = generate_password_hash(student_pass)
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, 'student')",
            ("Demo Student", student_email, s_hashed)
        )
        conn.commit()
        logger.info(f"Initialized default student account: {student_email}")

    conn.close()


# -------------------------------------------------------------------------
# User Operations
# -------------------------------------------------------------------------

def create_user(name: str, email: str, password_hash: str, role: str = "student") -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (name.strip(), email.strip().lower(), password_hash, role)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return get_user_by_id(user_id)
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    conn.close()
    return dict(user) if user else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    user = conn.execute("SELECT id, name, email, role, created_at, is_active FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None


def list_users(limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT id, name, email, role, created_at, is_active FROM users ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def toggle_user_active_status(user_id: int) -> Optional[bool]:
    conn = get_db_connection()
    row = conn.execute("SELECT is_active FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        conn.close()
        return None
    new_status = 0 if row["is_active"] == 1 else 1
    conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()
    return bool(new_status)


# -------------------------------------------------------------------------
# Analysis Operations
# -------------------------------------------------------------------------

def save_analysis(
    user_id: Optional[int],
    resume_filename: str,
    resume_text: str,
    job_description: str,
    analysis_data: Dict[str, Any],
    job_title: Optional[str] = None,
    company_name: Optional[str] = None
) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analyses (
            user_id, resume_filename, resume_text, job_title, company_name,
            job_description, match_score, ats_breakdown, matching_skills,
            missing_skills, partially_matching_skills, additional_skills,
            resume_strengths, improvement_suggestions, interview_questions,
            learning_roadmap, job_requirements_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        resume_filename,
        resume_text,
        job_title or analysis_data.get("job_title", "Target Role"),
        company_name or analysis_data.get("company_name", "Target Company"),
        job_description,
        int(analysis_data.get("match_score", 0)),
        json.dumps(analysis_data.get("ats_breakdown", {})),
        json.dumps(analysis_data.get("matching_skills", [])),
        json.dumps(analysis_data.get("missing_skills", [])),
        json.dumps(analysis_data.get("partially_matching_skills", [])),
        json.dumps(analysis_data.get("additional_skills", [])),
        json.dumps(analysis_data.get("resume_strengths", [])),
        json.dumps(analysis_data.get("improvement_suggestions", [])),
        json.dumps(analysis_data.get("interview_questions", [])),
        json.dumps(analysis_data.get("learning_roadmap", [])),
        analysis_data.get("job_requirements_summary", "")
    ))
    conn.commit()
    analysis_id = cursor.lastrowid
    conn.close()
    return analysis_id


def get_analysis_by_id(analysis_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    conn.close()
    if not row:
        return None

    data = dict(row)
    # Parse stored JSON fields safely
    for json_field in [
        "ats_breakdown", "matching_skills", "missing_skills",
        "partially_matching_skills", "additional_skills", "resume_strengths",
        "improvement_suggestions", "interview_questions", "learning_roadmap"
    ]:
        if data.get(json_field):
            try:
                data[json_field] = json.loads(data[json_field])
            except Exception:
                data[json_field] = []

    return data


def list_user_analyses(user_id: int, filter_type: str = "all", limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    query = "SELECT id, user_id, resume_filename, job_title, company_name, match_score, is_saved, created_at FROM analyses WHERE user_id = ?"
    params = [user_id]

    if filter_type == "saved":
        query += " AND is_saved = 1"
    elif filter_type == "recent":
        limit = min(limit, 10)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def toggle_save_analysis(analysis_id: int, user_id: int) -> Optional[bool]:
    conn = get_db_connection()
    row = conn.execute("SELECT is_saved FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id)).fetchone()
    if not row:
        conn.close()
        return None
    new_state = 0 if row["is_saved"] == 1 else 1
    conn.execute("UPDATE analyses SET is_saved = ? WHERE id = ? AND user_id = ?", (new_state, analysis_id, user_id))
    conn.commit()
    conn.close()
    return bool(new_state)


def delete_analysis(analysis_id: int, user_id: int) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# -------------------------------------------------------------------------
# Multi-Job Comparison Operations
# -------------------------------------------------------------------------

def save_multi_job_comparison(user_id: Optional[int], resume_filename: str, comparisons: List[Dict[str, Any]], best_match: Dict[str, Any]) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO multi_job_comparisons (user_id, resume_filename, comparisons, best_match)
        VALUES (?, ?, ?, ?)
    """, (
        user_id,
        resume_filename,
        json.dumps(comparisons),
        json.dumps(best_match)
    ))
    conn.commit()
    cid = cursor.lastrowid
    conn.close()
    return cid


# -------------------------------------------------------------------------
# Activity Logging & Admin Statistics
# -------------------------------------------------------------------------

def log_activity(user_id: Optional[int], action: str, details: Optional[str] = None, ip_address: Optional[str] = None):
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO activity_logs (user_id, action, details, ip_address) VALUES (?, ?, ?, ?)",
            (user_id, action, details, ip_address)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging activity: {e}")


def get_admin_statistics() -> Dict[str, Any]:
    conn = get_db_connection()
    
    total_users = conn.execute("SELECT COUNT(*) AS c FROM users WHERE role = 'student'").fetchone()["c"]
    total_analyses = conn.execute("SELECT COUNT(*) AS c FROM analyses").fetchone()["c"]
    total_resumes = conn.execute("SELECT COUNT(DISTINCT resume_filename) AS c FROM analyses").fetchone()["c"]
    
    # Analyses created in last 24h
    recent_analyses_count = conn.execute(
        "SELECT COUNT(*) AS c FROM analyses WHERE created_at >= datetime('now', '-1 day')"
    ).fetchone()["c"]

    # Recent activity logs (without private resumes)
    activity_rows = conn.execute("""
        SELECT a.id, a.action, a.details, a.created_at, u.name as user_name, u.email as user_email
        FROM activity_logs a
        LEFT JOIN users u ON a.user_id = u.id
        ORDER BY a.created_at DESC
        LIMIT 20
    """).fetchall()

    # Daily analysis trend (last 7 days)
    trend_rows = conn.execute("""
        SELECT date(created_at) as log_date, count(*) as count
        FROM analyses
        WHERE created_at >= datetime('now', '-7 days')
        GROUP BY date(created_at)
        ORDER BY log_date ASC
    """).fetchall()

    conn.close()

    return {
        "total_students": total_users,
        "total_analyses": total_analyses,
        "total_resumes_uploaded": total_resumes,
        "last_24h_analyses": recent_analyses_count,
        "recent_activities": [dict(r) for r in activity_rows],
        "daily_trends": [dict(r) for r in trend_rows]
    }


# Auto-initialize database on import
init_db()

