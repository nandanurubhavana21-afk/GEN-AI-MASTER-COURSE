"""
app.py - Main Flask Application
Generative AI-Based Resume and Job Description Analysis System.
Enterprise Edition with Authentication, ATS Scoring, Multi-Job Comparison,
Section Rewriter, PDF Report Generation, and Admin Analytics.
"""

import os
import io
import re
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file, g
from flask_cors import CORS
from dotenv import load_dotenv

from resume_parser import parse_resume_document, extract_text_from_pdf
from ai_service import (
    analyze_resume_and_jd,
    analyze_career_goal,
    validate_job_description_intent,
    verify_gemini_api_key,
    improve_resume_section,
    compare_multiple_jobs,
    generate_academic_demo_analysis,
    generate_academic_demo_career_goal,
    DEFAULT_MODEL
)
from database import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    list_users,
    toggle_user_active_status,
    save_analysis,
    get_analysis_by_id,
    list_user_analyses,
    toggle_save_analysis,
    delete_analysis,
    save_multi_job_comparison,
    log_activity,
    get_admin_statistics
)
from auth import (
    hash_password,
    verify_password,
    generate_auth_token,
    get_current_user_from_request,
    token_required,
    admin_required
)
from report_generator import generate_pdf_report

# Load environment configuration
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with absolute asset paths for Vercel & serverless compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates")
)
CORS(app)  # Enable Cross-Origin requests for React frontend

# Maximum uploaded file size: 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "docx"}


def is_allowed_file(filename: str) -> bool:
    """Validates allowed resume extensions."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------------------------------------------------------
# Web Interface Route
# -------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serves the main application web interface."""
    template_path = os.path.join(app.template_folder, "index.html")
    return send_file(template_path)


# -------------------------------------------------------------------------
# System Health & API Configuration Endpoints
# -------------------------------------------------------------------------

def is_demo_mode_active() -> bool:
    """Returns True if Academic Offline Demo Mode is enabled or default on cloud deployments."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    has_real_key = bool(
        key and len(key) > 5 and
        key not in ("your_gemini_api_key_here", "your_actual_api_key", "your_verified_supported_model")
    )
    if has_real_key:
        return False
    if os.getenv("DEMO_MODE", "").lower() in ("true", "1"):
        return True
    if os.getenv("VERCEL") and not has_real_key:
        return True
    return key.upper() in ("DEMO", "DEMO_MODE", "ACADEMIC_DEMO")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Returns platform health status, model, and configuration."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    demo_active = is_demo_mode_active()
    has_valid_key = bool(
        api_key and len(api_key.strip()) > 5 and
        api_key not in ("your_gemini_api_key_here", "your_actual_api_key", "your_verified_supported_model")
    )
    is_configured = demo_active or has_valid_key

    return jsonify({
        "status": "healthy",
        "service": "RAG-Enhanced Generative AI-Based Resume and Job Description Analysis Platform",
        "model": "Academic Offline Heuristic Engine" if (demo_active and not has_valid_key) else os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        "embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
        "api_key_configured": is_configured,
        "demo_mode": demo_active,
        "supported_formats": ["pdf", "docx"],
        "supported_modes": ["job_description", "career_goal"],
        "rag_architecture": {
            "enabled": True,
            "vector_store": "ChromaDB v1.5.9",
            "chunker": "LangChain RecursiveCharacterTextSplitter",
            "distance_metric": "Cosine Similarity"
        }
    }), 200


@app.route("/api/config/demo-mode", methods=["POST"])
def toggle_demo_mode():
    """Toggles Academic Offline Demo Mode for presentations and testing without external API calls."""
    data = request.get_json(silent=True) or request.form.to_dict()
    enable = data.get("enable", True)
    if isinstance(enable, str):
        enable = enable.lower() in ("true", "1", "yes")

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        env_lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                env_lines = f.readlines()

        updated = False
        new_lines = []
        for line in env_lines:
            if line.startswith("DEMO_MODE="):
                new_lines.append(f"DEMO_MODE={'True' if enable else 'False'}\n")
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            new_lines.append(f"DEMO_MODE={'True' if enable else 'False'}\n")

        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except (OSError, IOError) as write_err:
            logger.warning(f"Note: Could not write to .env (expected in serverless/Vercel read-only filesystem): {write_err}")

        os.environ["DEMO_MODE"] = "True" if enable else "False"
        return jsonify({
            "success": True,
            "demo_mode": enable,
            "message": "Academic Demo Mode activated. Full ATS analysis is enabled offline." if enable else "Academic Demo Mode disabled."
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to toggle demo mode: {str(e)}"}), 500



@app.route("/api/config/key", methods=["POST"])
def configure_api_key():
    """
    Validates and updates the Gemini API Key in environment and saves to .env file.
    Does not expose sensitive credentials in responses.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    key = (data.get("gemini_api_key") or "").strip()
    model = (data.get("model") or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)).strip()

    if not key:
        return jsonify({"error": "Please provide a valid Gemini API key."}), 400

    # Probe and verify the API key
    verification = verify_gemini_api_key(key)
    if not verification.get("valid"):
        return jsonify({
            "error": verification.get("error", "The provided Gemini API key could not be verified.")
        }), 400

    active_model = verification.get("active_model", model)

    # Persist securely to .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        env_lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                env_lines = f.readlines()

        updated_key = False
        updated_model = False
        new_lines = []

        for line in env_lines:
            if line.startswith("GEMINI_API_KEY="):
                new_lines.append(f"GEMINI_API_KEY={key}\n")
                updated_key = True
            elif line.startswith("GEMINI_MODEL="):
                new_lines.append(f"GEMINI_MODEL={active_model}\n")
                updated_model = True
            else:
                new_lines.append(line)

        if not updated_key:
            new_lines.append(f"GEMINI_API_KEY={key}\n")
        if not updated_model:
            new_lines.append(f"GEMINI_MODEL={active_model}\n")

        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except (OSError, IOError) as write_err:
            logger.warning(f"Note: Could not write to .env (expected in serverless/Vercel read-only filesystem): {write_err}")

        # Update running process environment
        os.environ["GEMINI_API_KEY"] = key
        os.environ["GEMINI_MODEL"] = active_model

        return jsonify({
            "success": True,
            "message": "Gemini API Key successfully verified and saved to .env",
            "active_model": active_model
        }), 200

    except Exception as e:
        logger.error(f"Error saving to .env: {e}")
        # Even if file write fails, keep in process memory
        os.environ["GEMINI_API_KEY"] = key
        os.environ["GEMINI_MODEL"] = active_model
        return jsonify({
            "success": True,
            "message": "API key verified and active for current session.",
            "active_model": active_model
        }), 200


@app.route("/api/extract-text", methods=["POST"])
def extract_text():
    """
    Extracts text and structured entities from an uploaded PDF or DOCX without invoking AI.
    Returns contact details, segmented sections, and categorized skills for user preview.
    """
    if "resume" not in request.files:
        return jsonify({"error": "No resume file attached. Key must be 'resume'."}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not is_allowed_file(file.filename):
        return jsonify({"error": "Supported formats are PDF and DOCX only."}), 400

    try:
        file_bytes = file.read()
        parsed = parse_resume_document(file_bytes, file.filename)
        return jsonify({
            "success": True,
            "parsed": parsed
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in extract_text endpoint: {e}")
        return jsonify({"error": f"Failed to extract text: {str(e)}"}), 500


# -------------------------------------------------------------------------
# User Authentication Endpoints
# -------------------------------------------------------------------------

@app.route("/api/auth/register", methods=["POST"])
def register():
    """Registers a new student account."""
    data = request.get_json(silent=True) or request.form.to_dict()
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or len(name) < 2:
        return jsonify({"error": "Please enter a valid full name (minimum 2 characters)."}), 400
    if not email or "@" not in email or "." not in email:
        return jsonify({"error": "Please provide a valid email address."}), 400
    if not password or len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long."}), 400

    existing = get_user_by_email(email)
    if existing:
        return jsonify({"error": "An account with this email address already exists."}), 409

    pwd_hash = hash_password(password)
    user = create_user(name=name, email=email, password_hash=pwd_hash, role="student")
    if not user:
        return jsonify({"error": "Failed to create user account."}), 500

    token = generate_auth_token(user["id"], user["role"])
    log_activity(user["id"], "REGISTER", f"Student registered: {email}", request.remote_addr)

    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        }
    }), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    """Authenticates student or administrator."""
    data = request.get_json(silent=True) or request.form.to_dict()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return jsonify({"error": "Invalid email or password."}), 401

    if not user.get("is_active"):
        return jsonify({"error": "Account has been deactivated. Please contact administrator."}), 403

    token = generate_auth_token(user["id"], user["role"])
    log_activity(user["id"], "LOGIN", f"Successful login by {user['role']}: {email}", request.remote_addr)

    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        }
    }), 200


@app.route("/api/auth/me", methods=["GET"])
@token_required
def get_me():
    """Returns profile of currently authenticated user."""
    return jsonify({
        "user": g.current_user
    }), 200


# -------------------------------------------------------------------------
# Core Resume Analysis Endpoints (Mode A & Mode B)
# -------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze_resume():
    """
    Main Analysis Endpoint supporting:
      Mode A: Specific Job Description ATS Analysis
      Mode B: Career Goal & Role Readiness Guidance
    """
    try:
        current_user = get_current_user_from_request()
        user_id = current_user["id"] if current_user else None

        resume_text = ""
        filename = "Manual_Resume_Input.txt"
        extraction_method = "manual_text"
        structured_preview = None

        # Check for direct text input
        if "resume_text" in request.form and request.form["resume_text"].strip():
            resume_text = request.form["resume_text"].strip()
        elif request.is_json and request.json and "resume_text" in request.json:
            resume_text = request.json.get("resume_text", "").strip()

        # Check for file upload
        elif "resume" in request.files:
            file = request.files["resume"]
            if file.filename == "":
                return jsonify({"error": "No file chosen."}), 400

            if not is_allowed_file(file.filename):
                return jsonify({"error": "Only PDF and DOCX files are supported."}), 400

            filename = file.filename
            file_bytes = file.read()
            parsed = parse_resume_document(file_bytes, filename)
            resume_text = parsed["extracted_text"]
            extraction_method = parsed["extraction_method"]
            structured_preview = {
                "contact_info": parsed.get("contact_info"),
                "categorized_skills": parsed.get("categorized_skills"),
                "warnings": parsed.get("warnings")
            }

        if not resume_text or len(resume_text.strip()) < 20:
            return jsonify({
                "error": "Could not extract sufficient text from resume. Ensure the file contains readable content."
            }), 400

        # Retrieve Job Description / Career Goal
        job_description = ""
        job_title = "Target Position"
        company_name = "Target Organization"
        analysis_mode = "job_description"  # or 'career_goal'

        if "job_description" in request.form:
            job_description = request.form["job_description"].strip()
            job_title = request.form.get("job_title", job_title).strip() or job_title
            company_name = request.form.get("company_name", company_name).strip() or company_name
            analysis_mode = request.form.get("analysis_mode", "job_description").strip()
        elif request.is_json and request.json:
            job_description = request.json.get("job_description", "").strip()
            job_title = request.json.get("job_title", job_title).strip() or job_title
            company_name = request.json.get("company_name", company_name).strip() or company_name
            analysis_mode = request.json.get("analysis_mode", "job_description").strip()

        if not job_description or len(job_description) < 5:
            return jsonify({
                "error": "Please provide a target job description or career goal."
            }), 400

        # Intelligent Job Description Validation (Phase 4)
        intent = validate_job_description_intent(job_description)

        # If Mode A was requested but user provided a conversational career goal:
        if analysis_mode == "job_description" and intent["detected_mode"] == "career_goal":
            force = request.form.get("force_mode") == "true" or (request.is_json and request.json.get("force_mode"))
            if not force:
                return jsonify({
                    "error": (
                        "The input looks like a career goal ('" + job_description[:60] + "...'). "
                        "Please select 'Mode B: Career Goal Readiness Mode' for career path guidance, "
                        "or provide a complete company job description with responsibilities and requirements."
                    ),
                    "detected_mode": "career_goal",
                    "can_switch_mode": True
                }), 400

        # Execute Analysis: Try Live AI first (or mock in test), fallback to Academic Demo if key unconfigured
        analysis_result = None
        try:
            if analysis_mode == "career_goal":
                analysis_result = analyze_career_goal(
                    resume_text=resume_text,
                    career_goal=job_description
                )
                analysis_result["job_title"] = job_title if job_title != "Target Position" else job_description[:40]
                analysis_result["company_name"] = "Industry Benchmark Track"
            else:
                parsed_sections = parsed.get("sections") if ("parsed" in locals() and parsed) else None
                analysis_result = analyze_resume_and_jd(
                    resume_text=resume_text,
                    job_description=job_description,
                    structured_sections=parsed_sections
                )
                analysis_result["job_title"] = job_title
                analysis_result["company_name"] = company_name
        except ValueError as ve:
            if "GEMINI_API_KEY" in str(ve) and is_demo_mode_active():
                if analysis_mode == "career_goal":
                    analysis_result = generate_academic_demo_career_goal(
                        resume_text=resume_text,
                        career_goal=job_description
                    )
                    analysis_result["job_title"] = job_title if job_title != "Target Position" else job_description[:40]
                    analysis_result["company_name"] = "Industry Benchmark Track"
                else:
                    analysis_result = generate_academic_demo_analysis(
                        resume_text=resume_text,
                        job_description=job_description
                    )
                    analysis_result["job_title"] = job_title
                    analysis_result["company_name"] = company_name
            else:
                raise ve

        analysis_result["resume_filename"] = filename
        if structured_preview:
            analysis_result["extraction_preview"] = structured_preview

        # Persist to Database
        analysis_id = save_analysis(
            user_id=user_id,
            resume_filename=filename,
            resume_text=resume_text,
            job_description=job_description,
            analysis_data=analysis_result,
            job_title=analysis_result.get("job_title"),
            company_name=analysis_result.get("company_name")
        )

        log_activity(
            user_id,
            "ANALYZE_RESUME",
            f"Mode: {analysis_mode} | File '{filename}' | Role '{analysis_result.get('job_title')}' (Score: {analysis_result['match_score']}%)",
            request.remote_addr
        )

        return jsonify({
            "success": True,
            "analysis_id": analysis_id,
            "analysis_mode": analysis_mode,
            "extraction_method": extraction_method,
            "analysis": analysis_result
        }), 200

    except ValueError as ve:
        logger.warning(f"Validation error in analyze_resume: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in analyze_resume: {e}", exc_info=True)
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


# -------------------------------------------------------------------------
# AI Resume Improvement Module (Section Rewriting)
# -------------------------------------------------------------------------

@app.route("/api/improve-section", methods=["POST"])
def improve_section():
    """
    Improves a specific resume section (Summary, Experience, Projects, etc.)
    using ATS action verbs and metrics without fabricating details.
    """
    try:
        data = request.get_json() or request.form.to_dict()
        section_name = (data.get("section_name") or "summary").strip()
        current_text = (data.get("current_text") or "").strip()
        job_description = (data.get("job_description") or "").strip()

        if not current_text or len(current_text) < 10:
            return jsonify({"error": "Please provide your current section content (minimum 10 characters)."}), 400

        result = improve_resume_section(
            section_name=section_name,
            current_text=current_text,
            job_description=job_description
        )

        current_user = get_current_user_from_request()
        if current_user:
            log_activity(current_user["id"], "IMPROVE_SECTION", f"Optimized section: {section_name}", request.remote_addr)

        return jsonify({
            "success": True,
            "section": section_name,
            "result": result
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in improve_section: {e}")
        return jsonify({"error": f"Failed to improve section: {str(e)}"}), 500


# -------------------------------------------------------------------------
# Multi-Job Description Comparison
# -------------------------------------------------------------------------

@app.route("/api/compare-jobs", methods=["POST"])
def compare_jobs():
    """
    Compares 1 resume against 2 to 5 job descriptions simultaneously.
    Ranks best fit role and highlights matching/missing skills across all.
    """
    try:
        current_user = get_current_user_from_request()
        user_id = current_user["id"] if current_user else None

        resume_text = ""
        filename = "Comparison_Resume.txt"

        if "resume_text" in request.form and request.form["resume_text"].strip():
            resume_text = request.form["resume_text"].strip()
        elif request.is_json and request.json and "resume_text" in request.json:
            resume_text = request.json.get("resume_text", "").strip()
        elif "resume" in request.files:
            file = request.files["resume"]
            filename = file.filename
            file_bytes = file.read()
            parsed = parse_resume_document(file_bytes, filename)
            resume_text = parsed["extracted_text"]

        if not resume_text or len(resume_text.strip()) < 20:
            return jsonify({"error": "Valid resume text or document required."}), 400

        # Retrieve jobs array
        jobs = []
        if request.is_json and request.json and "jobs" in request.json:
            jobs = request.json.get("jobs", [])
        elif "jobs_json" in request.form:
            import json
            jobs = json.loads(request.form["jobs_json"])

        if not jobs or len(jobs) < 2:
            return jsonify({"error": "Please provide at least 2 job descriptions to compare."}), 400

        result = compare_multiple_jobs(resume_text, jobs)

        # Save record
        cid = save_multi_job_comparison(
            user_id=user_id,
            resume_filename=filename,
            comparisons=result.get("comparisons", []),
            best_match=result.get("best_match", {})
        )

        log_activity(user_id, "MULTI_JOB_COMPARISON", f"Compared resume '{filename}' against {len(jobs)} jobs", request.remote_addr)

        return jsonify({
            "success": True,
            "comparison_id": cid,
            "result": result
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in compare_jobs: {e}")
        return jsonify({"error": f"Multi-job comparison failed: {str(e)}"}), 500


# -------------------------------------------------------------------------
# Analysis History & Saved Reports Endpoints
# -------------------------------------------------------------------------

@app.route("/api/history", methods=["GET"])
@token_required
def get_user_history():
    """Lists saved or recent analyses for the authenticated user."""
    filter_type = request.args.get("filter", "all")
    analyses = list_user_analyses(user_id=g.current_user["id"], filter_type=filter_type)
    return jsonify({
        "success": True,
        "count": len(analyses),
        "analyses": analyses
    }), 200


@app.route("/api/history/<int:analysis_id>", methods=["GET"])
@token_required
def get_single_analysis(analysis_id: int):
    """Retrieves full details of a previously saved analysis."""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        return jsonify({"error": "Analysis not found."}), 404

    if analysis.get("user_id") != g.current_user["id"] and g.current_user["role"] != "admin":
        return jsonify({"error": "Access denied to this analysis record."}), 403

    return jsonify({
        "success": True,
        "analysis": analysis
    }), 200


@app.route("/api/history/<int:analysis_id>/toggle-save", methods=["POST"])
@token_required
def toggle_save(analysis_id: int):
    """Toggles bookmark status of an analysis."""
    new_state = toggle_save_analysis(analysis_id, g.current_user["id"])
    if new_state is None:
        return jsonify({"error": "Analysis not found or access denied."}), 404

    return jsonify({
        "success": True,
        "is_saved": new_state
    }), 200


@app.route("/api/history/<int:analysis_id>", methods=["DELETE"])
@token_required
def delete_single_analysis(analysis_id: int):
    """Deletes an analysis report from the user's history."""
    deleted = delete_analysis(analysis_id, g.current_user["id"])
    if not deleted:
        return jsonify({"error": "Analysis not found or unauthorized."}), 404

    return jsonify({"success": True, "message": "Analysis deleted successfully."}), 200


# -------------------------------------------------------------------------
# Downloadable PDF Report Endpoint
# -------------------------------------------------------------------------

@app.route("/api/report/<int:analysis_id>/download", methods=["GET"])
def download_pdf_report(analysis_id: int):
    """Generates and streams a professional PDF report for a given analysis."""
    user = get_current_user_from_request()
    analysis = get_analysis_by_id(analysis_id)

    if not analysis:
        return jsonify({"error": "Report not found."}), 404

    if analysis.get("user_id"):
        if not user or (user["id"] != analysis["user_id"] and user["role"] != "admin"):
            return jsonify({"error": "Authentication required to download this report."}), 401

    try:
        pdf_stream = generate_pdf_report(analysis)
        clean_filename = f"Resume_Analysis_Report_{analysis_id}.pdf"

        if user:
            log_activity(user["id"], "DOWNLOAD_REPORT", f"Downloaded PDF report #{analysis_id}", request.remote_addr)

        return send_file(
            pdf_stream,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=clean_filename
        )
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500


# -------------------------------------------------------------------------
# Administrator Dashboard & Monitoring
# -------------------------------------------------------------------------

@app.route("/api/admin/stats", methods=["GET"])
@admin_required
def admin_stats():
    """Returns platform overview statistics, trends, and recent audit activity."""
    stats = get_admin_statistics()
    return jsonify({
        "success": True,
        "statistics": stats
    }), 200


@app.route("/api/admin/users", methods=["GET"])
@admin_required
def admin_users_list():
    """Returns list of registered student accounts for administrative management."""
    users = list_users(limit=150)
    return jsonify({
        "success": True,
        "users": users
    }), 200


@app.route("/api/admin/users/<int:user_id>/toggle-status", methods=["POST"])
@admin_required
def admin_toggle_user(user_id: int):
    """Enables or disables a student account."""
    if user_id == g.current_user["id"]:
        return jsonify({"error": "Cannot deactivate your own administrator account."}), 400

    new_status = toggle_user_active_status(user_id)
    if new_status is None:
        return jsonify({"error": "User not found."}), 404

    log_activity(
        g.current_user["id"],
        "ADMIN_USER_STATUS_CHANGE",
        f"Admin toggled active status for user #{user_id} to {new_status}",
        request.remote_addr
    )

    return jsonify({
        "success": True,
        "user_id": user_id,
        "is_active": new_status
    }), 200


# -------------------------------------------------------------------------
# Server Execution Entry Point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1")
    logger.info(f"Starting Resume & JD Analysis System on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
