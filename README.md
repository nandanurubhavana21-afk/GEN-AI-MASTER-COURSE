# ResumeX AI — Advanced Resume & Job Description Analysis Platform

An enterprise-grade, Generative AI-powered Career Guidance and ATS Resume Optimization Platform. Powered by Python, Flask, Google Gemini AI (`gemini-3.8-flash`), SQLite, React.js, Tailwind CSS, `pypdf`, `python-docx`, Tesseract OCR, and ReportLab.

---

## 🚀 Newly Completed Upgrade Features

1. **Advanced ATS Resume Analysis & Scoring Methodology**:
   - Computes an Overall ATS Compatibility Score (0–100) alongside 5 dimensional sub-scores:
     - **Job Description Matching Score**
     - **Technical Skills Coverage Score**
     - **Experience Relevance & Seniority Score**
     - **Education & Credentials Alignment Score**
     - **Keyword Density & Parse-Readiness Score**
   - Readability index, bullet point structure review, action-verb density rating, and ATS formatting danger flags.
   - Transparent scoring methodology note clarifying scores as evidence-based heuristic estimates.

2. **Advanced 4-Way Skill Gap Analysis**:
   - **Matching Skills**: Demonstrated and verified skills.
   - **Missing Skills**: Deep-dive learning breakdown (Why it matters, 3-4 study topics, practical mini-projects, and Beginner/Intermediate/Advanced learning levels).
   - **Partially Matching Skills**: Adjacent or foundational skills that need deepening.
   - **Additional Skills**: Value-add candidate strengths beyond the JD.
   - **Evidentiary Integrity Guarantee**: Strict rule preventing AI from hallucinating skills not present in the candidate resume.

3. **AI Resume Section Rewriter (`/api/improve-section`)**:
   - Rewrites specific sections: Professional Summary, Career Objective, Technical Skills, Work Experience, Academic Projects, and Certifications.
   - Generates high-impact, ATS-friendly action bullets following Google XYZ formula without fabricating credentials.

4. **Multi-Job Description Comparison (`/api/compare-jobs`)**:
   - Compares 1 candidate resume simultaneously against 2 to 5 job openings.
   - Displays comparative match percentages, skill alignment matrices, and a **Recommended Best Match Opportunity** banner.

5. **Personalized Career Learning Roadmap**:
   - Sequential, step-by-step learning milestones to bridge missing skill gaps with learning goals, exercises, mini-projects, and time estimates (e.g., "1-2 weeks").

6. **Advanced Categorized Interview Preparation**:
   - Questions categorized into: Technical, HR, Scenario-Based, Project-Based, and Behavioral.
   - Difficulty ratings: Easy, Medium, Hard.
   - Suggested answer strategies, essential concepts to mention, and potential follow-up questions.

7. **User Authentication & Role-Based Access Control**:
   - Secure student registration, login, and session management.
   - Password hashing via PBKDF2/sha256 (`werkzeug.security`).
   - Signed JWT tokens (`pyjwt` / `python-jose`).
   - Private student data isolation (users only access their own uploaded resumes and reports).

8. **Analysis History & Saved Reports Database**:
   - SQLite database (`resume_analyzer.db`) designed for easy PostgreSQL migration.
   - Filter tabs: **All Analyses**, **Recent Analyses**, **Saved/Bookmarked Reports**.
   - Re-open previous analyses directly into the active dashboard with 1 click.

9. **Downloadable Professional PDF Reports (`/api/report/<id>/download`)**:
   - Polished multi-page PDF generation via ReportLab.
   - Includes executive summary, ATS dimensional breakdown table, skills matrix, missing skill study guides, personalized learning roadmap, and interview prep questions.

10. **Admin Dashboard & Monitoring (`/api/admin/stats`)**:
    - Platform telemetry: Total students, total analyses run, unique resumes, last 24h activity.
    - User management: Enable or disable student accounts.
    - System activity audit log (with privacy safeguards that protect raw resume text).

11. **Expanded File Format Support**:
    - **PDF**: Digital extraction via `pypdf`, with automated fallback to **Tesseract OCR** (`pdf2image` + `pytesseract`) for scanned resumes.
    - **DOCX**: Complete parsing of paragraphs, tables, and headers via `python-docx`.
    - Strict file validation: size limit (16 MB), extension check, and prompt-injection sanitization.

12. **Modern SaaS UI/UX**:
    - Interactive sidebar and top navigation.
    - Light / Dark theme toggle.
    - Drag-and-drop file upload with animated AI processing indicators.
    - Built-in zero-dependency React 18 interface in `templates/index.html` (served directly by Flask) AND standalone Vite React app in `frontend/`.

---

## 📂 Project Structure

```text
bhavass/
├── app.py                  # Main Flask application & enterprise REST endpoints
├── ai_service.py           # Gemini Flash AI engine (ATS, Rewriter, Multi-Job, Roadmap)
├── resume_parser.py        # Advanced parser (pypdf, Tesseract OCR, python-docx)
├── database.py             # SQLite data layer (Users, Analyses, MultiJob, ActivityLogs)
├── auth.py                 # JWT token management & RBAC decorators (@token_required)
├── report_generator.py     # Multi-page PDF report generation with ReportLab
├── resume_analyzer.db      # SQLite database file (auto-initialized)
├── requirements.txt        # Updated Python package specifications
├── .env.example            # Environment variable template
├── .env                    # Local credentials & config (GEMINI_API_KEY)
├── test_ai.py              # AI service unit tests
├── test_api.py             # Flask API integration tests
├── test_auth_db.py         # Authentication, database, and PDF report tests
├── templates/
│   └── index.html          # Full-featured responsive React 18 UI served directly
└── frontend/               # Standalone Vite + React 18 project
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── App.jsx         # Modular React UI
        ├── index.css
        └── main.jsx
```

---

## 🛠️ Installation & Execution Guide

### 1. Configure Environment
Check `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
PORT=5000
FLASK_DEBUG=True
JWT_SECRET_KEY=resumex-super-secret-jwt-key-2026
DEFAULT_ADMIN_EMAIL=admin@resumex.ai
DEFAULT_ADMIN_PASSWORD=Admin@123
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Backend & Web Application
```bash
python app.py
```
Open your browser at:
👉 **`http://localhost:5000`**

### 4. Default Test Credentials
- **Administrator**: `admin@resumex.ai` / `Admin@123`
- Or register any new student account directly in the UI modal.

---

## 🧪 Testing Results

All 17 tests across three test suites passed with 100% success:

```bash
pytest
```
```text
============================= test session starts =============================
platform win32 -- Python 3.12.0rc3, pytest-8.3.5
collected 17 items

test_ai.py ......                                                        [ 35%]
test_api.py ......                                                       [ 70%]
test_auth_db.py .....                                                    [100%]

============================= 17 passed in 4.27s ==============================
```

---

## ⚠️ Remaining Limitations

1. **Scanned PDF OCR Dependency**: OCR fallback requires local installation of Tesseract and Poppler on the system `PATH`. Digital PDFs and DOCX files require no external software.
2. **Gemini API Network & Quotas**: Real-time AI analysis requires an active internet connection and a valid API key with sufficient quota on Google AI Studio.
