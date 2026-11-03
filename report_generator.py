"""
report_generator.py - Professional PDF Report Generation Module.
Generates comprehensive ATS Analysis, Skill Gap, and Career Guidance PDF reports
using ReportLab.
"""

import io
from datetime import datetime
from typing import Dict, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)


def generate_pdf_report(analysis_data: Dict[str, Any]) -> io.BytesIO:
    """
    Generates a structured, beautifully styled multi-page PDF report.
    Returns an in-memory BytesIO stream containing the PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    COLOR_PRIMARY = colors.HexColor("#4338ca")    # Indigo-700
    COLOR_SECONDARY = colors.HexColor("#6366f1")  # Indigo-500
    COLOR_DARK = colors.HexColor("#0f172a")       # Slate-900
    COLOR_TEXT = colors.HexColor("#334155")       # Slate-700
    COLOR_MUTED = colors.HexColor("#64748b")      # Slate-500
    COLOR_SUCCESS = colors.HexColor("#059669")    # Emerald-600
    COLOR_DANGER = colors.HexColor("#e11d48")     # Rose-600
    COLOR_BG_LIGHT = colors.HexColor("#f8fafc")   # Slate-50

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=COLOR_PRIMARY,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=COLOR_MUTED,
        spaceAfter=12
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=COLOR_DARK,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=COLOR_TEXT
    )

    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=COLOR_DARK
    )

    story = []

    # 1. Header & Title Banner
    story.append(Paragraph("Resume & Job Description Analysis Report", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Powered by Google Gemini Flash",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_PRIMARY, spaceAfter=14))

    # 2. Executive Overview Meta Table
    job_title = analysis_data.get("job_title") or "Target Role"
    company_name = analysis_data.get("company_name") or "Target Company"
    filename = analysis_data.get("resume_filename") or "Resume Document"
    match_score = int(analysis_data.get("match_score", 0))

    meta_data = [
        [
            Paragraph("<b>Target Role:</b>", body_style),
            Paragraph(f"{job_title} ({company_name})", body_style),
            Paragraph("<b>ATS Match Score:</b>", body_style),
            Paragraph(f"<b><font size=12 color='{COLOR_PRIMARY}'>{match_score}%</font></b>", body_style),
        ],
        [
            Paragraph("<b>Analyzed File:</b>", body_style),
            Paragraph(filename, body_style),
            Paragraph("<b>Analysis Date:</b>", body_style),
            Paragraph(datetime.now().strftime("%Y-%m-%d"), body_style),
        ]
    ]

    meta_table = Table(meta_data, colWidths=[80, 200, 110, 140])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 3. Job Requirements Summary
    summary_text = analysis_data.get("job_requirements_summary", "")
    if summary_text:
        story.append(Paragraph("1. Job Requirements Overview", section_heading))
        story.append(Paragraph(summary_text, body_style))
        story.append(Spacer(1, 10))

    # 4. ATS Scoring & Compatibility Breakdown
    ats = analysis_data.get("ats_breakdown", {})
    if ats:
        story.append(Paragraph("2. ATS Compatibility Breakdown", section_heading))
        story.append(Paragraph(
            "<i>Note: Scores are heuristic estimates derived from evidence-based keyword coverage, role seniority, and structural parseability.</i>",
            ParagraphStyle("Note", parent=body_style, fontSize=8, textColor=COLOR_MUTED, spaceAfter=6)
        ))

        score_rows = [
            [Paragraph("<b>Evaluation Dimension</b>", bold_body), Paragraph("<b>Score</b>", bold_body), Paragraph("<b>Assessment Level</b>", bold_body)],
            [Paragraph("Overall ATS Compatibility", body_style), Paragraph(f"<b>{ats.get('overall_ats_score', match_score)}%</b>", body_style), Paragraph("Overall candidate-job alignment", body_style)],
            [Paragraph("Job Description Matching", body_style), Paragraph(f"<b>{ats.get('jd_matching_score', 0)}%</b>", body_style), Paragraph("Direct alignment with responsibilities", body_style)],
            [Paragraph("Technical Skills Matching", body_style), Paragraph(f"<b>{ats.get('technical_skills_score', 0)}%</b>", body_style), Paragraph("Frameworks, languages, and technical tools", body_style)],
            [Paragraph("Experience Relevance", body_style), Paragraph(f"<b>{ats.get('experience_relevance_score', 0)}%</b>", body_style), Paragraph("Past projects and domain depth", body_style)],
            [Paragraph("Education & Certification Alignment", body_style), Paragraph(f"<b>{ats.get('education_relevance_score', 0)}%</b>", body_style), Paragraph("Degree and credential coverage", body_style)],
            [Paragraph("Keyword Optimization", body_style), Paragraph(f"<b>{ats.get('keyword_optimization_score', 0)}%</b>", body_style), Paragraph("Parse-readiness and natural phrasing", body_style)],
        ]

        score_table = Table(score_rows, colWidths=[200, 70, 260])
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 12))

    # 5. Skills Alignment Breakdown
    entry_roles = analysis_data.get("recommended_entry_roles", [])
    if entry_roles:
        story.append(Paragraph("Recommended Entry-Level / Stepping-Stone Roles", section_heading))
        for r in entry_roles:
            title = r.get("role_title", "Role") if isinstance(r, dict) else str(r)
            desc = r.get("description", "") if isinstance(r, dict) else ""
            story.append(Paragraph(f"• <b>{title}</b>: {desc}", body_style))
        story.append(Spacer(1, 10))

    story.append(Paragraph("3. Skills Gap & Competency Analysis", section_heading))

    matching_skills = analysis_data.get("matching_skills", [])
    partially_matching = analysis_data.get("partially_matching_skills", [])
    additional_skills = analysis_data.get("additional_skills", [])

    skills_summary = [
        [
            Paragraph("<b>Matching Skills (Demonstrated):</b>", bold_body),
            Paragraph(", ".join(matching_skills) if matching_skills else "None explicitly verified", body_style)
        ],
        [
            Paragraph("<b>Partially Matching Skills:</b>", bold_body),
            Paragraph(", ".join(partially_matching) if partially_matching else "None identified", body_style)
        ],
        [
            Paragraph("<b>Additional Candidate Strengths:</b>", bold_body),
            Paragraph(", ".join(additional_skills) if additional_skills else "None noted", body_style)
        ]
    ]

    skills_table = Table(skills_summary, colWidths=[180, 350])
    skills_table.setStyle(TableStyle([
        ("PADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), COLOR_BG_LIGHT),
    ]))
    story.append(skills_table)
    story.append(Spacer(1, 10))

    # 6. Missing Skills Details
    missing_skills = analysis_data.get("missing_skills", [])
    if missing_skills:
        story.append(Paragraph("<b>Detailed Missing Skills & Recommended Upgrades:</b>", bold_body))
        for m in missing_skills[:5]:
            if isinstance(m, dict):
                skill_name = m.get("name", "Skill")
                importance = m.get("importance", "")
                topics = ", ".join(m.get("recommended_topics", []))
                story.append(Paragraph(
                    f"• <b><font color='{COLOR_DANGER}'>{skill_name}</font></b>: {importance}<br/>"
                    f"&nbsp;&nbsp;<i>Key topics to study:</i> {topics}",
                    ParagraphStyle("MissingItem", parent=body_style, leftIndent=12, spaceAfter=4)
                ))
            else:
                story.append(Paragraph(f"• <font color='{COLOR_DANGER}'>{m}</font>", body_style))
        story.append(Spacer(1, 10))

    # 7. Resume Improvement Suggestions
    suggestions = analysis_data.get("improvement_suggestions", [])
    if suggestions:
        story.append(Paragraph("4. Actionable Resume Improvement Recommendations", section_heading))
        for idx, rec in enumerate(suggestions, 1):
            story.append(Paragraph(f"<b>{idx}.</b> {rec}", ParagraphStyle("Rec", parent=body_style, spaceAfter=4)))
        story.append(Spacer(1, 10))

    # 8. Learning Roadmap
    roadmap = analysis_data.get("learning_roadmap", [])
    if roadmap:
        story.append(Paragraph("5. Personalized Career Learning Roadmap", section_heading))
        roadmap_rows = [
            [Paragraph("<b>Step</b>", bold_body), Paragraph("<b>Target Skill & Goal</b>", bold_body), Paragraph("<b>Recommended Project</b>", bold_body), Paragraph("<b>Duration</b>", bold_body)]
        ]
        for step in roadmap:
            roadmap_rows.append([
                Paragraph(str(step.get("step", "-")), body_style),
                Paragraph(f"<b>{step.get('skill_name', '')}</b>: {step.get('objective', '')}", body_style),
                Paragraph(step.get("mini_project", step.get("practical_exercise", "")), body_style),
                Paragraph(step.get("estimated_duration", "1-2 weeks"), body_style),
            ])

        roadmap_table = Table(roadmap_rows, colWidths=[40, 240, 180, 70])
        roadmap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(roadmap_table)
        story.append(Spacer(1, 12))

    # 9. Interview Questions
    interview_q = analysis_data.get("interview_questions", [])
    if interview_q:
        story.append(Paragraph("6. Targeted Interview Preparation Questions", section_heading))
        for idx, q in enumerate(interview_q[:6], 1):
            category = q.get("category", "Technical")
            diff = q.get("difficulty", "Medium")
            question_text = q.get("question", "")
            strategy = q.get("suggested_answer", "")
            follow_up = q.get("follow_up_question", "")

            q_flow = [
                Paragraph(f"<b>Q{idx} [{category} • {diff}]:</b> {question_text}", bold_body),
                Paragraph(f"<b>Answer Strategy:</b> {strategy}", body_style)
            ]
            if follow_up:
                q_flow.append(Paragraph(f"<i>Potential Follow-up:</i> {follow_up}", ParagraphStyle("FollowUp", parent=body_style, textColor=COLOR_MUTED)))
            q_flow.append(Spacer(1, 6))
            story.append(KeepTogether(q_flow))

    # Build the PDF document
    doc.build(story)
    buffer.seek(0)
    return buffer

