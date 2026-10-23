from xml.sax.saxutils import escape
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, session, make_response, redirect
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sentence_transformers import SentenceTransformer, util
import os
import re
import tempfile
import fitz
import json

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY", "dev-only-change-this-key"
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///resume_history.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_title = db.Column(db.String(200), nullable=False)
    match_score = db.Column(db.Float, nullable=False)
    keyword_coverage = db.Column(db.Float, nullable=False)
    matched_skills = db.Column(db.Text, nullable=True)
    missing_skills = db.Column(db.Text, nullable=True)
    suggestions = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime,
        default=db.func.current_timestamp()
    )

SKILL_ALIASES = {
    "java": ["java"],
    "python": ["python"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp"],
    "javascript": ["javascript", "java script"],
    "typescript": ["typescript"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "react": ["react", "react.js", "reactjs"],
    "angular": ["angular"],
    "node.js": ["node.js", "nodejs", "node js"],
    "flask": ["flask"],
    "spring boot": ["spring boot", "springboot", "spring-boot"],
    "sql": ["sql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo db"],
    "rest api": ["rest api", "rest apis", "restful api", "restful apis"],
    "hibernate": ["hibernate"],
    "jpa": ["jpa", "java persistence api"],
    "git": ["git"],
    "github": ["github", "git hub"],
    "data structures": ["data structures", "data structure"],
    "algorithms": ["algorithms", "algorithm"],
    "oop": ["oop", "object-oriented programming", "object oriented programming"],
    "machine learning": ["machine learning"],
    "deep learning": ["deep learning"],
    "docker": ["docker"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "communication": ["communication", "communication skills"],
    "problem-solving": ["problem-solving", "problem solving"]
}

model = SentenceTransformer("stsb-roberta-large")


def extract_text(pdf_path):
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text())
    return "\n".join(pages).strip()

def find_skills(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    found = set()

    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, text):
                found.add(skill)
                break

    return found

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    resume_file = request.files.get("resume")
    job_title = request.form.get("job_title", "").strip()
    job_desc = request.form.get("job_description", "").strip()
    
    job_title_lower = job_title.lower()

    if "full stack" in job_title_lower or "full-stack" in job_title_lower:
        target_role = "fullstack"
    elif "frontend" in job_title_lower or "front-end" in job_title_lower:
        target_role = "frontend"
    elif "backend" in job_title_lower or "back-end" in job_title_lower:
        target_role = "backend"
    elif "java" in job_title_lower:
        target_role = "java"
    else:
        target_role = "general"

    if not resume_file or not resume_file.filename:
        return render_template(
            "index.html", error="Please upload your resume PDF."
        ), 400

    if not resume_file.filename.lower().endswith(".pdf"):
        return render_template(
            "index.html", error="Only PDF files are supported."
        ), 400

    if not job_desc:
        return render_template(
            "index.html", error="Please enter the job description."
        ), 400

    # Use a temporary file so the uploaded resume is not kept in the project.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as temp_file:
            temp_path = temp_file.name
            resume_file.save(temp_path)

        resume_text = extract_text(temp_path)

        if not resume_text:
            return render_template(
                "index.html",
                error="No readable text found. Please upload a text-based PDF."
            ), 400

        embeddings = model.encode(
            [resume_text, job_desc],
            convert_to_tensor=True
        )

        similarity = util.pytorch_cos_sim(
            embeddings[0], embeddings[1]
        ).item()

        match_score = round(max(0, min(1, similarity)) * 100, 2)

        resume_skills = find_skills(resume_text)
        job_skills = find_skills(job_desc)

        matched_skills = sorted(resume_skills & job_skills)
        missing_skills = sorted(job_skills - resume_skills)

        # Coverage means the percentage of detected job-description skills
        # that also appear in the resume. It is not an ATS score.
        keyword_coverage = (
            round(len(matched_skills) / len(job_skills) * 100, 2)
            if job_skills else 0
        )
        
        suggestions = []

        if missing_skills:
            skills_text = ", ".join(missing_skills)
            suggestions.append(
                f"[HIGH PRIORITY] These job-description skills "
                f"were not detected in your resume: {skills_text}. "
                "Add them only if you genuinely know them and can "
                "demonstrate them with relevant examples."
            )

        

        resume_lower = resume_text.lower()

        technology_keywords = [
            "java", "python", "javascript", "react", "spring boot",
            "mysql", "sql", "html", "css", "node.js", "flask",
            "rest api", "hibernate", "git", "github"
        ]

        
        contribution_keywords = [
            "developed", "built", "implemented", "designed",
            "created", "integrated", "configured", "tested",
            "deployed", "optimized", "automated", "maintained",
            "managed", "analyzed", "documented", "debugged"
        ]

        feature_keywords = [
            "features", "authentication", "login", "registration",
            "dashboard", "search", "filtering", "database",
            "rest api", "api integration", "summarization",
            "document upload", "file upload", "document management",
            "cart", "wishlist", "coupon", "order processing",
            "admin panel", "data extraction", "authorization",
            "report generation", "sorting", "pagination"
        ]


        if len(resume_text.split()) < 150:
            suggestions.append(
                "Your resume text is quite short. Consider adding relevant "
                "projects, responsibilities and measurable achievements."
            )

        resume_lines = [
            line.strip().lower()
            for line in resume_text.splitlines()
            if line.strip()
        ]

        project_heading_pattern = re.compile(
            r"^(projects?|academic projects|"
            r"work experience|professional experience|"
            r"experience|internships?)\s*[:\-]?$"
        )

        has_project_section = any(
            project_heading_pattern.match(line)
            for line in resume_lines
        )
        
        project_start = next(
            (
                i for i, line in enumerate(resume_lines)
                if re.fullmatch(r"projects\s*[:\-]?", line)
            ),
            -1
        )

        project_end = len(resume_lines)

        if project_start != -1:
            for i in range(project_start + 1, len(resume_lines)):
                if re.fullmatch(
                    r"(education|skills|certificates?|certifications?|languages|achievements)\s*[:\-]?",
                    resume_lines[i]
                ):
                    project_end = i
                    break

        project_text = (
            " ".join(resume_lines[project_start:project_end])
            if project_start != -1
            else ""
        )
    

        has_technology_details = any(
            keyword in project_text
            for keyword in technology_keywords
        )

        has_contribution_details = any(
            keyword in project_text
            for keyword in contribution_keywords
        )

        has_feature_details = any(
            keyword in project_text
            for keyword in feature_keywords
        )

        if has_project_section:
            if not has_technology_details:
                suggestions.append(
                    "[MEDIUM PRIORITY] Mention the technologies "
                    "used in your projects, where applicable."
                )

            if not has_contribution_details:
                suggestions.append(
                    "[MEDIUM PRIORITY] Explain your individual "
                    "contribution using specific action words."
                )

            if not has_feature_details:
                suggestions.append(
                    "[MEDIUM PRIORITY] Describe the main features "
                    "or functionality of your projects."
                )

        if not has_project_section:
            suggestions.append(
                "[MEDIUM PRIORITY] Add a clearly titled Projects, "
                "Experience or Internship section if relevant to your background."
            )
        
        achievement_patterns = [
            r"\b(?:increased|improved|reduced|saved|achieved|decreased|boosted)\b",
            r"\b\d+(?:\.\d+)?\s*%",
            r"\b\d+\s*(?:users|customers|projects|hours|products|features|records|requests)\b",
        ]


        has_measurable_achievement = any(
            re.search(pattern, project_text)
            for pattern in achievement_patterns
        )

        if not has_measurable_achievement:
            suggestions.append(
                "[LOW PRIORITY] Strengthen your project descriptions "
                "with a verified result, such as the number of features "
                "implemented, test cases passed, or a measured performance "
                "improvement. Include numbers only when you can verify them."
            )

        role_focus = {
            "java": [
                "java", "oop", "spring boot", "rest api",
                "sql", "mysql", "hibernate", "jpa"
            ],
            "frontend": [
                "html", "css", "javascript", "react", "angular"
            ],
            "backend": [
                "java", "python", "flask", "spring boot",
                "rest api", "sql", "mysql", "mongodb"
            ],
            "fullstack": [
                "html", "css", "javascript", "react",
                "node.js", "java", "spring boot", "rest api",
                "sql", "mysql"
            ]
        }

        
        if target_role in role_focus:
            role_name = {
                "java": "Java Developer",
                "frontend": "Frontend Developer",
                "backend": "Backend Developer",
                "fullstack": "Full Stack Developer"
            }[target_role]

            missing_role_skills = [
                skill for skill in role_focus[target_role]
                if skill in job_skills and skill in missing_skills
            ]

            if missing_role_skills:
                skills_text = ", ".join(missing_role_skills)
                suggestions.append(
                    f"[MEDIUM PRIORITY] For this {role_name} role, "
                    f"your resume does not show these detected job skills: "
                    f"{skills_text}. Add them only if you genuinely know "
                    "them and can demonstrate relevant experience."
                )
            else:
                suggestions.append(
                    f"[MEDIUM PRIORITY] For this {role_name} role, "
                    "describe relevant project features, your individual "
                    "contribution, and the technologies you actually used."
                )
        

        if not job_skills:
            suggestions.append(
                "No supported skills were detected in the job description. "
                "Include specific technical skills to make the comparison "
                "more useful."
            )


        if not suggestions:
            suggestions.append(
                "Review your resume for role relevance, clear project details "
                "and accurate descriptions of your technical contributions."
            )

        report_data = {
            "job_title": job_title or "Job Role",
            "match_score": match_score,
            "keyword_coverage": keyword_coverage,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "matched_count": len(matched_skills),
            "missing_count": len(missing_skills),
            "job_skill_count": len(job_skills),
            "suggestions": suggestions
        }
        
        session["report"] = report_data

        

        return render_template("result.html", **report_data)

    except Exception:
        app.logger.exception("Resume analysis failed")
        return render_template(
            "index.html",
            error="Analysis failed. Please try another text-based PDF."
        ), 400

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                app.logger.warning("Could not remove temporary upload file.")


@app.route("/delete-analysis/<int:analysis_id>", methods=["POST"])
def delete_analysis(analysis_id):
    return redirect("/")


@app.route("/delete-selected-analyses", methods=["POST"])
def delete_selected_analyses():
    return redirect("/")


@app.route("/view-report/<int:analysis_id>")
def view_report(analysis_id):
    return redirect("/")


@app.route("/history")
def history():
    return redirect("/")


@app.route("/download-report")
def download_report():
    report_data = session.get("report")

    if not report_data:
        return "Please analyze a resume before downloading.", 400

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportSection",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#172554"),
        spaceBefore=14,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=7
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12
    ))

    story = []

    # Report title
    story.append(Paragraph(
        "RESUME ANALYSIS REPORT",
        styles["Title"]
    ))

    story.append(Paragraph(
        "Resume-to-job description analysis",
        styles["ReportSubtitle"]
    ))

    story.append(Paragraph(
        f"<b>Target Role:</b> "
        f"{escape(str(report_data['job_title']))}",
        styles["ReportBody"]
    ))

    story.append(Spacer(1, 12))

    # Score summary
    story.append(Paragraph(
        "SCORE SUMMARY",
        styles["ReportSection"]
    ))

    story.append(Paragraph(
        f"<b>Semantic Match Score:</b> "
        f"{report_data['match_score']}%",
        styles["ReportBody"]
    ))

    story.append(Paragraph(
        f"<b>Job Skill Coverage:</b> "
        f"{report_data['keyword_coverage']}%",
        styles["ReportBody"]
    ))

    story.append(Paragraph(
        f"<b>Matched Skills:</b> "
        f"{report_data['matched_count']}",
        styles["ReportBody"]
    ))

    story.append(Paragraph(
        f"<b>Skills to Review:</b> "
        f"{report_data['missing_count']}",
        styles["ReportBody"]
    ))

    story.append(Spacer(1, 8))

    # Explanation
    story.append(Paragraph(
        "UNDERSTANDING YOUR SCORES",
        styles["ReportSection"]
    ))

    story.append(Paragraph(
        "<b>Semantic Match Score:</b> Estimates how semantically "
        "similar your resume is to the job description. "
        "It is not an official ATS score.",
        styles["ReportBody"]
    ))

    story.append(Paragraph(
        "<b>Job Skill Coverage:</b> The percentage of detected "
        "supported job-description skills also found in your resume.",
        styles["ReportBody"]
    ))

    story.append(Paragraph(
        "<b>Important:</b> A missing skill may be absent from your "
        "resume or may not have been recognized by the detector. "
        "Include only skills you genuinely know.",
        styles["ReportBody"]
    ))

    # Matched skills
    story.append(Paragraph(
        "MATCHED SKILLS",
        styles["ReportSection"]
    ))

    matched = report_data.get("matched_skills", [])

    if matched:
        half = (len(matched) + 1) // 2
        left_skills = matched[:half]
        right_skills = matched[half:]

        from reportlab.platypus import Table, TableStyle

        skill_rows = []
        for i in range(half):
            left = "• " + escape(str(left_skills[i]))
            right = (
                "• " + escape(str(right_skills[i]))
                if i < len(right_skills) else ""
            )

            skill_rows.append([
                Paragraph(left, styles["ReportBody"]),
                Paragraph(right, styles["ReportBody"])
            ])

        skills_table = Table(
            skill_rows,
            colWidths=[245, 245]
        )

        skills_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))

        story.append(skills_table)
    else:
        story.append(Paragraph(
            "No matching skills detected.",
            styles["ReportBody"]
        ))

    # Skills to review
    story.append(Paragraph(
        "SKILLS TO REVIEW",
        styles["ReportSection"]
    ))

    missing = report_data.get("missing_skills", [])

    if missing:
        for skill in missing:
            story.append(Paragraph(
                "&#8226; " + escape(str(skill)),
                styles["ReportBody"]
            ))
    else:
        story.append(Paragraph(
            "No missing supported skills detected.",
            styles["ReportBody"]
        ))

    # Recommendations
    story.append(Paragraph(
        "RECOMMENDED IMPROVEMENTS",
        styles["ReportSection"]
    ))

    suggestions = report_data.get("suggestions", [])

   
    if suggestions:
        priority_styles = {
            "HIGH PRIORITY": {
                "color": "#B91C1C",
                "label": "HIGH PRIORITY",
            },
            "MEDIUM PRIORITY": {
                "color": "#C2410C",
                "label": "MEDIUM PRIORITY",
            },
            "LOW PRIORITY": {
                "color": "#15803D",
                "label": "LOW PRIORITY",
            },
        }

        for tip in suggestions:
            tip_text = str(tip)
            priority = None

            for label in priority_styles:
                marker = f"[{label}]"
                if tip_text.startswith(marker):
                    priority = label
                    tip_text = tip_text[len(marker):].strip()
                    break

            if priority:
                info = priority_styles[priority]
                label_html = (
                    f'<font color="{info["color"]}">'
                    f'<b>{info["label"]}</b></font>'
                )
                paragraph_text = (
                    f'&#8226; {label_html} — {escape(tip_text)}'
                )
            else:
                paragraph_text = "&#8226; " + escape(tip_text)

            story.append(Paragraph(
                paragraph_text,
                styles["ReportBody"]
            ))
    else:
        story.append(Paragraph(
            "No additional recommendations at this time.",
            styles["ReportBody"]
        ))

    try:
        pdf.build(story)

        response = make_response(buffer.getvalue())
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = (
            'attachment; filename="resume_analysis_report.pdf"'
        )

        return response

    except Exception:
        app.logger.exception("PDF generation failed")
        return "PDF generation failed. Check the VS Code terminal.", 500

    finally:
        buffer.close()

@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="File is too large. Please upload a PDF under 10 MB."
    ), 413


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        columns = {
            column["name"]
            for column in db.session.execute(
                db.text("PRAGMA table_info(analysis_history)")
            ).mappings()
        }

        new_columns = {
            "matched_skills": "TEXT",
            "missing_skills": "TEXT",
            "suggestions": "TEXT",
        }

        for column_name, column_type in new_columns.items():
            if column_name not in columns:
                db.session.execute(
                    db.text(
                        f"ALTER TABLE analysis_history "
                        f"ADD COLUMN {column_name} {column_type}"
                    )
                )

        db.session.commit()

    app.run(debug=True)

