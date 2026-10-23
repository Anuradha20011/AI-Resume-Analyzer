# AI Resume Analyzer

A web application that analyzes a resume against a job description and provides insights into skill matching, resume relevance, and areas for improvement.

## Features

- Upload resumes in PDF format.
- Estimate resume and job description similarity using semantic analysis.
- Identify matched and missing skills.
- Generate role-specific improvement suggestions for Java, Frontend, Backend, and Full Stack roles.
- Download analysis reports as PDF files.
- View and revisit previous analyses.
- Search analyses by job title and filter them by score range.
- View reports and delete individual or multiple history records.

## Tech Stack

- **Language:** Python
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite
- **ORM:** Flask-SQLAlchemy
- **NLP:** Sentence Transformers
- **PDF Text Extraction:** PyMuPDF
- **PDF Report Generation:** ReportLab

## How It Works

1. Upload a resume in PDF format.
2. Enter the target job title and job description.
3. The application extracts text from the resume.
4. Semantic similarity and supported skills are analyzed.
5. View the match score, matched skills, missing skills, and improvement suggestions.
6. Download the PDF report or revisit results through analysis history.

## Installation and Setup

### Prerequisites

- Python
- pip package manager
- Git (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/Anuradha20011/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Open your browser and visit:

`http://127.0.0.1:5000`

The first run may take longer if the semantic model needs to download its files.

## Limitations

- The match score is an estimate, not an official ATS score or a guarantee of selection.
- Skill detection depends on supported keywords and the quality of extracted resume text.
- Recommendations should be reviewed, and resumes should include only genuine skills and experience.

## Future Improvements

- Expand skill detection and improve recommendation accuracy.
- Add automated tests.
- Implement secure user authentication.
- Improve deployment and security configuration.