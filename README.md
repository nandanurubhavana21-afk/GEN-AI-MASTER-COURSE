RAG-Enhanced Generative AI-Based Resume and Job Description Analysis System

📌 About the Project

This project is an AI-powered resume analysis system that helps students and job seekers evaluate how well their resumes match specific job descriptions. It uses Generative AI, Retrieval-Augmented Generation (RAG), and ATS-based scoring to identify matching skills, highlight skill gaps, and provide personalized career recommendations.

🎯 Objectives

- Analyze resumes against job descriptions.
- Identify matching and missing technical skills.
- Generate an estimated ATS compatibility score.
- Provide personalized resume improvement suggestions.
- Generate interview questions based on job requirements.
- Recommend learning paths to improve employability.

🚀 Key Features

- Resume Upload: Upload resumes in PDF or DOCX format.
- Resume Parsing: Extract relevant information from uploaded documents.
- AI-Powered Analysis: Use Gemini AI to understand resume content and job requirements.
- RAG Integration: Retrieve relevant information using embeddings and a vector database.
- ATS Score: Calculate an estimated resume compatibility score.
- Skill Gap Analysis: Identify matching, partially matching, and missing skills.
- Resume Improvement: Generate suggestions to improve resume content.
- Interview Preparation: Generate job-related interview questions.
- Career Guidance: Provide personalized learning recommendations.

🛠️ Technologies Used

Category| Technologies
Frontend| React.js, HTML, CSS, JavaScript
Backend| Python, Flask
Generative AI| Google Gemini API
RAG Framework| LangChain
Vector Database| ChromaDB
Embeddings| Gemini Embedding Model
Document Processing| PyPDF, Tesseract OCR
Database| SQLite
Environment Management| Python dotenv

🏗️ System Architecture

1. User uploads a resume and enters a job description.
2. The Flask backend extracts text from the uploaded resume.
3. Extracted text is divided into smaller chunks.
4. Embeddings are generated and stored in ChromaDB.
5. The RAG pipeline retrieves relevant information.
6. Gemini AI analyzes the retrieved information and generates insights.
7. The Python ATS engine calculates an estimated compatibility score.
8. Results are displayed through the React frontend.

📊 Expected Output

- Resume analysis report.
- ATS compatibility score.
- Matching and missing skills.
- Resume improvement suggestions.
- Interview questions and answers.
- Personalized career learning roadmap.

💡 Advantages

- Reduces manual resume screening effort.
- Provides context-aware resume analysis.
- Helps candidates understand their skill gaps.
- Supports better resume preparation.
- Provides personalized career guidance.

🔮 Future Enhancements

- Integration with job portals.
- AI-powered mock interview system.
- Multiple resume comparison.
- Multilingual resume analysis.
- Advanced career recommendation system.

👨‍💻 Project Status

Academic project developed as part of the Generative AI subject. Features are being implemented and tested.

📄 License

This project is developed for educational and academic purposes.
