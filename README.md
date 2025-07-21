# resume-matcher


Running the Application
bashpython app.py
The application will start and be available at http://localhost:7860
Using the System
1. Resume Analysis

Navigate to the "Resume Analysis & Job Recommendations" tab
Either paste your resume text or upload a PDF file
Click "Analyze Resume & Get Recommendations"
View your predicted job category, extracted skills, and personalized job recommendations

2. Job Search

Go to the "Job Search & Browse" tab
Enter keywords or select a category filter
Browse available job listings with detailed descriptions

3. Understanding Results

Match Score: Overall compatibility percentage
Skill Match: Percentage of required skills you possess
Content Match: Semantic similarity between your resume and job description
Matched Skills: Specific skills that align with the job requirements

📊 Supported Job Categories

Technical: Data Science, AI/ML, Software Development, UI/UX Design
Business: Sales, Marketing, Project Management, Finance & Accounting
Operations: Human Resources, Talent Acquisition, Leadership
Specialized: Sports & Fitness, Creative & Design, Content Creation


Technologies Used

Python 3.7+
Gradio - Web interface
scikit-learn - Machine learning models
spaCy - Natural language processing
PDFplumber - PDF text extraction
Pandas & NumPy - Data manipulation
TF-IDF - Text vectorization
