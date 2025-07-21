import gradio as gr
import pdfplumber
import numpy as np
from scipy.spatial.distance import cosine
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import spacy
import warnings
warnings.filterwarnings('ignore')


try:
    nlp = spacy.load("en_core_web_md")
    USE_SPACY = True
    print("spaCy model loaded successfully")
except:
    print("spaCy model not found. Please install it with: python -m spacy download en_core_web_md")
    USE_SPACY = False
    # dummy nlp object to prevent errors
    class DummyNLP:
        def __call__(self, text):
            return DummyDoc(text)
    
    class DummyDoc:
        def __init__(self, text):
            self.text = text
        def similarity(self, other):
            return 0.5  
    
    nlp = DummyNLP()

class ResumeParser:
    """Enhanced Resume Parser with NLP techniques for all job types"""
    
    def __init__(self):
        #  stopwords for fallback
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once'
        }
    
    def extract_comprehensive_skills(self, text):
        """Extract both technical and non-technical skills"""
        text_lower = text.lower()
        found_skills = []
        
        skill_patterns = {
            # Technical Skills
            "Python": [r'\bpython\b', r'\bpy\b'],
            "Java": [r'\bjava\b'],
            "Machine Learning": [r'machine\s+learning', r'\bml\b'],
            "Deep Learning": [r'deep\s+learning', r'neural\s+networks?'],
            "Flask": [r'\bflask\b'],
            "Django": [r'\bdjango\b'],
            "React": [r'\breact\b', r'reactjs'],
            "JavaScript": [r'javascript', r'\bjs\b'],
            "Docker": [r'\bdocker\b'],
            "Kubernetes": [r'kubernetes', r'\bk8s\b'],
            "AWS": [r'\baws\b', r'amazon\s+web\s+services'],
            "Azure": [r'\bazure\b', r'microsoft\s+azure'],
            "TensorFlow": [r'tensorflow'],
            "PyTorch": [r'pytorch'],
            "Pandas": [r'\bpandas\b'],
            "Scikit-learn": [r'scikit-learn', r'sklearn'],
            
            # HR & Management Skills
            "Recruitment": [r'recruitment', r'talent\s+acquisition', r'hiring'],
            "Employee Engagement": [r'employee\s+engagement', r'team\s+engagement'],
            "Performance Management": [r'performance\s+management', r'appraisals'],
            "Training & Development": [r'training\s+and\s+development', r'l&d', r'learning'],
            "HR Policies": [r'hr\s+policies', r'policy\s+development'],
            "Compensation & Benefits": [r'compensation', r'benefits\s+administration'],
            "Leadership": [r'leadership', r'team\s+leadership', r'people\s+management'],
            "Project Management": [r'project\s+management', r'pmp', r'agile', r'scrum'],
            
            # Sales & Marketing Skills
            "Sales": [r'\bsales\b', r'selling', r'business\s+development'],
            "Digital Marketing": [r'digital\s+marketing', r'online\s+marketing'],
            "Social Media Marketing": [r'social\s+media\s+marketing', r'smm'],
            "Content Marketing": [r'content\s+marketing', r'content\s+creation'],
            "SEO": [r'\bseo\b', r'search\s+engine\s+optimization'],
            "CRM": [r'\bcrm\b', r'customer\s+relationship\s+management'],
            "Lead Generation": [r'lead\s+generation', r'lead\s+gen'],
            "Market Research": [r'market\s+research', r'market\s+analysis'],
            "Brand Management": [r'brand\s+management', r'branding'],
            "Customer Service": [r'customer\s+service', r'client\s+support'],
            
            # Finance & Accounting Skills
            "Financial Analysis": [r'financial\s+analysis', r'financial\s+modeling'],
            "Accounting": [r'accounting', r'bookkeeping'],
            "Budgeting": [r'budgeting', r'budget\s+planning'],
            "Auditing": [r'auditing', r'internal\s+audit'],
            "Tax Preparation": [r'tax\s+preparation', r'taxation'],
            "Risk Management": [r'risk\s+management', r'risk\s+assessment'],
            "Investment Analysis": [r'investment\s+analysis', r'portfolio\s+management'],
            "Excel": [r'microsoft\s+excel', r'\bexcel\b', r'spreadsheets'],
            "QuickBooks": [r'quickbooks', r'accounting\s+software'],
            
            # Sports & Fitness Skills
            "Athletic Training": [r'athletic\s+training', r'sports\s+training'],
            "Fitness Coaching": [r'fitness\s+coaching', r'personal\s+training'],
            "Sports Management": [r'sports\s+management', r'athletic\s+administration'],
            "Team Coaching": [r'team\s+coaching', r'coaching'],
            "Sports Analytics": [r'sports\s+analytics', r'performance\s+analysis'],
            "Physical Therapy": [r'physical\s+therapy', r'physiotherapy'],
            "Nutrition Planning": [r'nutrition\s+planning', r'diet\s+planning'],
            "Event Management": [r'event\s+management', r'tournament\s+organization'],
            
            # Creative & Design Skills
            "Graphic Design": [r'graphic\s+design', r'visual\s+design'],
            "UI/UX Design": [r'ui\s*\/?\s*ux', r'user\s+interface', r'user\s+experience'],
            "Photography": [r'photography', r'photo\s+editing'],
            "Video Editing": [r'video\s+editing', r'video\s+production'],
            "Adobe Creative Suite": [r'adobe\s+creative', r'photoshop', r'illustrator'],
            "Content Writing": [r'content\s+writing', r'copywriting'],
            
            # General Business Skills
            "Communication": [r'communication\s+skills', r'verbal\s+communication'],
            "Presentation": [r'presentation\s+skills', r'public\s+speaking'],
            "Negotiation": [r'negotiation', r'deal\s+closing'],
            "Problem Solving": [r'problem\s+solving', r'analytical\s+thinking'],
            "Time Management": [r'time\s+management', r'organization'],
            "Microsoft Office": [r'microsoft\s+office', r'ms\s+office', r'office\s+suite']
        }
        
        for skill, patterns in skill_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    found_skills.append(skill)
                    break
        
        return found_skills
    
    def extract_experience_years(self, text):
        """Extract years of experience using regex patterns"""
        patterns = [
            r'(\d+)\s*years?\s*of\s*experience',
            r'(\d+)\s*yrs?\s*experience',
            r'experience:\s*(\d+)\s*years?',
            r'(\d+)\+?\s*years?\s*in\s+\w+'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        
        return 0

class JobMatcher:
    """Advanced Job Matching System with ML Models using only spaCy"""
    
    def __init__(self):
        self.parser = ResumeParser()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text data using spaCy"""
        if USE_SPACY:
           
            doc = nlp(text)
            
            tokens = [token.lemma_.lower() for token in doc 
                     if not token.is_stop and not token.is_punct 
                     and not token.is_space and len(token.text) > 2]
            return ' '.join(tokens)
        else:
           
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            text = ' '.join(text.split())
            
            tokens = text.split()
            tokens = [word for word in tokens if word not in self.parser.stop_words and len(word) > 2]
            return ' '.join(tokens)
    
    def train_models(self):
        """Train models with sample data automatically"""
        if self.is_trained:
            return
            
        
        import json

        with open("sample_resumes.json",'r') as f:
            sample_resumes = json.load(f)

        resume_texts = []
        categories = []
        
        for category, resumes in sample_resumes.items():
            for resume in resumes:
                resume_texts.append(resume)
                categories.append(category)
        
       
        processed_texts = [self.preprocess_text(resume) for resume in resume_texts]
        X = self.vectorizer.fit_transform(processed_texts)
        y = self.label_encoder.fit_transform(categories)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)

        
        y_pred = self.classifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(
            y_test, y_pred,
            labels=range(len(self.label_encoder.classes_)),
            target_names=self.label_encoder.classes_,
            zero_division=0
        ))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        self.is_trained = True
    
    def calculate_similarity_scores(self, resume_text, job_descriptions):
        """Calculate similarity scores using spaCy"""
        if not USE_SPACY:
            
            resume_words = set(self.preprocess_text(resume_text).split())
            similarities = []
            
            for job_desc in job_descriptions:
                job_words = set(self.preprocess_text(job_desc).split())
                if len(resume_words | job_words) == 0:
                    similarity = 0.0
                else:
                    similarity = len(resume_words & job_words) / len(resume_words | job_words)
                similarities.append(similarity)
            
            return similarities
        
        
        resume_doc = nlp(resume_text)
        similarities = []

        for job_desc in job_descriptions:
            job_doc = nlp(job_desc)
            similarity = resume_doc.similarity(job_doc)
            similarities.append(similarity)

        return similarities
    
    def predict_job_category(self, resume_text):
        """Predict the most suitable job category for a resume"""
        if not self.is_trained:
            self.train_models()
        
        processed_text = self.preprocess_text(resume_text)
        text_vector = self.vectorizer.transform([processed_text])
        
        predicted_category = self.classifier.predict(text_vector)[0]
        predicted_category_name = self.label_encoder.inverse_transform([predicted_category])[0]
        
        prediction_proba = self.classifier.predict_proba(text_vector)[0]
        confidence = max(prediction_proba)
        
        return predicted_category_name, confidence
    
    def get_job_recommendations(self, resume_text, job_listings):
        """Get personalized job recommendations using spaCy"""
        if not self.is_trained:
            self.train_models()
        
        
        user_skills = self.parser.extract_comprehensive_skills(resume_text)
        
        job_scores = []
        
        for i, job in enumerate(job_listings):
            
            job_skills = job.get('required_skills', [])
            skill_overlap = len(set([s.lower() for s in user_skills]) & set([s.lower() for s in job_skills]))
            skill_score = skill_overlap / max(len(job_skills), 1) if job_skills else 0
            
            
            similarity = self.calculate_similarity_scores(resume_text, [job['description']])[0]
            
            
            combined_score = 0.6 * skill_score + 0.4 * similarity
            
            job_scores.append({
                'job_index': i,
                'job': job,
                'skill_score': skill_score,
                'similarity_score': similarity,
                'combined_score': combined_score,
                'matched_skills': list(set([s.lower() for s in user_skills]) & set([s.lower() for s in job_skills]))
            })
        
        
        job_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return job_scores


COMPREHENSIVE_JOBS = [
    # Technical Roles
    {
        "title": "Data Scientist",
        "company": "TechCorp Analytics",
        "location": "San Francisco, CA",
        "salary": "$120,000 - $150,000",
        "type": "Full-time",
        "description": "Analyze large datasets, build predictive models using machine learning algorithms, work with Python, pandas, scikit-learn, and statistical analysis. Experience with data visualization and SQL required. Join our team to solve complex business problems with data-driven insights.",
        "required_skills": ["Python", "Pandas", "Scikit-learn", "Machine Learning", "Excel"],
        "experience_required": "2-4 years",
        "category": "Data Science"
    },
    {
        "title": "Machine Learning Engineer", 
        "company": "AI Innovations Inc",
        "location": "New York, NY",
        "salary": "$130,000 - $160,000",
        "type": "Full-time",
        "description": "Design and implement machine learning systems, deep learning models, work with TensorFlow, PyTorch, MLOps, and cloud platforms. Build scalable AI applications that serve millions of users. Experience with model deployment and monitoring required.",
        "required_skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch"],
        "experience_required": "3-5 years",
        "category": "AI/ML"
    },
    {
        "title": "Backend Developer",
        "company": "WebSolutions Ltd",
        "location": "Austin, TX",
        "salary": "$95,000 - $125,000",
        "type": "Full-time",
        "description": "Develop server-side applications using Python Flask/Django, work with databases like PostgreSQL, implement REST APIs, and containerization with Docker. Build robust and scalable backend systems for web applications.",
        "required_skills": ["Python", "Flask", "Django", "Excel", "Docker"],
        "experience_required": "2-4 years", 
        "category": "Software Development"
    },
    {
        "title": "Frontend Developer",
        "company": "Digital Creative Agency",
        "location": "Seattle, WA",
        "salary": "$85,000 - $115,000",
        "type": "Full-time",
        "description": "Build responsive web applications using React, JavaScript, HTML5, CSS3. Experience with modern frontend frameworks and responsive design principles. Create beautiful and intuitive user interfaces.",
        "required_skills": ["React", "JavaScript", "UI/UX Design", "Communication"],
        "experience_required": "2-3 years",
        "category": "Software Development"
    },
    
    # HR & Management Roles
    {
        "title": "HR Manager",
        "company": "Global Enterprise Corp",
        "location": "Chicago, IL",
        "salary": "$85,000 - $110,000",
        "type": "Full-time",
        "description": "Lead human resources operations including recruitment, employee engagement, performance management, training and development, and policy implementation. Drive HR initiatives that support business growth and employee satisfaction.",
        "required_skills": ["Recruitment", "Employee Engagement", "Performance Management", "Leadership", "Communication"],
        "experience_required": "5-7 years",
        "category": "Human Resources"
    },
    {
        "title": "Talent Acquisition Specialist",
        "company": "Startup Hub Ventures",
        "location": "Boston, MA",
        "salary": "$65,000 - $85,000",
        "type": "Full-time",
        "description": "Focus on recruiting top talent, managing hiring processes, conducting interviews, and building talent pipelines for various departments. Work closely with hiring managers to understand staffing needs.",
        "required_skills": ["Recruitment", "Communication", "Negotiation", "CRM", "Time Management"],
        "experience_required": "2-4 years",
        "category": "Human Resources"
    },
    {
        "title": "Project Manager",
        "company": "Construction Solutions Inc",
        "location": "Denver, CO",
        "salary": "$90,000 - $120,000",
        "type": "Full-time",
        "description": "Lead cross-functional teams, manage project timelines, budgets, and deliverables. Experience with Agile methodologies and stakeholder management. Ensure projects are delivered on time and within budget.",
        "required_skills": ["Project Management", "Leadership", "Communication", "Time Management", "Problem Solving"],
        "experience_required": "4-6 years",
        "category": "Management"
    },
    
    # Sales & Marketing Roles
    {
        "title": "Digital Marketing Manager",
        "company": "E-commerce Giant",
        "location": "Los Angeles, CA",
        "salary": "$80,000 - $105,000",
        "type": "Full-time",
        "description": "Develop and execute digital marketing campaigns, manage social media presence, SEO optimization, content marketing, and analyze marketing metrics. Drive online customer acquisition and brand awareness.",
        "required_skills": ["Digital Marketing", "Social Media Marketing", "SEO", "Content Marketing", "Market Research"],
        "experience_required": "3-5 years",
        "category": "Marketing"
    },
    {
        "title": "Sales Executive",
        "company": "Software Solutions Corp",
        "location": "Miami, FL",
        "salary": "$70,000 - $95,000 + Commission",
        "type": "Full-time",
        "description": "Drive revenue growth through lead generation, client relationship management, sales presentations, and achieving monthly targets. Build and maintain strong relationships with enterprise clients.",
        "required_skills": ["Sales", "Lead Generation", "CRM", "Negotiation", "Customer Service"],
        "experience_required": "2-4 years",
        "category": "Sales"
    },
    {
        "title": "Brand Manager",
        "company": "Consumer Goods International",
        "location": "Atlanta, GA",
        "salary": "$85,000 - $110,000",
        "type": "Full-time",
        "description": "Develop brand strategy, manage brand positioning, coordinate marketing campaigns, and analyze brand performance metrics. Lead brand initiatives across multiple product lines.",
        "required_skills": ["Brand Management", "Digital Marketing", "Market Research", "Communication", "Project Management"],
        "experience_required": "4-6 years",
        "category": "Marketing"
    },
    
    # Finance & Accounting Roles
    {
        "title": "Financial Analyst",
        "company": "Investment Banking Corp",
        "location": "New York, NY",
        "salary": "$75,000 - $95,000",
        "type": "Full-time",
        "description": "Perform financial modeling, budgeting, forecasting, investment analysis, and prepare financial reports for management decision-making. Support strategic business initiatives with financial insights.",
        "required_skills": ["Financial Analysis", "Excel", "Budgeting", "Accounting", "Communication"],
        "experience_required": "2-4 years",
        "category": "Finance"
    },
    {
        "title": "Senior Accountant",
        "company": "Manufacturing Excellence LLC",
        "location": "Phoenix, AZ",
        "salary": "$65,000 - $80,000",
        "type": "Full-time",
        "description": "Manage accounts payable/receivable, prepare financial statements, ensure compliance with accounting standards, and support audit processes. Work with QuickBooks and advanced Excel functions.",
        "required_skills": ["Accounting", "Excel", "QuickBooks", "Auditing", "Tax Preparation"],
        "experience_required": "3-5 years",
        "category": "Finance"
    },
    
    # Sports & Fitness Roles
    {
        "title": "Sports Coach",
        "company": "Elite Athletic Academy",
        "location": "Orlando, FL",
        "salary": "$45,000 - $65,000",
        "type": "Full-time",
        "description": "Train athletes, develop training programs, analyze performance, provide motivational support, and manage team dynamics. Work with athletes at various skill levels to achieve their goals.",
        "required_skills": ["Team Coaching", "Athletic Training", "Leadership", "Communication", "Sports Analytics"],
        "experience_required": "3-5 years",
        "category": "Sports"
    },
    {
        "title": "Fitness Trainer",
        "company": "Premium Health Club",
        "location": "San Diego, CA",
        "salary": "$40,000 - $55,000 + Bonuses",
        "type": "Full-time",
        "description": "Design personalized workout plans, provide fitness coaching, nutrition guidance, and help clients achieve their fitness goals. Maintain certifications and stay updated with fitness trends.",
        "required_skills": ["Fitness Coaching", "Nutrition Planning", "Communication", "Time Management", "Customer Service"],
        "experience_required": "2-4 years",
        "category": "Sports"
    },
    
    # Creative & Design Roles
    {
        "title": "UI/UX Designer",
        "company": "Design Studio Pro",
        "location": "Portland, OR",
        "salary": "$80,000 - $100,000",
        "type": "Full-time",
        "description": "Design user interfaces and user experiences for web and mobile applications, conduct user research, create wireframes and prototypes. Collaborate with development teams to implement designs.",
        "required_skills": ["UI/UX Design", "Graphic Design", "Adobe Creative Suite", "Communication", "Problem Solving"],
        "experience_required": "2-4 years",
        "category": "Design"
    },
    {
        "title": "Content Writer",
        "company": "Digital Media House",
        "location": "Nashville, TN",
        "salary": "$50,000 - $65,000",
        "type": "Full-time",
        "description": "Create engaging content for websites, blogs, social media, and marketing materials. Research topics and optimize content for SEO. Work with marketing team to develop content strategies.",
        "required_skills": ["Content Writing", "SEO", "Communication", "Market Research", "Time Management"],
        "experience_required": "1-3 years",
        "category": "Content"
    }
]

class ResumeJobMatchingApp:
    """Main application class for Resume-Job Matching System"""
    
    def __init__(self):
        self.job_matcher = JobMatcher()
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"
    
    def process_resume_and_recommend(self, resume_text, pdf_file):
        """Process resume and return predictions with job recommendations"""

        if pdf_file:
            resume_text = self.extract_text_from_pdf(pdf_file)
            if resume_text.startswith("Error"):
                return resume_text, ""
        
        if not resume_text or resume_text.strip() == "":
            return "Please provide either resume text or upload a PDF file.", ""
        
        try:
            
            predicted_category, confidence = self.job_matcher.predict_job_category(resume_text)
            
            
            job_recommendations = self.job_matcher.get_job_recommendations(resume_text, COMPREHENSIVE_JOBS)
            
            
            skills = self.job_matcher.parser.extract_comprehensive_skills(resume_text)
            experience_years = self.job_matcher.parser.extract_experience_years(resume_text)
            
            
            prediction_result = f"""
##  Resume Analysis & Prediction

### **Predicted Job Category**: {predicted_category}
### **Years of Experience**: {experience_years} years
### **Skills Identified**: {len(skills)} skills

**Top Skills Found:**
{', '.join(skills[:10]) if skills else 'No specific skills detected'}
{f'... and {len(skills) - 10} more' if len(skills) > 10 else ''}

---
            """
            
            
            recommendations_result = "##  Recommended Jobs\n\n"
            
            for i, rec in enumerate(job_recommendations[:3], 1): 
                job = rec['job']
                
                recommendations_result += f"""
### {i}. **{job['title']}**
**Company:** {job['company']}  
**Location:** {job['location']}  
**Salary:** {job['salary']}  
**Type:** {job['type']}  
**Experience Required:** {job['experience_required']}

**Match Score:** {rec['combined_score']:.1%} | **Skill Match:** {rec['skill_score']:.1%} | **Content Match:** {rec['similarity_score']:.1%}

**Description:** {job['description'][:200]}...

**Required Skills:** {', '.join(job['required_skills'])}

**Your Matching Skills:** {', '.join(rec['matched_skills']) if rec['matched_skills'] else 'General profile match'}

---
"""
            
            return prediction_result, recommendations_result
            
        except Exception as e:
            return f"Error processing resume: {str(e)}", ""


app = ResumeJobMatchingApp()

def search_and_filter_jobs(search_query, category_filter):
    """Search and filter jobs based on user input"""
    if not search_query and category_filter == "All Categories":
        
        filtered_jobs = COMPREHENSIVE_JOBS
    else:
        filtered_jobs = []
        search_query_lower = search_query.lower().strip() if search_query else ""
        
        for job in COMPREHENSIVE_JOBS:
           
            category_match = (category_filter == "All Categories" or 
                            job.get('category', '').lower() == category_filter.lower())
            
            
            if search_query_lower:
                search_match = (
                    search_query_lower in job['title'].lower() or
                    search_query_lower in job['company'].lower() or
                    search_query_lower in job['description'].lower() or
                    search_query_lower in job['location'].lower() or
                    any(search_query_lower in skill.lower() for skill in job.get('required_skills', []))
                )
            else:
                search_match = True
            
            if category_match and search_match:
                filtered_jobs.append(job)
    
    
    if not filtered_jobs:
        return "No jobs found matching your criteria."
    
    results = f"## Found {len(filtered_jobs)} Job(s)\n\n"
    
    for i, job in enumerate(filtered_jobs, 1):
        results += f"""
### {i}. **{job['title']}**
**Company:** {job['company']}  
**Location:** {job['location']}  
**Salary:** {job['salary']}  
**Type:** {job['type']}  
**Experience Required:** {job['experience_required']}

**Description:** {job['description'][:200]}...

**Required Skills:** {', '.join(job['required_skills'])}

---
"""
    
    return results


def add_new_job(title, company, location, salary, job_type, description, required_skills, experience, category):
    """Add a new job to the COMPREHENSIVE_JOBS list"""
    global COMPREHENSIVE_JOBS
    

    if not all([title, company, location, salary, job_type, description, category]):
        return " Error: Please fill in all required fields.", ""
    
    skills_list = [skill.strip() for skill in required_skills.split(',') if skill.strip()]
    
    new_job = {
        "title": title.strip(),
        "company": company.strip(),
        "location": location.strip(),
        "salary": salary.strip(),
        "type": job_type,
        "description": description.strip(),
        "required_skills": skills_list,
        "experience_required": experience.strip() if experience.strip() else "Not specified",
        "category": category
    }
    

    COMPREHENSIVE_JOBS.append(new_job)
    
    success_msg = f"Job '{title}' at {company} has been successfully added!"
    

    updated_list = f" Total jobs in database: {len(COMPREHENSIVE_JOBS)}"
    
    return success_msg, updated_list

def create_gradio_interface():
    """Create and configure the Gradio interface with Add Job tab"""
    
    try:
        
        categories = ["All Categories"] + list(set(job.get('category', 'Other') for job in COMPREHENSIVE_JOBS))
        categories.sort()
        
        
        job_types = ["Full-time", "Part-time", "Contract", "Internship", "Remote"]
        
        
        job_categories = [cat for cat in categories if cat != "All Categories"]
        if not job_categories: 
            job_categories = ["Other"]
        
        with gr.Blocks(title="AI-Powered Resume-Job Matching System", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""
            #  AI-Powered Resume-Job Matching System
            
            Upload your resume or paste your resume text to get personalized job recommendations using advanced machine learning algorithms.
            """)
            
            with gr.Tabs():
               
                with gr.Tab(" Resume Analysis & Job Recommendations"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Upload Resume")
                            resume_text = gr.Textbox(
                                label="Resume Text",
                                placeholder="Paste your resume text here...",
                                lines=10,
                                max_lines=15
                            )
                            resume_file = gr.File(
                                label="Or Upload PDF Resume",
                                file_types=[".pdf"],
                                type="filepath"
                            )
                            analyze_btn = gr.Button("Analyze Resume & Get Recommendations", variant="primary")
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### Analysis Results")
                            prediction_output = gr.Markdown(label="Resume Analysis")
                            recommendations_output = gr.Markdown(label="Job Recommendations")
                    
                    analyze_btn.click(
                        fn=app.process_resume_and_recommend,
                        inputs=[resume_text, resume_file],
                        outputs=[prediction_output, recommendations_output]
                    )
                
                # Job Search & Browse Tab
                with gr.Tab(" Job Search & Browse"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Search Jobs")
                            search_query = gr.Textbox(
                                label="Search Query",
                                placeholder="e.g., Python, Data Scientist, Marketing...",
                                lines=1
                            )
                            category_filter = gr.Dropdown(
                                choices=categories,
                                label="Filter by Category",
                                value="All Categories"
                            )
                            search_btn = gr.Button("🔍 Search Jobs", variant="primary")
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### Search Results")
                            search_results = gr.Markdown()
                    
                    search_btn.click(
                        fn=search_and_filter_jobs,
                        inputs=[search_query, category_filter],
                        outputs=[search_results]
                    )
                    
                    category_filter.change(
                        fn=search_and_filter_jobs,
                        inputs=[search_query, category_filter],
                        outputs=[search_results]
                    )
                
                # Add Job Listing Tab
                with gr.Tab(" Add Job Listing"):
                    gr.Markdown("### Add New Job to Database")
                    gr.Markdown("Fill in the details below to add a new job listing to the system.")
                    
                    with gr.Row():
                        with gr.Column():
                            # Basic Job Information
                            gr.Markdown("#### Basic Information")
                            job_title = gr.Textbox(
                                label="Job Title *",
                                placeholder="e.g., Senior Data Scientist",
                                lines=1
                            )
                            job_company = gr.Textbox(
                                label="Company Name *",
                                placeholder="e.g., Tech Innovations Inc.",
                                lines=1
                            )
                            job_location = gr.Textbox(
                                label="Location *",
                                placeholder="e.g., San Francisco, CA",
                                lines=1
                            )
                            job_salary = gr.Textbox(
                                label="Salary Range *",
                                placeholder="e.g., $120,000 - $150,000",
                                lines=1
                            )
                            job_type_dropdown = gr.Dropdown(
                                choices=job_types,
                                label="Job Type *",
                                value="Full-time"
                            )
                            job_category_dropdown = gr.Dropdown(
                                choices=job_categories,
                                label="Job Category *",
                                value=job_categories[0]
                            )
                        
                        with gr.Column():
                           
                            gr.Markdown("#### Job Details")
                            job_description = gr.Textbox(
                                label="Job Description *",
                                placeholder="Detailed description of the role, responsibilities, and requirements...",
                                lines=8,
                                max_lines=12
                            )
                            job_skills = gr.Textbox(
                                label="Required Skills",
                                placeholder="Python, Machine Learning, SQL, Communication (separate with commas)",
                                lines=3
                            )
                            job_experience = gr.Textbox(
                                label="Experience Required",
                                placeholder="e.g., 3-5 years",
                                lines=1
                            )
                    
                   
                    add_job_btn = gr.Button(" Add Job Listing", variant="primary", size="lg")
                    
                    with gr.Row():
                        job_add_status = gr.Markdown()
                        job_count_status = gr.Markdown()
                    
                   
                    add_job_btn.click(
                        fn=add_new_job,
                        inputs=[
                            job_title, job_company, job_location, job_salary,
                            job_type_dropdown, job_description, job_skills,
                            job_experience, job_category_dropdown
                        ],
                        outputs=[job_add_status, job_count_status]
                    )
                
                # About Tab
                with gr.Tab(" About"):
                 gr.Markdown("""
                ## About This System
                
                This AI-powered Resume-Job Matching System uses advanced machine learning techniques to:
                
                ###  **Resume Analysis**
                - **Skill Extraction**: Identifies 50+ technical and soft skills across various industries
                - **Experience Detection**: Automatically extracts years of experience from resume text
                - **Job Category Prediction**: Uses Random Forest classifier to predict the most suitable job category
                - **PDF Processing**: Supports PDF resume uploads with text extraction
                
                ###  **Matching Algorithm**
                - **Word2Vec Embeddings**: Creates semantic representations of resume and job descriptions
                - **TF-IDF Vectorization**: Analyzes text similarity and keyword importance
                - **Multi-factor Scoring**: Combines skill matching, text similarity, and experience levels
                - **Personalized Recommendations**: Ranks jobs based on individual profile compatibility
                
                ###  **Supported Job Categories**
                - **Technical**: Data Science, AI/ML, Software Development, UI/UX Design
                - **Business**: Sales, Marketing, Project Management, Finance & Accounting
                - **People & Operations**: Human Resources, Talent Acquisition, Leadership
                - **Specialized**: Sports & Fitness, Creative & Design, Content Creation
                
                ###  **Key Features**
                - **Multi-format Support**: Text input and PDF upload
                - **Real-time Analysis**: Instant results with confidence scores
                - **Comprehensive Job Database**: 15+ curated job listings across industries
                - **Advanced Search**: Filter and search jobs by keywords and categories
                - **Skill Matching**: Detailed breakdown of matching skills for each recommendation
                
                ###  **Technical Stack**
                - **Machine Learning**: Scikit-learn, Random Forest, Word2Vec
                - **NLP Processing**: NLTK, spaCy, TF-IDF Vectorization
                - **PDF Processing**: PDFplumber for text extraction
                - **Interface**: Gradio for interactive web interface
                - **Data Processing**: Pandas, NumPy for data manipulation
                
                """)
        return interface
    
        
    except Exception as e:
        print(f"Error creating interface: {e}")
        raise e


if __name__ == "__main__":
        print(" Starting AI-Powered Resume-Job Matching System...")
        

        print(" Initializing application...")
        app = ResumeJobMatchingApp()
        

        print(" Training AI models...")
        app.job_matcher.train_models()
        print(" Models trained successfully!")
        

        print(" Creating user interface...")
        interface = create_gradio_interface()
        print("Interface created successfully!")
        
        print(" Launching web application...")
        print(" Application will be available at: http://localhost:7860")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            debug=True
        )