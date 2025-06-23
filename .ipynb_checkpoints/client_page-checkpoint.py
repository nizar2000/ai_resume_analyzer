# client_page.py (version améliorée)
import streamlit as st
from database_utils import insert_data, create_table_if_not_exists
import pandas as pd
import base64, random, time, datetime, io, re
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from streamlit_tags import st_tags
from pyresparser import ResumeParser
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import yt_dlp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Configuration du modèle ML
DOMAIN_CLASSIFIER_PATH = 'domain_classifier.pkl'
DOMAIN_RECOMMENDATIONS = {
    "Data Science": {
        "skills": ['Data Visualization', 'Machine Learning', 'Statistical Modeling', 
                 'Python', 'SQL', 'TensorFlow', 'PyTorch'],
        "courses": ds_course
    },
    "Web Development": {
        "skills": ['JavaScript', 'React', 'HTML/CSS', 'Node.js', 
                 'REST APIs', 'Git', 'Responsive Design'],
        "courses": web_course
    },
    "Android Development": {
        "skills": ['Android SDK', 'Kotlin', 'Java', 'XML', 
                 'Firebase', 'Material Design'],
        "courses": android_course
    },
    "UI/UX Design": {
        "skills": ['Figma', 'Adobe XD', 'User Research', 
                 'Wireframing', 'Prototyping', 'Color Theory'],
        "courses": uiux_course
    }
}

class DomainClassifier:
    @staticmethod
    def init_classifier():
        try:
            return joblib.load(DOMAIN_CLASSIFIER_PATH)
        except:
            # Dataset minimal pour la première initialisation
            data = {
                'texte': [
                    "python machine learning data analysis",
                    "javascript react web development",
                    "android kotlin mobile development",
                    "ux ui design prototyping"
                ],
                'domaine': [
                    "Data Science",
                    "Web Development",
                    "Android Development",
                    "UI/UX Design"
                ]
            }
            df = pd.DataFrame(data)
            
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('svm', SVC(kernel='linear'))
            ])
            pipeline.fit(df['texte'], df['domaine'])
            joblib.dump(pipeline, DOMAIN_CLASSIFIER_PATH)
            return pipeline

    @staticmethod
    def predict_domain(text, classifier):
        try:
            return classifier.predict([text])[0]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def analyze_resume(resume_path, classifier):
    resume_data = ResumeParser(resume_path).get_extracted_data()
    if not resume_data:
        return None
    
    # Analyse avec ML
    skills_text = ' '.join(resume_data.get('skills', []))
    reco_field = DomainClassifier.predict_domain(skills_text, classifier)
    
    # Fallback aux règles si ML échoue
    if not reco_field:
        reco_field = fallback_domain_detection(resume_data['skills'])
    
    return {
        **resume_data,
        'predicted_field': reco_field,
        'recommendations': DOMAIN_RECOMMENDATIONS.get(reco_field, {})
    }

def fallback_domain_detection(skills):
    # Votre logique existante de détection par mots-clés
    ds_keywords = ['tensorflow','keras','pytorch','machine learning']
    web_keywords = ['react', 'django', 'node', 'javascript']
    
    for skill in skills:
        if skill.lower() in ds_keywords:
            return "Data Science"
        elif skill.lower() in web_keywords:
            return "Web Development"
    return "General"

def show_client_page():
    create_table_if_not_exists()
    st.title("📄 AI Resume Analyzer - Client Portal")
    
    # Initialisation du classifieur
    classifier = DomainClassifier.init_classifier()
    
    st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload your resume for smart analysis</h5>''',
                unsafe_allow_html=True)
    
    pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
    if not pdf_file:
        return
        
    with st.spinner('Analyzing your Resume...'):
        # Sauvegarde et traitement du PDF
        save_path = f'./Uploaded_Resumes/{pdf_file.name}'
        with open(save_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Analyse complète
        analysis = analyze_resume(save_path, classifier)
        if not analysis:
            st.error("Failed to analyze resume")
            return
        
        # Affichage des résultats
        display_results(analysis, save_path)
def display_resume_score(score):
    st.subheader("📝 Resume Quality Score")
    with st.expander("How we calculate your score"):
        st.write(f"Your resume scored {score}/100 based on completeness")
    
    progress_bar = st.progress(0)
    for percent in range(score):
        time.sleep(0.01)
        progress_bar.progress(percent + 1)
    st.metric("Overall Score", f"{score}/100")

def determine_candidate_level(score):
    """Determine candidate level based on resume score"""
    if score >= 80:
        return "Senior"
    elif score >= 50:
        return "Mid-level"
    else:
        return "Junior"
def calculate_resume_score(resume_path):
    resume_text = pdf_reader(resume_path)
    score = 0
    criteria = {
        'Profile': 20,
        'Experience': 20,
        'Projects': 20,
        'Skills': 20,
        'Education': 10,
        'Achievements': 10
    }
    
    for section, points in criteria.items():
        if section.lower() in resume_text.lower():
            score += points
    return score


def display_results(analysis, resume_path):
    st.header("**Resume Analysis**")
    st.success(f"Hello {analysis['name']}")
    
    # Calculate resume score
    resume_score = calculate_resume_score(resume_path)
    
    # Determine candidate level
    cand_level = determine_candidate_level(resume_score)
    
    # Get recommendations
    recommended_skills = analysis['recommendations'].get('skills', [])[:5]  # Top 5 skills
    recommended_courses = analysis['recommendations'].get('courses', [])
    
    # Automatically save to database
    with st.spinner('Saving your analysis...'):
        success = insert_data(
            name=analysis.get('name', ''),
            email=analysis.get('email', ''),
            res_score=resume_score,
            no_of_pages=analysis.get('no_of_pages', 1),
            reco_field=analysis['predicted_field'],
            cand_level=cand_level,
            skills=analysis.get('skills', []),
            recommended_skills=recommended_skills,
            courses=recommended_courses
        )
        
        if not success:
            st.error("Failed to automatically save analysis results")

    # Display basic info
    with st.expander("🔍 Basic Information"):
        cols = st.columns(3)
        cols[0].text(f"Name: {analysis.get('name', 'N/A')}")
        cols[1].text(f"Email: {analysis.get('email', 'N/A')}")
        cols[2].text(f"Pages: {analysis.get('no_of_pages', 1)}")
    
    # Rest of your display code remains the same...
    st.subheader("🎯 Career Field Prediction")
    st.markdown(f"**Our analysis suggests:** `{analysis['predicted_field']}`")
    
    st.subheader("🛠 Skills Analysis")
    st_tags(
        label='### Your Current Skills',
        text='See recommendations below',
        value=analysis.get('skills', []),
        key='skills_display'
    )
    
    if analysis['recommendations']:
        st.subheader("✨ Recommended Skills")
        st_tags(
            label='### Skills to Boost',
            text='These skills would complement your profile',
            value=recommended_skills,
            key='rec_skills'
        )
        
        st.subheader("📚 Recommended Courses")
        course_recommender(recommended_courses)
    
    display_resume_score(resume_score)
    display_bonus_content()
def resume_score_analysis(resume_path):
    resume_text = pdf_reader(resume_path)
    score = 0
    
    st.subheader("📝 Resume Quality Score")
    with st.expander("How we calculate your score"):
        criteria = {
            'Profile': 20,
            'Experience': 20,
            'Projects': 20,
            'Skills': 20,
            'Education': 10,
            'Achievements': 10
        }
        
        for section, points in criteria.items():
            if section.lower() in resume_text.lower():
                score += points
                st.success(f"+{points} for {section}")
            else:
                st.warning(f"Missing {section} section")
    
    # Barre de progression
    progress_bar = st.progress(0)
    for percent in range(score):
        time.sleep(0.01)
        progress_bar.progress(percent + 1)
    st.metric("Overall Score", f"{score}/100")

def course_recommender(course_list):
    st.subheader("🎓 Recommended Courses & Certifications")
    num_recommendations = st.slider("Select number of recommendations", 1, 10, 3)
    
    for i, (name, link) in enumerate(course_list[:num_recommendations], 1):
        st.markdown(f"{i}. [{name}]({link})")

def display_bonus_content():
    st.subheader("💡 Bonus Resources")
    
    # Vidéo pour le CV
    resume_vid = random.choice(resume_videos)
    st.video(resume_vid)
    
    # Vidéo pour l'entretien
    interview_vid = random.choice(interview_videos)
    st.video(interview_vid)

if __name__ == "__main__":
    show_client_page()