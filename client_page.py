import streamlit as st
from database_utils import insert_data, create_table_if_not_exists
import pandas as pd
import base64, random, time, datetime, io, re, os # Added os for path handling
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
import docx # Import python-docx for DOCX text extraction

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
    },
    "iOS Development": { # Added based on your Courses.py import
        "skills": ['iOS', 'Swift', 'Objective-C', 'Xcode', 'Cocoa Touch'],
        "courses": ios_course
    }
}

class DomainClassifier:
    @staticmethod
    def init_classifier():
        try:
            return joblib.load(DOMAIN_CLASSIFIER_PATH)
        except FileNotFoundError:
            # Dataset minimal pour la première initialisation
            data = {
                'texte': [
                    "python machine learning data analysis",
                    "javascript react web development",
                    "android kotlin mobile development",
                    "ux ui design prototyping",
                    "sql data warehousing etl",
                    "swift ios mobile app development" # Added iOS example
                ],
                'domaine': [
                    "Data Science",
                    "Web Development",
                    "Android Development",
                    "UI/UX Design",
                    "Data Science",
                    "iOS Development" # Corresponding domain for the new example
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
        except Exception as e:
            print(f"Error initializing classifier: {e}")
            raise # Re-raise to prevent the app from running without a classifier


    @staticmethod
    def predict_domain(text, classifier):
        try:
            return classifier.predict([text])[0]
        except Exception as e:
            # In a batch processing, you might log this instead of st.error
            print(f"Prediction error: {str(e)}")
            return None

def pdf_reader(file_path):
    """
    Reads text from a PDF file.
    """
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    text = ""
    try:
        with open(file_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    finally:
        converter.close()
        fake_file_handle.close()
    return text

def docx_reader(file_path):
    """
    Reads text from a DOCX file.
    """
    try:
        document = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in document.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""

def analyze_resume(resume_path, classifier):
    """
    Analyzes a resume (PDF or DOCX) using pyresparser and predicts the career field.
    """
    try:
        # pyresparser can directly handle .pdf and .docx files
        # It handles the internal text extraction for both formats
        resume_data = ResumeParser(resume_path).get_extracted_data()
        if not resume_data:
            print(f"No data extracted from {os.path.basename(resume_path)}. This might indicate a parsing error or empty file.")
            return None

        # Use skills from pyresparser output for domain prediction
        skills_text = ' '.join(resume_data.get('skills', []))
        reco_field = DomainClassifier.predict_domain(skills_text, classifier)

        # Fallback to rules if ML fails or provides 'General'
        if not reco_field or reco_field == "General":
            print(f"ML prediction for {os.path.basename(resume_path)} was '{reco_field}', attempting keyword fallback.")
            reco_field = classify_domain_by_keywords(resume_data.get('skills', []))
            print(f"Keyword fallback for {os.path.basename(resume_path)} resulted in: {reco_field}")


        return {
            **resume_data,
            'predicted_field': reco_field,
            'recommendations': DOMAIN_RECOMMENDATIONS.get(reco_field, {})
        }
    except Exception as e:
        print(f"Error analyzing resume {os.path.basename(resume_path)}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

def classify_domain_by_keywords(skills):
    """Fallback mechanism for domain detection using keyword matching."""
    ds_keywords = ['tensorflow','keras','pytorch','machine learning', 'data science', 'analytics', 'statistics', 'numpy', 'pandas', 'scikit-learn']
    web_keywords = ['react', 'django', 'node', 'javascript', 'html', 'css', 'web development', 'frontend', 'backend', 'fullstack']
    android_keywords = ['android sdk', 'kotlin', 'java', 'mobile development', 'android studio', 'gradle']
    uiux_keywords = ['figma', 'adobe xd', 'user research', 'wireframing', 'prototyping', 'ui/ux', 'usability', 'design thinking']
    ios_keywords = ['ios', 'swift', 'objective-c', 'xcode'] # Added based on Courses.py

    for skill in skills:
        skill_lower = skill.lower()
        if any(keyword in skill_lower for keyword in ds_keywords):
            return "Data Science"
        elif any(keyword in skill_lower for keyword in web_keywords):
            return "Web Development"
        elif any(keyword in skill_lower for keyword in android_keywords):
            return "Android Development"
        elif any(keyword in skill_lower for keyword in uiux_keywords):
            return "UI/UX Design"
        elif any(keyword in skill_lower for keyword in ios_keywords): # Check for iOS skills
            return "iOS Development"
    return "General" # Default if no specific domain is found

def calculate_resume_score(resume_path):
    """
    Calculates a basic resume score based on section presence.
    Handles both PDF and DOCX files.
    """
    resume_text = ""
    if resume_path.lower().endswith('.pdf'):
        resume_text = pdf_reader(resume_path)
    elif resume_path.lower().endswith('.docx'):
        resume_text = docx_reader(resume_path)
    else:
        print(f"Warning: Unsupported file type for scoring: {os.path.basename(resume_path)}")
        return 0

    if not resume_text:
        return 0 # No text extracted, score is 0

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

def determine_candidate_level(score):
    """Determine candidate level based on resume score"""
    if score >= 80:
        return "Senior"
    elif score >= 50:
        return "Mid-level"
    else:
        return "Junior"

# Simplified display_results for batch processing (without Streamlit UI)
def process_and_save_analysis(analysis, resume_path):
    """Processes analysis data and attempts to save it to the database."""
    if not analysis:
        print(f"Skipping saving for {os.path.basename(resume_path)} due to failed analysis.")
        return False

    resume_score = calculate_resume_score(resume_path)
    cand_level = determine_candidate_level(resume_score)
    recommended_skills = analysis['recommendations'].get('skills', [])[:5]
    recommended_courses = analysis['recommendations'].get('courses', [])

    print(f"Processing: {analysis.get('name', 'N/A')} ({os.path.basename(resume_path)})")
    print(f"   Predicted Field: {analysis['predicted_field']}")
    print(f"   Resume Score: {resume_score}")
    print(f"   Candidate Level: {cand_level}")
    print(f"   Recommended Skills: {', '.join(recommended_skills)}")
    print(f"   Recommended Courses: {len(recommended_courses)} courses")

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
        print(f"Failed to save analysis for {os.path.basename(resume_path)}")
    return success

# --- Streamlit UI Functions ---
def show_client_page():
    create_table_if_not_exists()
    st.title("📄 AI Resume Analyzer - Client Portal")

    # Initialisation du classifieur
    classifier = DomainClassifier.init_classifier()

    st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload your resume for smart analysis</h5>''',
                unsafe_allow_html=True)

    # Allow both PDF and DOCX uploads
    uploaded_file = st.file_uploader("Choose your Resume", type=["pdf", "docx"])
    if not uploaded_file:
        st.info("Please upload a resume to analyze.")
        return

    # Create Uploaded_Resumes directory if it doesn't exist
    upload_dir = './Uploaded_Resumes'
    os.makedirs(upload_dir, exist_ok=True)

    save_path = None # Initialize save_path to None
    try:
        with st.spinner('Analyzing your Resume...'):
            # Save the uploaded file (PDF or DOCX) for analysis
            save_path = os.path.join(upload_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Analyze the resume
            analysis = analyze_resume(save_path, classifier)
            if not analysis:
                st.error("Failed to analyze resume. Please try a different file or check its content.")
                return # Exit function if analysis fails

            # Display results for the UI
            display_results(analysis, save_path)

    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        # This will catch the OSError as well and display it in Streamlit
    finally:
        # Ensure the file is deleted even if an error occurs
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
                print(f"Cleaned up temporary file: {save_path}")
            except Exception as cleanup_e:
                print(f"Error during file cleanup for {save_path}: {cleanup_e}")


def display_results(analysis, resume_path):
    st.header("**Resume Analysis**")
    st.success(f"Hello {analysis.get('name', 'there')}") # Use .get for robustness

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
            st.error("Failed to automatically save analysis results to database.")

    # Display basic info
    with st.expander("🔍 Basic Information"):
        cols = st.columns(3)
        cols[0].text(f"Name: {analysis.get('name', 'N/A')}")
        cols[1].text(f"Email: {analysis.get('email', 'N/A')}")
        cols[2].text(f"Pages: {analysis.get('no_of_pages', 1)}")

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

def display_resume_score(score):
    st.subheader("📝 Resume Quality Score")
    with st.expander("How we calculate your score"):
        st.write(f"Your resume scored **{score}/100** based on completeness.")
        if score == 0:
            st.warning("Note: A score of 0 might indicate issues with text extraction or an empty resume. Ensure your DOCX files are parsable if the score remains 0 for them.")

    progress_bar = st.progress(0)
    for percent in range(score):
        time.sleep(0.01)
        progress_bar.progress(percent + 1)
    st.metric("Overall Score", f"{score}/100")

def course_recommender(course_list):
    st.subheader("🎓 Recommended Courses & Certifications")
    if not course_list:
        st.info("No specific course recommendations for this domain yet. Check back later!")
        return

    num_recommendations = st.slider("Select number of recommendations", 1, min(10, len(course_list)), min(3, len(course_list)))

    for i, (name, link) in enumerate(course_list[:num_recommendations], 1):
        st.markdown(f"{i}. [{name}]({link})")

def display_bonus_content():
    st.subheader("💡 Bonus Resources")

    st.markdown("### Resume Tips")
    resume_vid = random.choice(resume_videos)
    st.video(resume_vid)

    st.markdown("### Interview Prep")
    interview_vid = random.choice(interview_videos)
    st.video(interview_vid)

# --- New function for batch testing ---
def run_batch_analysis(folder_path):
    create_table_if_not_exists()
    classifier = DomainClassifier.init_classifier()

    print(f"\n--- Starting Batch Analysis in: {folder_path} ---")
    processed_count = 0
    failed_count = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and (filename.lower().endswith('.pdf') or filename.lower().endswith('.docx')):
            print(f"\nAnalyzing: {filename}")
            analysis = analyze_resume(file_path, classifier)
            if analysis:
                if process_and_save_analysis(analysis, file_path):
                    processed_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
                print(f"Skipping {filename} due to analysis failure.")
        else:
            print(f"Skipping non-resume file: {filename}")

    print(f"\n--- Batch Analysis Complete ---")
    print(f"Successfully processed and saved: {processed_count} resumes")
    print(f"Failed to process/save: {failed_count} resumes")
    print("Check the database for results and console for detailed logs.")


if __name__ == "__main__":
    # Choose how you want to run:

    # 1. To run the Streamlit UI for single file uploads (PDF or DOCX):
    # Uncomment the line below and comment out the batch processing block.
    # To launch, open your terminal and run: streamlit run client_page.py
    show_client_page()

    # 2. To run a batch analysis with a specified dataset (DOCX files or PDF/DOCX mix):
    # Uncomment the block below and comment out the show_client_page() call.
    # To launch, open your terminal and run: python client_page.py
    # KAGGLE_DATASET_PATH = './KaggleDataset' # <--- IMPORTANT: Adjust this path as needed!
    # if os.path.exists(KAGGLE_DATASET_PATH):
    #     run_batch_analysis(KAGGLE_DATASET_PATH)
    # else:
    #     print(f"Error: The specified path for batch processing does not exist: {KAGGLE_DATASET_PATH}")
    #     print("Please create this directory and place your resume files there, or update the path.")