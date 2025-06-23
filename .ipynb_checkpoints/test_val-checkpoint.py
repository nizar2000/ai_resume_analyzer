def test_pdf_extraction():
    """Test de l'extraction de texte PDF"""
    test_results = []
    
    for pdf_file in test_dataset:
        extracted_text = pdf_reader(pdf_file)
        
        # Critères de validation
        assert len(extracted_text) > 0, "Texte extrait vide"
        assert not contains_garbled_text(extracted_text), "Texte corrompu"
        
        test_results.append({
            'file': pdf_file,
            'text_length': len(extracted_text),
            'extraction_quality': assess_extraction_quality(extracted_text)
        })
    
    return test_results


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )),
    ('svm', SVC(
        kernel='linear',
        C=1.0,
        random_state=42
    ))
])


potential_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ))
])


def extract_features(resume_data):
    """Extraction des features pour Random Forest"""
    features = {
        'experience_years': calculate_experience_years(resume_data),
        'education_level': encode_education_level(resume_data),
        'skills_count': len(resume_data.get('skills', [])),
        'projects_count': len(resume_data.get('projects', [])),
        'certifications_count': len(resume_data.get('certifications', [])),
        'resume_completeness': calculate_completeness_score(resume_data),
        'technical_skills_ratio': calculate_technical_ratio(resume_data)
    }
    return list(features.values())






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








