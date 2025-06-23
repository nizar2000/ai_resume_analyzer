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

