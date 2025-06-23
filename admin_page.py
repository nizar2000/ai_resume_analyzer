# admin_page.py
import ast
import base64
import pymysql
import streamlit as st
import pandas as pd
import plotly.express as px
from database_utils import get_persistent_connection, close_persistent_connection

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





def get_table_download_link(df, filename, text):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
def get_user_data(connection):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM data")  # 'data' is the table name
            result = cursor.fetchall()
            df = pd.DataFrame(result)
            return df

    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
# Streamlit App
def show_user_data(connection):
    st.title("User Data Viewer (MySQL + PyMySQL)")

    df = get_user_data(connection)  # You'll need to modify get_user_data as well
    if not df.empty:
        st.success("User data loaded successfully!")
        st.dataframe(df)
    else:
        st.warning("No data found or failed to load.")
def show_user_management(connection):
    """Display user management interface"""
    st.subheader("🚀 User Management & Moderation")
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT ID, Name, Email_ID, resume_score, Predicted_Field FROM data")
            data = cursor.fetchall()
    except Exception as e:
        st.error(f"Error fetching user data: {e}")
        return
    
    if not data:
        st.warning("No user data found")
        return
    
    df = pd.DataFrame(data, columns=["ID", "Name", "Email_ID", "Resume Score", "Job Field"])
    selected_indices = st.multiselect(
        "Select users:", 
        df.index, 
        format_func=lambda x: f"{df.loc[x, 'Name']} ({df.loc[x, 'Email_ID']})"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Delete Selected") and selected_indices:
            try:
                with connection.cursor() as cursor:
                    ids_to_delete = [str(df.loc[i, 'ID']) for i in selected_indices]
                    placeholders = ','.join(['%s'] * len(ids_to_delete))
                    cursor.execute(f"DELETE FROM data WHERE ID IN ({placeholders})", ids_to_delete)
                    connection.commit()
                st.success(f"Deleted {len(selected_indices)} users!")
                st.rerun()
            except pymysql.Error as e:
                st.error(f"Database error during deletion: {e}")
                connection.rollback()
            except Exception as e:
                st.error(f"Error deleting users: {e}")
    
    with col2:
        if selected_indices:
            selected_users = df.loc[selected_indices]
            st.download_button(
                label="📤 Export Selected",
                data=selected_users.to_csv(index=False),
                file_name="selected_users.csv",
                mime="text/csv"
            )

def show_analytics(connection):
    """Display analytics charts with improved error handling"""
    st.subheader("📈 Analytics Dashboard")
    
    # Skill Distribution - Updated with robust parsing
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT Actual_skills FROM data WHERE Actual_skills IS NOT NULL")
            skills_data = cursor.fetchall()
        
        if skills_data:
            all_skills = []
            for row in skills_data:
                skill_str = row['Actual_skills']
                
                # Skip empty/null values
                if not skill_str or str(skill_str).strip() == "":
                    continue
                    
                try:
                    # Case 1: Already a proper list format
                    if isinstance(skill_str, list):
                        all_skills.extend(skill_str)
                    
                    # Case 2: String that looks like a list
                    elif skill_str.startswith('[') and skill_str.endswith(']'):
                        try:
                            parsed = ast.literal_eval(skill_str)
                            if isinstance(parsed, list):
                                all_skills.extend(parsed)
                        except:
                            # Fallback for malformed lists
                            cleaned = skill_str[1:-1].replace("'", "").replace('"', '')
                            all_skills.extend([s.strip() for s in cleaned.split(',') if s.strip()])
                    
                    # Case 3: Comma-separated string
                    elif ',' in skill_str:
                        all_skills.extend([s.strip() for s in skill_str.split(',') if s.strip()])
                    
                    # Case 4: Single skill
                    else:
                        all_skills.append(skill_str.strip())
                        
                except Exception as e:
                    st.warning(f"Skipping malformed skill entry: {skill_str}")
                    continue

            if all_skills:
                skills_df = pd.DataFrame({'Skill': all_skills})
                skills_distribution = skills_df['Skill'].value_counts().reset_index()
                skills_distribution.columns = ['Skill', 'Count']

                st.subheader("Skill Distribution Among Users")
                fig = px.bar(skills_distribution.head(20),  # Show top 20 only
                            x='Skill', 
                            y='Count',
                            title='Top 20 Skill Distribution',
                            height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid skill data found after processing")

    except Exception as e:
        st.error(f"Error processing skills data: {str(e)}")
        if st.checkbox("Show debug info"):
            st.write("Sample of problematic data:")
            st.write(skills_data[0] if skills_data else "No skills data found")

        else:
            st.warning("No data found")


    except Exception as e:
        st.error(f"Error processing skills data: {e}")
    
    # Resume Score Distribution
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT resume_score FROM data")
            scores_data = cursor.fetchall()
        
        if scores_data:
            scores_df = pd.DataFrame(scores_data)
            # Rename column only if necessary
            if 'resume_score' in scores_df.columns:
               scores_df.rename(columns={'resume_score': 'Resume Score'}, inplace=True)

            st.subheader("📈 Resume Score Distribution")
            fig = px.histogram(
            scores_df,
            x='Resume Score',
            nbins=10,
            title='Distribution of Resume Scores',
            color_discrete_sequence=['#00B4D8'],
            height=500
            ) 
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No resume scores found.")
    except Exception as e:
        st.error(f"Error processing scores data: {e}")

def check_admin_credentials(username, password):
    """Validate admin credentials"""
    # In production, use proper password hashing and secrets management
    return username == "admin" and password == "admin"


def show_admin_page():
    """Main admin page function with session login flag"""
    st.title("🔒 AI Resume Analyzer - Admin Portal")

    # Session state initialization
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    # If not logged in, show login form
    if not st.session_state.admin_logged_in:
        st.sidebar.subheader('Admin Login')
        ad_user = st.sidebar.text_input("Username")
        ad_password = st.sidebar.text_input("Password", type='password')
        
        if st.sidebar.button('Login'):
            if check_admin_credentials(ad_user, ad_password):
                st.session_state.admin_logged_in = True
                st.success("Welcome Admin!")
            else:
                st.error("Wrong ID & Password Provided")
        return

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.admin_logged_in = False
        st.success("Logged out successfully")
        st.rerun()
        return

    # Initialize or reuse connection
    if 'db_conn' not in st.session_state or not st.session_state.db_conn.open:
        try:
            st.session_state.db_conn = get_persistent_connection()
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            return

    try:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 User Data", "🧹 User Management", "📈 Analytics", "🎯 Candidate Selector", "🔮 Candidate Potential Prediction"])        
        with tab1:
            show_user_data(st.session_state.db_conn)
        with tab2:
            show_user_management(st.session_state.db_conn)

        with tab3:
            show_analytics(st.session_state.db_conn)
        with tab4:
            show_candidate_selector(st.session_state.db_conn)
       
        with tab5:
            predict_candidate_potential(st.session_state.db_conn)

    except pymysql.Error as e:
        st.error(f"Database error: {e}")
        close_persistent_connection()
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Unexpected error: {e}")



def safe_eval_list_length(value):
    import ast
    try:
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return len(parsed)
    except Exception:
        pass
    return 0

def predict_candidate_potential(connection):
  
    st.subheader("🔮 Prédiction du Potentiel des Candidats")

    try:
        # ==============================================================
        # 2.5.1.2 a) Extraction des données
        # ==============================================================
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT resume_score, Actual_skills, Recommended_skills, 
                       User_Level, Predicted_Field 
                FROM data
                WHERE resume_score IS NOT NULL
            """)
            data = cursor.fetchall()

        if len(data) < 10:
            st.warning("Minimum 10 candidats requis pour l'analyse")
            return

        df = pd.DataFrame(data, columns=[
            'resume_score', 'Actual_skills', 'Recommended_skills',
            'User_Level', 'Predicted_Field'
        ])

        # ==============================================================
        # 2.5.1.2 b) Transformation des données
        # ==============================================================
        
        # Fonction de conversion sécurisée des compétences
        def safe_parse_skills(skill_str):
            try:
                if pd.isna(skill_str) or not skill_str.strip():
                    return []
                # Gestion des formats JSON et list-like
                if skill_str.startswith('[') and skill_str.endswith(']'):
                    parsed = ast.literal_eval(skill_str)
                    return [str(skill).strip() for skill in parsed] if parsed else []
                # Format alternatif (séparé par des virgules)
                return [s.strip() for s in skill_str.split(',') if s.strip()]
            except:
                return []

        # Conversion des scores
        df['resume_score'] = pd.to_numeric(df['resume_score'], errors='coerce')
        df = df.dropna(subset=['resume_score'])

        # Encodage du niveau (Tableau 2.1)
        level_map = {'Junior': 1, 'Mid-level': 2, 'Senior': 3}
        df['User_Level_Code'] = (
            df['User_Level']
            .str.strip()
            .str.lower()
            .map({'junior': 1, 'mid-level': 2, 'senior': 3})
            .fillna(1)
            .astype(int)
        )

        # Encodage des domaines
        df['Predicted_Field_Code'] = pd.factorize(df['Predicted_Field'])[0]

        # Conversion des compétences
        df['Actual_skills_list'] = df['Actual_skills'].apply(safe_parse_skills)
        df['Recommended_skills_list'] = df['Recommended_skills'].apply(safe_parse_skills)
        
        # Calcul des caractéristiques
        df['Actual_skills_count'] = df['Actual_skills_list'].apply(len)
        df['Recommended_skills_count'] = df['Recommended_skills_list'].apply(len)
        df['skill_gap'] = df['Recommended_skills_count'] - df['Actual_skills_count']

        # ==============================================================
        # 2.5.1.4 Définition des étiquettes
        # ==============================================================
        def assign_potential(row):
            try:
                if (row['resume_score'] >= 80 and 
                    row['User_Level_Code'] >= 2 and 
                    row['Actual_skills_count'] >= 5):
                    return 2  # Élevé
                elif row['resume_score'] >= 60:
                    return 1  # Moyen
                return 0  # Faible
            except:
                return 0

        df['Potential_Label'] = df.apply(assign_potential, axis=1)

        # ==============================================================
        # 2.5.1.5 Entraînement du modèle
        # ==============================================================
        features = df[[
            'resume_score', 'User_Level_Code', 'Predicted_Field_Code',
            'Actual_skills_count', 'Recommended_skills_count', 'skill_gap'
        ]]
        target = df['Potential_Label']

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        # Évaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # ==============================================================
        # Affichage des résultats
        # ==============================================================
        st.success(f"""
        **Performance du modèle :**
        - Précision : {accuracy:.1%}
        - Taille de l'ensemble d'entraînement : {len(X_train)} candidats
        - Taille du test : {len(X_test)} candidats
        """)

        # Importance des caractéristiques
        importance_df = pd.DataFrame({
            'Caractéristique': features.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig = px.bar(
            importance_df,
            x='Caractéristique',
            y='Importance',
            title="Importance des caractéristiques "
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"""
        Erreur lors de l'analyse :
        {str(e)}
        
        Conseils de dépannage :
        1. Vérifiez le format des compétences dans la base de données
        2. Contrôlez les valeurs manquantes
        3. Inspectez les types de données (df.dtypes)
        """)

def find_candidates_with_skills(connection, min_score=70, field=None, required_skills=None, min_level=None):
    """
    Enhanced candidate search with proper parameter handling
    """
    try:
        # First check if we have any data
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM data")
            result = cursor.fetchone()
            if not result or result.get('count', 0) == 0:
                st.warning("No candidates available in database")
                return []

        # Base query
        query = """
            SELECT ID, Name, Email_ID, resume_score, Predicted_Field, 
                   User_Level, Actual_skills, Recommended_skills
            FROM data 
            WHERE resume_score >= %s
        """
        params = [min_score]

        # Add field filter if specified
        if field:
            query += " AND LOWER(TRIM(Predicted_Field)) = LOWER(TRIM(%s))"
            params.append(field)

        # Handle User_Level filtering
        if min_level is not None:
            # For numeric levels (1-3)
            query += " AND (CASE "
            query += "WHEN User_Level LIKE '%junior%' THEN 1 "
            query += "WHEN User_Level LIKE '%mid%' THEN 2 "
            query += "WHEN User_Level LIKE '%senior%' THEN 3 "
            query += "ELSE 1 END) >= %s"
            params.append(min_level)

        # Execute query
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            candidates = cursor.fetchall()

            # If no exact matches, try partial field matching
            if not candidates and field:
                query = """
                    SELECT ID, Name, Email_ID, resume_score, Predicted_Field, 
                           User_Level, Actual_skills, Recommended_skills
                    FROM data 
                    WHERE resume_score >= %s
                    AND Predicted_Field LIKE %s
                    ORDER BY resume_score DESC
                    LIMIT 50
                """
                cursor.execute(query, [min_score, f"%{field}%"])
                candidates = cursor.fetchall()

            return candidates if candidates else []

    except pymysql.Error as e:
        st.error(f"Database error: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []


def show_candidate_selector(connection):
    """Improved candidate selector with better error handling"""
    st.subheader("🎯 Smart Candidate Selector")
    st.markdown("""
    **Find the best candidates** based on your specific requirements.
    This tool evaluates candidates based on:
    - Resume Score (50% weight)
    - Skill Match (30% weight)
    - Experience Level (20% weight)
    """)
    
    # Get filter criteria from user
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum Resume Score", 0, 100, 70)
        field = st.text_input("Desired Job Field (optional)")
    with col2:
        level = st.selectbox("Minimum Experience Level", 
                           ["Any", "Junior", "Mid-level", "Senior"])
        required_skills = st.text_input("Required Skills (comma separated, optional)")
    
    # Convert inputs
    skill_list = [s.strip().lower() for s in required_skills.split(",")] if required_skills else []
    level_map = {"Any": None, "Junior": 1, "Mid-level": 2, "Senior": 3}
    min_level = level_map[level]
    
    if st.button("🔍 Find Best Candidates"):
        with st.spinner("Searching for candidates..."):
            try:
                candidates = find_candidates_with_skills(
                    connection,
                    min_score=min_score,
                    field=field if field else None,
                    required_skills=skill_list if skill_list else None,
                    min_level=min_level
                )
                
                if not candidates:
                    st.warning("No candidates match all your criteria. Try broadening your search.")
                    return
                    
                # Prepare data for scoring
                candidate_data = []
                for cand in candidates:
                    # More robust skill parsing
                    actual_skills = []
                    if cand['Actual_skills']:
                        try:
                            # Try parsing as JSON/list first
                            if cand['Actual_skills'].startswith('['):
                                actual_skills = ast.literal_eval(cand['Actual_skills'])
                            # Fallback to comma-separated string
                            else:
                                actual_skills = [s.strip() for s in cand['Actual_skills'].split(',') if s.strip()]
                        except:
                            # Final fallback - treat as single skill
                            actual_skills = [cand['Actual_skills'].strip()]
                    
                    actual_skills_lower = [str(s).lower() for s in actual_skills]
                    
                    # Calculate skill match percentage
                    skill_match = 0
                    if skill_list:
                        matched_skills = sum(1 for skill in skill_list if skill in actual_skills_lower)
                        skill_match = (matched_skills / len(skill_list)) * 100 if skill_list else 100
                    else:
                        skill_match = 100  # full match if no skills specified
                    
                    # Convert level to numeric (1-3)
                    user_level = 1  # default
                    if cand['User_Level']:
                        user_level_str = str(cand['User_Level']).lower()
                        if 'senior' in user_level_str:
                            user_level = 3
                        elif 'mid' in user_level_str:
                            user_level = 2
                    
                    candidate_data.append({
                        'ID': cand['ID'],
                        'Name': cand['Name'],
                        'Email': cand['Email_ID'],
                        'Resume Score': float(cand['resume_score']),
                        'Field': cand['Predicted_Field'],
                        'Level': user_level,
                        'Skill Match %': round(skill_match, 1),
                        'Skills': ", ".join(str(s) for s in actual_skills) if actual_skills else "None listed"
                    })
                
                df = pd.DataFrame(candidate_data)
                
                # Calculate simple weighted score (no ML for simplicity)
                df['Match Score %'] = (
                    (df['Resume Score'] * 0.5) + 
                    (df['Level'] * 20) + 
                    (df['Skill Match %'] * 0.3)
                ).round(1)
                
                # Normalize to 0-100 scale
                max_score = df['Match Score %'].max()
                if max_score > 0:
                    df['Match Score %'] = (df['Match Score %'] / max_score * 100).round(1)
                
                # Show results
                st.success(f"Found {len(df)} matching candidates")
                
                # Sort and display top candidates
                top_candidates = df.sort_values('Match Score %', ascending=False).head(50)
                
                st.dataframe(
                    top_candidates[['Name', 'Email', 'Field', 'Resume Score', 
                                  'Skill Match %', 'Match Score %', 'Skills']],
                    height=600,
                    column_config={
                        "Resume Score": st.column_config.ProgressColumn(
                            "Resume Score",
                            help="The candidate's resume score",
                            format="%f",
                            min_value=0,
                            max_value=100,
                        ),
                        "Skill Match %": st.column_config.ProgressColumn(
                            "Skill Match %",
                            min_value=0,
                            max_value=100,
                        ),
                        "Match Score %": st.column_config.ProgressColumn(
                            "Overall Match %",
                            min_value=0,
                            max_value=100,
                        )
                    }
                )
                
                # Download button
                csv = top_candidates.to_csv(index=False)
                st.download_button(
                    "📥 Download Top Candidates",
                    csv,
                    "top_candidates.csv",
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"Error in candidate selection: {str(e)}")
                st.error("Please check if the database contains valid candidate data")