# database_utils.py
from contextlib import contextmanager
import datetime
import pymysql
import streamlit as st
import json  # Add this import

def get_persistent_connection():
    if 'db_conn' not in st.session_state or not st.session_state.db_conn.open:
        st.session_state.db_conn = pymysql.connect(
            host="localhost",
            user="root",
            password="",
            db="cv",
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,  # Changed to DictCursor for easier handling
            autocommit=False
        )
    return st.session_state.db_conn

def close_persistent_connection():
    if 'db_conn' in st.session_state:
        try:
            st.session_state.db_conn.close()
            del st.session_state['db_conn']
        except:
            pass

@contextmanager
def db_connection():
    """Alternative context manager for non-persistent connections"""
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        db='cv',
        charset='utf8mb4',  # Added charset
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )
    try:
        yield conn
    finally:
        conn.close()

def insert_data(name, email, res_score, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses):
    """Insert user data into database with explicit column names"""
    try:
        with db_connection() as conn:
            with conn.cursor() as cursor:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Explicit column names in INSERT statement
                sql = """INSERT INTO data (
                            Name, 
                            Email_ID, 
                            resume_score, 
                            Timestamp, 
                            Page_no,
                            Predicted_Field, 
                            User_level, 
                            Actual_skills,
                            Recommended_skills, 
                            Recommended_courses
                         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                
                # Execute with parameterized values
                cursor.execute(sql, (
                    name, 
                    email, 
                    str(res_score), 
                    timestamp, 
                    str(no_of_pages),
                    reco_field, 
                    cand_level, 
                    json.dumps(skills),
                    json.dumps(recommended_skills), 
                    json.dumps(courses)
                ))
                
                # Verify insertion
                if cursor.rowcount == 1:
                    conn.commit()
                    st.success("Data successfully inserted into database!")
                    return True
                else:
                    conn.rollback()
                    st.error("No rows were inserted")
                    return False
                    
    except pymysql.Error as e:
        st.error(f"Database error: {e.args[1]} (Error code: {e.args[0]})")
        return False
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return False
    
def create_table_if_not_exists():
    """Create the user_data table if it doesn't exist"""
    try:
        with db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data (
                        ID INT NOT NULL AUTO_INCREMENT,
                        Name varchar(500) NOT NULL,
                        Email_ID VARCHAR(500) NOT NULL,
                        resume_score VARCHAR(8) NOT NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        Page_no VARCHAR(5) NOT NULL,
                        Predicted_Field VARCHAR(255) NOT NULL,  # Changed from BLOB
                        User_level VARCHAR(255) NOT NULL,      # Changed from BLOB
                        Actual_skills TEXT NOT NULL,           # Changed from BLOB to TEXT
                        Recommended_skills TEXT NOT NULL,      # Changed from BLOB to TEXT
                        Recommended_courses TEXT NOT NULL,      # Changed from BLOB to TEXT
                        PRIMARY KEY (ID)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                conn.commit()
                return True
    except pymysql.Error as e:
        st.error(f"Database error in create_table: {e}")
        return False
    