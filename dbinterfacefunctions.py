import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def connectDroneReconDB():
    """Establish connection to the DroneRecon database

    Returns:
        psycopg2 objects: cursor, conn
    """
    try:
        host = "dronerecon-postgress-server.postgres.database.azure.com"
        dbname = "dronerecon-database"
        user = "USERNAME" # Replace with your username
        password = "PASSWORD" # Replace with your password
        sslmode = "require"

        # Construct connection string
        conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
        conn = psycopg2.connect(conn_string)
        print("Connection established")

        cursor = conn.cursor()
        # Print PostgreSQL Connection properties
        print ( conn.get_dsn_parameters(),"\n")

        # Print PostgreSQL version
        cursor.execute("SELECT version();")
        record = cursor.fetchone()
        print("You are connected to - ", record,"\n")

    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
        return 1
    return cursor, conn


def getRecruitmentInfo(external_study_id,cursor=None):
    """
    Get the recruitment info for a given external study id (e.g., Prolific study ID)
    
    Returns:
        pandas.DataFrame: recruitment info, such as subject IDs, for that study
    """
    if cursor is None:
        cursor, _ = connectDroneReconDB()
    cursor.execute('rollback')

    query_text = f"""SELECT  
                        Subject.id AS subject_id, 
                        Subject."external_ID" AS external_ID,
                        Recruitment.task AS task,
                        Recruitment.notes AS notes
                    FROM drone_recon_recruitment Recruitment, 
                        drone_recon_subject Subject
                    WHERE Recruitment.prolific_study_id='{external_study_id}'
                        and Recruitment.subject_id = Subject.id
                        """

    column_names = ['subject_id','external_id','task','notes']
    # Use query
    cursor.execute(query_text)
    sessions = cursor.fetchall()
    data = pd.DataFrame(sessions, columns=column_names)

    return data


def getSessionData(start_date,end_date,cursor=None,external_study_id=None,task=None):
    """Get the session data for a given date range and external study id
    
    Args:
        start_date (str): start date in format 'YYYY-MM-DD'
        end_date (str): end date in format
        cursor (psycopg2 cursor, optional): cursor to use. Defaults to None.
        external_study_id (str, optional): external study id to use. Defaults to None.
        task (str, optional): task to use. Defaults to None.
    
    Returns:
        pandas.DataFrame: session data for the given dates and external study id
    """
    if cursor is None:
            cursor, _ = connectDroneReconDB()
    cursor.execute('rollback')
    query_text = f"""SELECT  
                            Subject."external_ID" AS external_ID,
                            Subject."external_source" AS external_source,
                            Session.id AS session_id,
                            Session."external_study_ID" AS external_study_ID,
                            Session.subject_id AS subject_id, 
                            Session.task AS task,
                            Session.passed_attention_check AS passed_attention_check,
                            Session.session_completed AS session_completed,
                            Session.questionnaire_completed AS questionnaire_completed,
                            Session.task_completed AS task_completed,
                            Session.payment_token AS payment_token,
                            Session.final_performance AS final_performance,
                            Session.start_time AS start_time,
                            Session.end_time AS end_time,
                            Session.browser AS browser
            FROM drone_recon_session Session, 
                    drone_recon_subject Subject
            WHERE Session.start_time BETWEEN 
                    date '{start_date}' and 
                    date '{end_date}'
                    and Session.subject_id = Subject.id%s%s
                    """
    if external_study_id is not None:
        external_study_id_text = '\nand Session."external_study_ID" = ' + f"'{external_study_id}'"
    else:
        external_study_id_text = ''
    if task is not None:
        task_text = '\nand Session.task = ' + f"'{task}'"
    else:
        task_text = ''
    query_text = query_text % (external_study_id_text, task_text)
    column_names = ['external_ID',"external_source",'session_id','external_study_ID','subject_id','task',\
        'passed_attention_check','session_completed','questionnaire_completed',\
        'task_completed','payment_token','final_performance','start_time','end_time','browser']
    cursor.execute(query_text)
    sessions = cursor.fetchall()
    data = pd.DataFrame(sessions, columns=column_names)
    if len(data)>0:
        data['session_duration'] = (data.end_time - data.start_time).dt.total_seconds()
    else:
        print('No data found')
    data = data.drop(columns='end_time')
    return data


def getPriorProlificRecruitment(start_date=None,end_date=None,cursor=None):
    """
    Get list of all subjects who have been recruited for studies within a given date range
    
    Args:
        start_date (str, optional): start date in format 'YYYY-MM-DD'. Defaults to None.
        end_date (str, optional): end date in format
        cursor (psycopg2 cursor, optional): cursor to use. Defaults to None.
        
    Returns:
        pandas.DataFrame: recruitment info, such as subject IDs, for that study
    """
    if start_date is None:
        start_date = '2023-05-27'
    if end_date is None:
        end_date=(datetime.now()+timedelta(days=2)).strftime("%Y-%m-%d")
    if cursor is None:
        cursor, conn = connectDroneReconDB()
    cursor.execute('rollback')
    query_text = f"""SELECT  
                                Recruitment.id AS recruitment_id,
                                Subject."external_ID" AS external_ID,
                                Recruitment.prolific_study_id AS external_study_ID,
                                Recruitment.time AS time,
                                Recruitment.task AS task,
                                Recruitment.notes AS notes,
                                Recruitment.source AS source,
                                Recruitment.accepted AS accepted
                    FROM drone_recon_recruitment Recruitment, 
                            drone_recon_subject Subject
                    WHERE Recruitment.time BETWEEN 
                        date '{start_date}' and 
                        date '{end_date}' and
                        Recruitment.subject_id = Subject.id
                        """
    cursor.execute(query_text)
    sessions = cursor.fetchall()
    column_names = ['recruitment_id','external_ID','external_study_ID','time','task','notes','source','accepted']
    data = pd.DataFrame(sessions, columns=column_names)
    return data


def getAnalysisData(cursor=None,start_date='2022-11-05',end_date='2022-11-17',external_source='prolific',task=None,
    session_completed_bool=True):
    """
    Get the analysis data for a given date range and external source
    
    Args:
        start_date (str): start date in format 'YYYY-MM-DD'
        end_date (str): end date in format
        cursor (psycopg2 cursor, optional): cursor to use. Defaults to None, in which case it will be created.
        external_source (str, optional): external source from which subjects were recruited. Defaults to 'prolific'.
        task (str, optional): task to use. Defaults to None.
        session_completed_bool (bool, optional): whether to use only completed sessions. Defaults to True.
        
    Returns:
        pandas.DataFrame: session data for the given dates and external study id"""
    if cursor is None:
        cursor, _ = connectDroneReconDB()
    cursor.execute('rollback')
    # We added fields for Prolific. To get the old data, use legacy
    print('Running query to collect analysis data')
    query_text = f"""SELECT  
                                Session.id AS session_id,
                                Session.subject_id AS subject_id, 
                                Session."external_study_ID" AS external_study_ID,
                                Session.substances AS substances,
                                Session.sleep_quality AS sleep_quality,
                                Session.sleep_quantity AS sleep_quantity,
                                Session.start_time AS start_time,
                                Strategy.prompt as strategy_prompt,
                                Strategy.response as strategy_response,
                                Subject."external_ID" AS external_ID,
                                Trial.response AS response,
                                Trial.confidence AS confidence,
                                Trial.correct_class as correct_class,
                                Trial.correct AS correct,
                                Trial.rt_classification AS rt_classification,
                                Trial.rt_confidence as rt_confidence,
                                Trial.feedback_given as feedback_given,
                                Trial.block AS block,
                                Trial.trial_number AS trial_number,
                                Trial.stimulus_id AS stimulus_id,
                                Stim.name AS name,
                                Stim.use AS use
                    FROM drone_recon_session Session, 
                            drone_recon_subject Subject, 
                            drone_recon_trial Trial,
                            drone_recon_stimulus Stim,
                            drone_recon_strategy Strategy
                    WHERE Session.start_time BETWEEN 
                        date '{start_date}' and 
                        date '{end_date}'%s
                        and Session.subject_id = Subject.id%s
                        and Strategy.session_id = Session.id
                        and Subject.external_source = '{external_source}'
                        and Trial.session_id = Session.id
                        and Trial.stimulus_id = Stim.id;
                        """ 

    # Allow the specification of a specific task
    column_names = ['session_id', 'subject_id', 'external_study_id',
        'substances','sleep_quality', 'sleep_quantity','start_time','strategy_prompt','strategy_response','external_id','response',
        'confidence','correct_class','correct', 'rt_classification', 'rt_confidence','feedback_given','block','trial_number',
        'stimulus_id','name', 'use']
    if task is not None:
        task_txt = f"\n                        and Session.task = '{task}'"
    else:
        task_txt = ''
    if session_completed_bool:
        session_completed_text = '\n                        and Session.session_completed = true'
    else:
        session_completed_text = ''
    # Update query
    query_text = query_text  % (session_completed_text, task_txt)
    # Use query
    cursor.execute(query_text)
    sessions = cursor.fetchall()
    data = pd.DataFrame(sessions, columns=column_names)
    #Sort them by time
    data.index = data['trial_number'] + data.start_time.astype(int)
    data = data.sort_index()
    #Clean the dataframe of unused columns
    data.reset_index(drop=True,inplace=True)
    return data


def changePaymentListingDB(cursor, conn, session_id, issue_payment=True):
    """
    Change the payment listing in the database
    
    Args:
        cursor (psycopg2 cursor): cursor to use
        conn (psycopg2 connection): connection to use
        session_id (int): session id to change
        issue_payment (bool, optional): whether to issue payment. Defaults to True.
        
    Returns:
        None
    """
    print('Changing listing in db')   
    sql = f""" UPDATE drone_recon_session
                SET payment_issued = {str(issue_payment).lower()}
                WHERE id = {session_id}"""
    cursor.execute('rollback')
    cursor.execute(sql)
    conn.commit()


def getQuestionnaireData(cursor=None,start_date='2020-12-07',end_date='2020-12-10'):
    """
    Get the questionnaire data for a given date range
    
    Args:
        start_date (str): start date in format 'YYYY-MM-DD'
        end_date (str): end date in format
        cursor (psycopg2 cursor, optional): cursor to use. Defaults to None, in which case it will be created.
        
    Returns:
        pandas.DataFrame: questionnaire data for the given dates
    """
    if cursor is None:
        cursor, _ = connectDroneReconDB()
    cursor.execute('rollback')
    query_text = f"""SELECT  
                            Session.id AS session_id,
                            Session.subject_id AS subject_id, 
                            Session.task AS task,
                            Session.passed_attention_check AS passed_attention_check,
                            Session.session_completed AS session_completed,
                            Subject."external_ID" AS external_ID,
                            Session."external_study_ID" AS external_study_ID,
                            Session."external_session_ID" AS external_session_ID,
                            QuestionnaireQ.questionnaire_name AS questionnaire_name,
                            QuestionnaireQ.subscale AS subscale,
                            QuestionnaireQ.possible_answers AS possible_answers,
                            QuestionnaireQ.question AS question,
                            QuestionnaireQ.answer AS answer,
                            QuestionnaireQ.questionnaire_question_number AS questionnaire_question_number,
                            Session.substances AS substances,
                            Subject.gender AS gender,
                            Subject.sex AS sex,
                            Subject.age AS age,
                            Subject.education AS education,
                            Subject.psych_history AS psych_history
                FROM drone_recon_session Session, 
                        drone_recon_subject Subject,
                        drone_recon_questionnaireq QuestionnaireQ
                WHERE Session.start_time BETWEEN 
                    date '{start_date}' and 
                    date '{end_date}'
                    and Session.subject_id = Subject.id
                    and QuestionnaireQ.session_id = Session.id
                    and Session.questionnaire_completed = True
                    """
    column_names = ['session_id','subject_id','task','passed_attention_check','session_completed','external_id','external_study_ID',
        'external_session_ID','questionnaire_name','subscale','possible_answers','question','answer','questionnaire_question_number',
        'substances','gender','sex','age','education','psych_history']
    cursor.execute(query_text)
    sessions = cursor.fetchall()
    data = pd.DataFrame(sessions, columns=column_names)

    return data