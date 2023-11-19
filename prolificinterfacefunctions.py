
from datetime import datetime, timedelta
from sys import displayhook
import numpy as np
import requests
import json 

from dbinterfacefunctions import connectDroneReconDB, getSessionData, getPriorProlificRecruitment, \
    changePaymentListingDB, getRecruitmentInfo


def getProlificRequestHeader(prolific_requester_token=None):
    """Get the header for a request to Prolific

    Args:
        prolific_requester_token (str, optional): the identifying token for a user. Defaults to None.

    Raises:
        ValueError: If no token is provided

    Returns:
        _type_: _description_
    """
    if prolific_requester_token is None:
        raise ValueError(f"To interact with prolific, user must supply an authorization token using the prolific_requester_token variable")
    header = {
            'Authorization': f"Token {prolific_requester_token}"
        }
    return header


def prolificEndpointURLS():
    """stores the endpoints for different prolific interface APIS

    Returns:
        dict: functions and urls
    """
    # Endpoint URLs
    prolific_urls = {
        "url_get_submission_list": 'https://api.prolific.co/api/v1/submissions/',
        "url_post_status_transaction": "https://api.prolific.co/api/v1/submissions/%s/transition/",
        "url_post_set_bonus": "https://api.prolific.co/api/v1/submissions/bonus-payments/",
        "url_post_pay_bonus": "https://api.prolific.co/api/v1/bulk-bonus-payments/%s/pay/",
        "url_message": "https://api.prolific.co/api/v1/messages/",
        "url_get_submission_details": 'https://api.prolific.co/api/v1/submissions/%s/'
    }

    return prolific_urls


def extractStudyDetailsWhitelist(study_details):
    """Gets the white list from the study details

    Args:
        study_details (dict): study details from Prolific

    Returns:
        list: white list of Prolific IDs
    """
    white_list = []
    for i in range(len(study_details['eligibility_requirements'])):
        if 'name' in study_details['eligibility_requirements'][i]['attributes'][0].keys():
            if study_details['eligibility_requirements'][i]['attributes'][0]['name'] == 'white_list':
                white_list.extend(study_details['eligibility_requirements'][i]['attributes'][0]['value'])
    return white_list


def getPastStudies(prolific_requester_token=None):
    """Gets all past studies associated with the requester token

    Args:
        prolific_requester_token (str, optional): user's token from prolific. Defaults to None.

    Returns:
        dict: information about past studies
    """
    url_get_study_list = "https://api.prolific.co/api/v1/studies/"
    header = getProlificRequestHeader(prolific_requester_token=prolific_requester_token)
    r_get_study_list = requests.get(url_get_study_list, headers=header)
    past_studies = r_get_study_list.json()["results"]
    return past_studies


def getAllProlificPriorTaskParticipants(cursor=None):
    """creates a list of all Prolific IDs who have participated in a task. It is useful for excluding them from future studies

    Args:
        cursor (psycopg2 cursor object, optional): cursor object for interfacing with DB. Defaults to None.

    Returns:
        list: subject IDs
    """
    if cursor is None:
        cursor, _ = connectDroneReconDB()
    # Get all dates that studies have been run
    start_date = '2023-05-27'
    end_date=(datetime.now()+timedelta(days=2)).strftime("%Y-%m-%d")
    # Get subjects from the Yale study
    displayhook('Pulling subject IDs who participated previously')
    exclude_subj_ids_recorded = getSessionData(start_date,end_date,cursor=cursor)
    exclude_subj_ids_recorded= exclude_subj_ids_recorded.loc[exclude_subj_ids_recorded.external_source=='prolific','external_ID'].to_list()
    # Get all who have been recruited through Prolific (so we don't miss those who haven't participated yet)
    displayhook('Fetching recruited subjects')
    exclude_subj_ids_recruited = list(getPriorProlificRecruitment(start_date=start_date,end_date=end_date,\
        cursor=cursor).external_ID.unique())
    # exclude_subj_ids_recruited = getProlificPriorRecruitedSubjects()
    # Join and clean out invalid IDs
    exclude_subj_ids = np.unique(exclude_subj_ids_recorded + exclude_subj_ids_recruited)
    exclude_subj_ids = [sub_id for sub_id in exclude_subj_ids if len(sub_id)==24]
    return exclude_subj_ids


def payProlificSubjects(data_session=None,prolific_requester_token=None,estimate_cost=False,
                completion_payment=2.00,performance_payment=3.00,hit_id=None,cursor=None,conn=None,
                skip_balance_check=False,platform_fee_percentage=0.2,accuracy_calibrationconverted_mean_threshold=0.7):
    """
    Pay subjects on Prolific.
    
    Args:
        data_session (pandas dataframe, optional): Dataframe with session data. Defaults to None.
        prolific_requester_token (str, optional): user's token from prolific. Defaults to None.
        completion_payment (float, optional): payment for completing the task. Defaults to 2.00.
        performance_payment (float, optional): payment for performing well on the task. Defaults to 3.00.
        hit_id (str, optional): ID of the HIT. Defaults to None.
        cursor (psycopg2 cursor object, optional): cursor object for interfacing with DB. Defaults to None.
        conn (psycopg2 connection object, optional): connection object for interfacing with DB. Defaults to None.
        skip_balance_check (bool, optional): check the balance in the account to make sure there's enough. Defaults to False.
        platform_fee_percentage (float, optional): percentage of the payment that goes to Prolific. Defaults to 0.2.
        accuracy_calibrationconverted_mean_threshold (float, optional): threshold for accuracy. Defaults to 0.7.
        
    Returns:
        list: base_cost, bonus
    """
    if hit_id is None:
        raise ValueError('Must provide hit_id')
    if cursor is None or conn is None:
        cursor, conn = connectDroneReconDB()
    if data_session is None:
        raise ValueError("The function calling has not provided the DB data")        
    if 'accuracy_calibrationconverted_mean' not in data_session.columns:
        raise ValueError("You must first add the accuracy_calibrationconverted_mean column to data_session.")          
    if prolific_requester_token is None:
        raise ValueError(f"To interact with prolific, user must supply an authorization token using the prolific_requester_token variable")
    
    # Endpoint URLs
    urls = prolificEndpointURLS()

    # Set the prolific requester token
    header = {
        'Authorization': f"Token {prolific_requester_token}"
    }

    # Get the list of submissions
    params = {
        'study': hit_id,
        'limit': 400
    }
    params['study'] = hit_id
    r = requests.get(url = urls["url_get_submission_list"], headers=header, params=params)
    submissions = r.json()["results"]
    print(f'Found {len(submissions)} submissions')

    assignment_ids = []
    worker_ids = []
    status = []
    for submission in submissions:
        assignment_ids.append(submission['id'])
        worker_ids.append(submission['participant_id'])
        status.append(submission['status'])

    #make sure there's enough money in the account
    if not skip_balance_check:
        print('Need to implement balance check for Prolific')

    base_cost, bonus = [0,0], [0,0]
    #Loop through the subjects
    for row in data_session.iterrows():
        if row[1].external_ID in worker_ids:
            indx = worker_ids.index(row[1].external_ID)
        else:
            continue
        # Check if they've already been approved. If so, move on
        if status[indx] != "AWAITING REVIEW":
            continue
        assignment_id = assignment_ids[indx]
        print(f'External id: {row[1].external_ID}, assignment id: {assignment_id}')
        # Check if they entered the completion code
        url = urls["url_get_submission_details"] % (assignment_id)
        r = requests.get(url, headers=header)
        # Check if they passed the attention check. 
        if row[1].passed_attention_check is False:
            print('Subject failed the attention check')
            if not estimate_cost:
                params = {
                    'action': "REJECT",
                    'message': 'You did not complete the session.'
                }
                url = urls["url_post_status_transaction"] % (assignment_ids[indx])
                requests.post(url, headers=header, json=params)
                print('Changing listing in db')
                sql = f""" UPDATE drone_recon_session
                            SET payment_issued = true
                            WHERE id = {row[1].session_id}"""
                cursor.execute('rollback')
                cursor.execute(sql)
                conn.commit()
            continue
        # Let's see if they completed the whole session. If so, approve! 
        if row[1].session_completed:
            print('session done')
            if estimate_cost:
                base_cost[0] += completion_payment
            elif completion_payment > 0:
                # Approve the submission
                params = {
                    'action': 'APPROVE',
                    'message': 'Thanks for completing the session!'
                }
                url = urls["url_post_status_transaction"] % (assignment_ids[indx])
                requests.post(url, headers=header, json=params)
                # Tell them they got it. 
                params = {
                    'study_id': hit_id,
                    'recipient_id': worker_ids[indx],
                    'body': f"You received ${completion_payment} for completing the session. Thanks!"
                }
                requests.post(urls["url_message"], headers=header, json=params)
        # See if they did better than the threshold
        if row[1].accuracy_calibrationconverted_mean is not None and (row[1].accuracy_calibrationconverted_mean > accuracy_calibrationconverted_mean_threshold):
            print('final mean of accuracy and calibration good')
            if estimate_cost:
                bonus[0] += performance_payment
            elif performance_payment > 0:
                 # Set up the bonus
                params = {
                    'study_id': hit_id,
                    'csv_bonuses': f"{worker_ids[indx]},{performance_payment}"
                }
                r = requests.post(urls["url_post_set_bonus"], headers=header, json=params)
                bonus_id = None
                try:
                    response_data = r.json()
                    bonus_id = response_data.get('id')
                except json.JSONDecodeError:
                    print(f"Failed to decode response JSON when issuing the final completion bonus for subject {row[1].external_ID}")
                if bonus_id:
                    # Release the bonus
                    url = urls["url_post_pay_bonus"] % (bonus_id)
                    requests.post(url, headers=header)
                    print('Bonus issued')
                    # Tell them they got it. 
                    params = {
                        'study_id': hit_id,
                        'recipient_id': worker_ids[indx],
                        'body': f"You received a ${performance_payment} bonus for performing well on the task and doing a good job judging your decisions. Thanks for working hard!"
                    }
                    requests.post(urls["url_message"], headers=header, json=params)
        if not estimate_cost:
            print('Changing listing in db')
            sql = f""" UPDATE drone_recon_session
                        SET payment_issued = true
                        WHERE id = {row[1].session_id}"""
            cursor.execute('rollback')
            cursor.execute(sql)
            conn.commit()
    if estimate_cost:
        base_cost[1] = base_cost[0] * platform_fee_percentage
        bonus[1] = bonus[0] * platform_fee_percentage
        print(f'Base subject = {base_cost[0]}, Prolific = {base_cost[1]}, Total base = {sum(base_cost)}')
        print(f'Bonus subject = {bonus[0]}, Prolific = {bonus[1]}, Total bonus = {sum(bonus)}')
        print(f'Total All = {sum(base_cost) + sum(bonus)}')
        return base_cost, bonus
    else:
        return None, None
    
    
def payProlificScreenSubjects(data=None,prolific_requester_token=None,completion_payment=8.00,
                              hit_id=None,cursor=None,conn=None,estimate_cost=False,platform_fee_percentage=0.2):
    """Pay the subjects that have completed the screening study

    Args:
        data (pandas.DataFrame, optional): dataframe of subject data. Defaults to None.
        prolific_requester_token (str, optional): requestor token. Defaults to None.
        completion_payment (float, optional): payment for completion. Defaults to 8.00.
        hit_id (str, optional): study ID. Defaults to None.
        cursor (psycopg2 cursor object, optional): cursor object for interfacing with DB. Defaults to None.
        conn (psycopg2 connection object, optional): connection object for interfacing with DB. Defaults to None.
        estimate_cost (bool, optional): whether to estimate cost or pay. Defaults to False.
        platform_fee_percentage (float, optional): percentage of payment that goes to prolific. Defaults to 0.2.

    Raises:
        ValueError: if no HIT ID is provided
        ValueError: if there's no study data
        ValueError: if there's an issue with the requestor token
        ValueError: if the subject did multiple sessions.

    Returns:
        float: cost estimate, if estimate_cost is True
    """
    if hit_id is None:
        raise ValueError('Must provide hit_id')
    if cursor is None or conn is None:
        cursor, conn = connectDroneReconDB()
    if data is None:
        raise ValueError("The function calling has not provided the DB data")                  
    if prolific_requester_token is None:
        raise ValueError(f"To interact with prolific, user must supply an authorization token using the prolific_requester_token variable")
    if 'external_ID' not in data.columns: # Correct for the changing of case that sometimes happens. 
        data['external_ID'] = data['external_id']

    # Clean up multiple sessions
    for external_id in data.external_ID.unique():
        if len(external_id) < 11:
            continue
        if data.loc[data.external_ID == external_id,'session_id'].nunique() > 1:
            print(f'{external_id} has multiple sessions.')
            session_completed_list = []
            for session_id in data.loc[data.external_ID == external_id,'session_id'].unique():
                session_completed_list.append(data.loc[(data.external_ID == external_id) &\
                    (data.session_id==session_id),'session_completed'].iloc[0])
            if sum(session_completed_list) > 1:
                raise ValueError('Subject completed multiple sessions')
            if True in session_completed_list:
                print('dropping duplicate rows for subject with multiple sessions')
                # session_ids = data.loc[data.external_ID == external_id,'session_id'].unique()[np.array(session_completed_list)<1]
                data = data.drop(data.index[data.session_id.isin(data.loc[data.external_ID == external_id,'session_id'].unique()\
                    [np.array(session_completed_list)<1])])
                
    # Cut data down to just that Prolific experiment
    if hit_id is not None:
        data = data.loc[data.external_study_ID==hit_id,:]
        
    # Endpoint URLs
    urls = prolificEndpointURLS()

    # Set the prolific requester token
    header = {
        'Authorization': f"Token {prolific_requester_token}"
    }

    # Get the list of submissions
    params = {
        'study': hit_id,
        'limit': 1000
    }
    params['study'] = hit_id
    r = requests.get(url = urls["url_get_submission_list"], headers=header, params=params)
    submissions = r.json()["results"]
    print(f'Found {len(submissions)} submissions')

    assignment_ids = []
    worker_ids = []
    status = []
    for submission in submissions:
        assignment_ids.append(submission['id'])
        worker_ids.append(submission['participant_id'])
        status.append(submission['status'])

    base_cost, bonus = [0,0], [0,0]
    #Loop through the subjects
    for row in data.iterrows():
        if row[1].external_ID in worker_ids:
            indx = worker_ids.index(row[1].external_ID)
        else:
            continue
        # Check if they've already been approved. If so, move on
        if status[indx] != "AWAITING REVIEW":
            continue
        assignment_id = assignment_ids[indx]
        print(f'External id: {row[1].external_ID}, assignment id: {assignment_id}')
        # Check if they entered the completion code
        url = urls["url_get_submission_details"] % (assignment_id)
        r = requests.get(url, headers=header)
        if row[1].payment_token != r.json()['entered_code']:
            print('Provided invalid completion code')
            if not estimate_cost:
                params = {
                    'action': "REJECT",
                    'message': 'You did not enter the code indicating successful completion of the task.'
                }
                url = urls["url_post_status_transaction"] % (assignment_id)
                requests.post(url, headers=header, json=params)
                changePaymentListingDB(cursor, conn, row[1].session_id, issue_payment=True)
            continue            
        # Check if they passed the attention check. 
        if row[1].passed_attention_check is False:
            print('Subject failed the attention check')
            if not estimate_cost:
                params = {
                    'action': "REJECT",
                    'message': 'You failed the internal checks for attention.'
                }
                url = urls["url_post_status_transaction"] % (assignment_ids[indx])
                requests.post(url, headers=header, json=params)
                changePaymentListingDB(cursor, conn, row[1].session_id, issue_payment=True)
            continue
        # Let's see if they completed the whole session.
        print('Passed attention check')
        if row[1].session_completed:
            print('Completed session')
            if estimate_cost:
                base_cost[0] += completion_payment
            else:
                params = {
                    'action': 'APPROVE',
                    'message': 'Thanks for answering the questions! Please be on the lookout for an invitation to our future studies.'
                }
                url = urls["url_post_status_transaction"] % (assignment_ids[indx])
                requests.post(url, headers=header, json=params)
                changePaymentListingDB(cursor, conn, row[1].session_id, issue_payment=True)
        else:
            if not estimate_cost:
                params = {
                        'action': "REJECT",
                        'message': 'You did not complete the session.'
                    }
                url = urls["url_post_status_transaction"] % (assignment_ids[indx])
                requests.post(url, headers=header, json=params)
                changePaymentListingDB(cursor, conn, row[1].session_id, issue_payment=True)
    if estimate_cost:
        base_cost[1] = base_cost[0] * platform_fee_percentage
        print(f'Base subject = {base_cost[0]}, Prolific = {base_cost[1]}, Total = {sum(base_cost)}')
        return base_cost
    else:
        return None


def studyParams(prolific_ids=None,description_type='game',payment_participate=4,study_description=None,
    payment_performance=2,completion_code=None,failed_attention_code=None,total_available_places=2,cursor=None,
    estimated_completion_time=None,internal_name='Internal Pilot',exclude_prior_subjects=None,exclude_subj_ids=None,
    webapp_url='https://dronerecongame.azurewebsites.net',webapp_use=None,project_id='64727db217702aac526b4a5b',
    title=None):
    if webapp_use is None:
        webapp_use = ''
    elif webapp_use in ['screen','task','both']:
        webapp_use = f'WEBAPP_USE={webapp_use}&'
    else:
        raise ValueError(f'{webapp_use} invalid for webapp_use. It can be "screen", "task", "both" or None')
    if (description_type == 'game') or description_type == 'task':
        display('Creating params for a game')
        if study_description is None:
            study_description = \
                '<p>Based on your answers to our prior study, we would like to recruit you to play a game ' + \
                'where you perform reconnaissance on different types of drones! You will:</p>' + \
                '<ul>' + \
                '<li>Complete in a tutorial that you must pass to move onto the main game (~5 min);</li>' + \
                '<li>Play the main game (~10-20 min).</li>' + \
                '</ul> ' + \
                '<p>Payment:</p>' + \
                '<ul>' + \
                f'<li>${payment_participate} for active participation (what the hourly rate is based on);</li>' + \
                f'<li>${payment_performance} bonus for high performance (second bonus).</li>' + \
                '</ul> ' +\
                '</p>'
        if title is None:
            title = "We need YOU for drone recon!"
        if completion_code is None:
            completion_code = 'RBUEUHDM'
        if failed_attention_code is None:
            failed_attention_code = 'QFBERJMZ'
        if estimated_completion_time is None:
            estimated_completion_time=15
        if exclude_prior_subjects is None:
            exclude_prior_subjects = False
    elif description_type == 'screen':
        display('Creating params for a screening study')
        if study_description is None:
            study_description = \
                '<p>Answer questions with the opportunity for future recruitment!</p>' + \
                '<br>' + \
                '<p>Answer a series of confidential questions. According to those answers, you may be recruited ' + \
                'to a future, high-paying studies involving online games.</p>'
        if title is None:
            title = "Answer survey questions with potential for game recruitment"
        if completion_code is None:
            completion_code = 'HWIYLLCB'
        if failed_attention_code is None:
            failed_attention_code = 'IBHJQBEP'
        if estimated_completion_time is None:
            estimated_completion_time=20
        if exclude_prior_subjects is None:
            exclude_prior_subjects = True
    elif description_type == 'both':
        display('Creating params for a screening study')
        if study_description is None:
            study_description = \
                '<p>Answer questions and play a game!</p>' + \
                '<br>' + \
                '<p>Answer a series of confidential questions, then play a fun and challending drone reconnaissance game.</p>'
        if title is None:
            title = "Answer some questions and play a Drone Recon!"
        if completion_code is None:
            completion_code = 'NYPFTPXN'
        if failed_attention_code is None:
            failed_attention_code = 'FFVRMXQU'
        if estimated_completion_time is None:
            estimated_completion_time=20
        if exclude_prior_subjects is None:
            exclude_prior_subjects = True
    if prolific_ids is not None:
        display('Including specific Prolific subjects')
        eligibility_requirements = [
            {
                "attributes": [
                {
                    "name": "white_list",
                    "value": prolific_ids,
                }
                ],
                "_cls": "web.eligibility.models.CustomWhitelistEligibilityRequirement"
            }
            ]
        total_available_places = len(prolific_ids)
    else:
        eligibility_requirements = []
    if exclude_prior_subjects:
        display('Excluding prior subjects')
        if exclude_subj_ids is None:
            exclude_subj_ids = getAllProlificPriorTaskParticipants(cursor=cursor)
        if len(exclude_subj_ids) > 0:
            # Add to eligibility requirements
            eligibility_requirements.append(
                {
                "attributes": [
                {
                    "name": "black_list",
                    "value": exclude_subj_ids
                }
                ],
                "_cls": "web.eligibility.models.CustomBlacklistEligibilityRequirement"
            })

    amount = payment_participate*100 # $4 for CG, $5 for EG

    params = {
        "name": title,
        "internal_name": internal_name,
        "description": study_description,
        "external_study_url": webapp_url + "?" + webapp_use + \
            "PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}",
        "prolific_id_option": "url_parameters",
        "completion_code": completion_code,
        "failed_attention_code": failed_attention_code,
        "completion_option": "code",
        "total_available_places": total_available_places,
        "estimated_completion_time": estimated_completion_time,
        "reward": amount,
        "device_compatibility": [
            "desktop"
        ],
        "peripheral_requirements": [],
        "project": project_id, # 
        "eligibility_requirements": eligibility_requirements
    }

    return params


def createReplicationStudy(start_date,end_date,target_study_hits=[],recruitment_note='this is a recruitment study',
    task="category_metacog-v0",get_estimate=True,internal_name='Internal Pilot',cursor=None,completion_code='YHFDJWKP',
    failed_attention_code='QZLBMGXE',conn=None,eligible_prolific_subject_ids=[],start_date_recruitment='2023-06-26',
    payment_participate=4,payment_performance=2,drop_retest=True,drop_recruited=True,prolific_requester_token=None,
    n_subjects=None,webapp_url='your_webapp_url',webapp_use='task',message_body=None,
    study_title="The agency needs you!!",study_description=None,project_id='project_id'):
    """
    Create a study to recruit subjects for a replication study.
    
    Args:
        start_data (str YYYY-MM-DD): start of when to look for subject data.
        end_date (str YYYY-MM-DD): end of when to look for subject data.
        target_study_hits (list, optional): list of study IDs to recruit from. Defaults to [].
        recruitment_note (str, optional): note to include in the recruitment listing. Defaults to 'this is a recruitment study'.
        task (str, optional): task to recruit for. Defaults to "category_metacog-v0".
        get_estimate (bool, optional): whether to just get an estimate of the number of subjects. Defaults to True.
        internal_name (str, optional): internal name of the study. Defaults to 'Internal Pilot'.
        cursor (psycopg2 cursor object, optional): cursor object for interfacing with DB. Defaults to None.
        completion_code (str, optional): completion code for the study. Defaults to 'YHFDJWKP'.
        failed_attention_code (str, optional): code for failing the attention check. Defaults to 'QZLBMGXE'.
        conn (psycopg2 connection object, optional): connection object for interfacing with DB. Defaults to None.
        eligible_prolific_subject_ids (list, optional): list of Prolific IDs to recruit from. Defaults to [].
        start_date_recruitment (str YYYY-MM-DD, optional): start of when to look for prior recruitment. Defaults to '2023-06-26'.
        payment_participate (float, optional): payment for participating. Defaults to 4.
        payment_performance (float, optional): payment for performance. Defaults to 2.
        drop_retest (bool, optional): whether to drop subjects who have already done a retest. Defaults to True.
        drop_recruited (bool, optional): whether to drop subjects who have already been recruited. Defaults to True.
        prolific_requester_token (str, optional): requestor token. Defaults to None.
        n_subjects (int, optional): number of subjects to recruit. Defaults to None.
        webapp_url (str, optional): URL for the webapp. Defaults to 'your_webapp_url'.
        webapp_use (str, optional): whether to use the webapp for the screen, task, or both. Defaults to 'task'.
        message_body (str, optional): message to send to subjects. Defaults to None.
        study_title (str, optional): title of the study. Defaults to "The agency needs you!!".
        study_description (str, optional): description of the study. Defaults to None.
        project_id (str, optional): project ID. Defaults to 'project_id'.
        
    Returns:
        str: study ID
        list: list of external IDs
    """
    # Get all sessions from the period
    if (cursor is None) or (conn is None):
        cursor, conn = connectDroneReconDB()
    display('Load session data')
    data = getSessionData(start_date,end_date,cursor=cursor,external_study_id=None)
    if len(data) == 0:
        display('No session data found')
    data.columns= data.columns.str.lower()
    print(f'After loading session data, have {data.external_id.nunique()} unique subjects')
    
    #Filter it down to just the target studies
    if len(target_study_hits) > 0:
        data = data.loc[data.external_study_ID.isin(target_study_hits),:]
    print(f'After filtering for target HITs, have {data.external_id.nunique()} unique subjects')

    # Use only subjects on the eligiblee subjects list
    if (eligible_prolific_subject_ids is not None) and (len(eligible_prolific_subject_ids) > 0):
        data = data.loc[data.external_id.isin(eligible_prolific_subject_ids),:]
        print(f'After using only elibigle list subject, have {data.external_id.nunique()} unique subjects')

    if drop_retest:
        # Drop subjects who have already done a retest
        data = data.loc[~data.external_id.isin(data.loc[data.task.str.contains('_retest'),'external_id']),:]
        print(f'After dropping subjects who have done retest, have {data.external_id.nunique()} unique subjects')

    # Exclude subjects who have been recruited to a re-test
    if drop_recruited:
        end_date_recruitment = (datetime.now()+timedelta(days=2)).strftime("%Y-%m-%d")
        data_recruit = getPriorProlificRecruitment(start_date=start_date_recruitment,end_date=end_date_recruitment,\
                cursor=cursor)
        data_recruit.columns= data_recruit.columns.str.lower()
        data = data.loc[~data.external_id.isin(data_recruit.loc[data_recruit.task.str.contains('_retest'),'external_id']),:]
        print(f'After excluding recruited subjects, we have {data.external_id.nunique()} unique subjects')

    # Extract the subject IDs
    # external_ids = data.loc[data.task.str.contains(task),'external_id'].unique()
    external_ids = data.external_id.unique()
    
    print(f'final count is {len(external_ids)} unique subjects')

    if (n_subjects is not None) and (n_subjects < len(external_ids)):
        external_ids = np.array(external_ids)[np.random.choice(len(external_ids),n_subjects,replace=False)]
        display(f'Subselected {n_subjects} subjects.')
        
    # Just provide how many subjects are eligible
    if get_estimate:
        return None, external_ids

    # For those that meet criteria, create study and invite them to participate.
    # Create the params
    study_params = studyParams(prolific_ids=list(external_ids),description_type='game',payment_participate=payment_participate,
        title=study_title,payment_performance=payment_performance,completion_code=completion_code,failed_attention_code=failed_attention_code,
        total_available_places=len(external_ids),cursor=cursor,webapp_url=webapp_url,estimated_completion_time=12,
        internal_name=internal_name,exclude_prior_subjects=False,webapp_use=webapp_use,study_description=study_description,
        project_id=project_id)
    display('Study params created.')
    # Post the draft
    header = getProlificRequestHeader(prolific_requester_token=prolific_requester_token)
    url_post_study_draft = 'https://api.prolific.co/api/v1/studies/'
    r_create_draft = requests.post(url_post_study_draft, headers=header, json=study_params)
    try: 
        r_create_draft.json()['id']
    except: 
        raise Exception('Draft study not posted. Error: %s')
    display('Draft Study posted.')
    # Launch the study
    study_id = r_create_draft.json()['id']    
    url_post_study_publish = 'https://api.prolific.co/api/v1/studies/%s/transition/'
    params = {
        "action": "PUBLISH"
    }
    r_publish_draft = requests.post(url_post_study_publish % (study_id), headers=header, json=params)
    display('Study launched.')
    # Create the DB listing for each subject who has been invited. Include external study ID
    for external_id in external_ids:
        #Create DB listing
        sql = f"""  INSERT INTO drone_recon_recruitment 
                        (subject_id, session_id, prolific_study_id, time, task, notes, source, accepted)
                    VALUES ((SELECT id from drone_recon_subject 
                        WHERE   "external_ID"='{external_id}'), 
                                NULL, 
                                '{study_id}', 
                                '{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}', 
                                '{task + '_retest'}', 
                                '{recruitment_note}', 
                                'prolific', 
                                NULL);  """
        cursor.execute('rollback')
        cursor.execute(sql)
        conn.commit()
        #Send them a message
        url_message = "https://api.prolific.co/api/v1/messages/"
        if message_body is None:
            message_body = f"The agency needs you again for Drone Recon! Check out the study '{study_title}' See you soon agent!"
        params = {
                    'study_id': study_id,
                    'recipient_id': external_id,
                    'body': message_body
                }
        requests.post(url_message, headers=header, json=params)
    display('DB updated and messages sent.')
    # Return study information
    return study_id, external_ids


def getIDsRecruitedNotParticipated(start_date,end_date,external_study_id,cursor=None):
    """
    Find the external IDs of users who were recruited to a study, but have not yet participated in it.
    
    Args:
        start_date (str YYYY-MM-DD): _description_
        end_date (str YYYY-MM-DD): _description_
        external_study_id (str): _description_
        message_body (str, optional): _description_. Defaults to None.
        cursor (psycopg2 cursor object, optional): cursor object. Defaults to None.
    
    Returns:
        list: list of external IDs
    """
    if cursor is None:
        cursor, _ = connectDroneReconDB()
    #Get the IDs
    data_recruitment = getRecruitmentInfo(external_study_id=external_study_id,cursor=cursor)
    data_sessions_task = getSessionData(start_date,end_date,cursor=cursor,external_study_id=external_study_id)
    external_ids_not_participated = list(set(data_recruitment.external_id) ^ set(data_sessions_task.external_ID))
    
    return external_ids_not_participated


def messageReruitedNotParticipated(start_date,end_date,external_study_id,message_body=None,cursor=None,prolific_requester_token=None):
    """Sends a message to the subjects who were recruited, but have not yet participated

    Args:
        start_date (str YYYY-MM-DD): date the study was launched
        end_date (str YYYY-MM-DD): last date to look for subjects who have participated
        external_study_id (str): Prolific study ID
        message_body (_type_, optional): The message to send subjects. Defaults to None.
        cursor (psycopg2 cursor object, optional): the cursor object for subjects. Defaults to None.
        prolific_requester_token (str, optional): token sent from the requestor. Defaults to None.
    """
    if message_body is None:
        message_body = 'You did a great job answering a very long list of questions. We would love for you to ' +\
            "participate in our game! Please have a look out for the study title, 'We need YOU for drone recon!"
    if cursor is None:
        cursor, _ = connectDroneReconDB(resource_group='uminn')
    #Get the IDs
    external_ids_not_participated = getIDsRecruitedNotParticipated(start_date,end_date,external_study_id,cursor=cursor)
    #Send them a message
    header = getProlificRequestHeader(prolific_requester_token=prolific_requester_token)
    url_message = "https://api.prolific.co/api/v1/messages/"
    for external_id in external_ids_not_participated:
        params = {
                    'study_id': external_study_id,
                    'recipient_id': external_id,
                    'body': message_body
                }
        try:
            requests.post(url_message, headers=header, json=params)
            print(f'{external_id}, message sent.')
        except:
            print(f'{external_id}, message failed.')