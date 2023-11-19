"""
This script contains functions that facilitate analysis of task and questionnaire data from metacognitive category learning task. 

By Warren Woodrich Pettine, M.D.
Last updated 2023-11-18
"""


import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import ast
import scipy.stats as stats
from scipy.stats import zscore
import psycopg2
import requests
from datetime import datetime, timedelta

import functools as ft


## GENERAL FUNCTIONS ##
def cleanDataOnlyProlificSubjects(data,min_id_length=23):
    """
    Removes subjects that are not from Prolific
    Args:
        data (pandas.DataFrame): dataframe with "external_id" as a field containing the Prolific ID
        min_id_length (int): minimum length of the Prolific ID

    Returns:
        pandas.DataFrame: copy of dataframe contianing only the identified Prolific subjects
    """
    # Get the external IDs
    external_ids = data.external_id.unique()
    idx = [len(external_id)>min_id_length for external_id in external_ids] & (external_ids != '62ab189e3d87cd46615a50d5')
    external_ids = external_ids[idx]
    return data.loc[data.external_id.isin(external_ids),:]


def renameDfColumns(df,key_subscales=None,key_plot_names=None):
    """
    Remames the columns of a dataframe to nicer-formatted names for plotting

    Args:
        df (_type_): _description_
        key_subscales (_type_, optional): Columns to rename. If None, it renames several subscales. Defaults to None.
        key_plot_names (_type_, optional): The new names. If None, it provides the corresponding subscales. Defaults to None.

    Returns:
        pandas.DataFrame: Same dataframe with renamed columns
    """
    if key_subscales is None:
       key_subscales = ['bfi10_extraversion', 'bfi10_agreeableness',
        'bfi10_conscientiousness', 'bfi10_neuroticism', 'bfi10_openness',
        'bapq_aloof', 'bapq_pragmatic_language', 'bapq_rigid',
        'asrs_inattention', 'asrs_hyperactivity_impulsivity', 'phq9_na']
    if key_plot_names is None:
        key_plot_names = ['BFI-10: Extraversion','BFI-10: Agreeableness','BFI-10: Conscientiousness',
                  'BFI-10: Neuroticism','BFI-10: Openness','BAPQ: Aloof','BAPQ: PragLang',
                  'BAPQ: Rigidity','ASRS: Inattention','ASRS: Hyperactivity/Impulsivity','PHQ9']
    df.rename(columns=dict(zip(key_subscales,key_plot_names)),inplace=True)
    return df


## TASK ANALYSIS FUNCTIONS ##
def binEstimatesOutcomes(estimates, outcomes):
    """Bin the estimates and outcomes by confidence, and supply counts.

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates
        outcomes (list or numpy.array): vector of trial by trial outcomes

    Returns:
        numpy.arrays: estimates_binned, outcomes_binned, counts
    """
    confidence_keys = np.arange(np.max(estimates)) 
    counts = [0] * len(confidence_keys)
    correct = [0] * len(confidence_keys)

    for i in range(len(estimates)):
        counts[estimates[i] - 1] += 1
        correct[estimates[i] - 1] += outcomes[i]

    estimates_binned_ = np.linspace(0.5, 1, len(confidence_keys) + 1)
    estimates_binned = [(estimates_binned_[i] + estimates_binned_[i+1]) / 2 for i in range(len(estimates_binned_) - 1)]

    outcomes_binned = [n / counts[i] if counts[i] != 0 else 0 for i, n in enumerate(correct)]

    return np.array(estimates_binned), np.array(outcomes_binned), np.array(counts)


def convertConfidenceKeysToProbs(estimates):
    """Creates confidence probabilities from confidence keys

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates

    Returns:
        numpy.array: estimates_binned
    """
    confidence_keys = np.arange(np.max(estimates)) 
    estimates_binned_ = np.linspace(0.5, 1, len(confidence_keys) + 1)
    estimates_binned = [(estimates_binned_[i] + estimates_binned_[i+1]) / 2 for i in range(len(estimates_binned_) - 1)]
    return np.array(estimates_binned)


def calcBrierScore(estimates,outcomes,by_trial=False):
    """Calculates Brier score (averaged across trials) or probability score (not averaged across trials)

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates
        outcomes (list or numpy.array): vector of trial by trial outcomes
        by_trial (bool, optional): whether to calculate by trial or averaged across. Defaults to False.

    Raises:
        ValueError: variable check for by_trial
        ValueError: equal length of estimate and outcome vectors

    Returns:
        float or vector of floats
    """
    try:
        assert(type(by_trial)==bool)
    except:
        raise ValueError('by_trial must be a boolean variable')
    if len(estimates) != len(outcomes):
        raise ValueError("The number of estimates and outcomes must be equal")
    if not by_trial:
        estimates_binned, outcomes_binned, N_j = binEstimatesOutcomes(estimates,outcomes)
        brier_score = np.mean((estimates_binned-outcomes_binned)**2)
    elif by_trial:
        print('Calculating probability score (by trial)')
        brier_score = (estimates-outcomes)**2
    return brier_score


def calcCalibration(estimates,outcomes):
    """Computes the calibration score (C) for a set of estimates and outcomes

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates
        outcomes (list or numpy.array): vector of trial by trial outcomes

    Returns:
        float: calibration score
    """
    estimates_binned, outcomes_binned, N_j = binEstimatesOutcomes(estimates,outcomes)
    C = np.sum(N_j*(estimates_binned-outcomes_binned)**2) / sum(N_j)
    return C


def convertCalibration(estimates, outcomes):
    """Calculates calibration and converts it to a 0-1 scale, where 0 is the worst possible calibration and 
       1 is the best possible calibration.

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates.
        outcomes (list or numpy.array): vector of trial by trial outcomes.

    Returns:
        float: converted calibration score
    """
    estimates_worst = estimates<2
    calibration_worst = calcCalibration(estimates, estimates_worst)
    calibration = calcCalibration(estimates, outcomes)

    calibration_converted_score = 1 - (calibration / calibration_worst)
    calibration_converted_score = max(min(calibration_converted_score, 1),0)
    
    return calibration_converted_score


def calcOutcomeVariance(outcomes):
    """Calculates the variance of the outcomes (O) in the Brier score decomposition. 

    Args:
        outcomes (list or numpy.array): vector of trial by trial outcomes

    Returns:
        float: outcome variance
    """
    O = np.mean(outcomes) * (1 - np.mean(outcomes))
    return O


def calcResolution(estimates,outcomes):
    """Calculates the resolution score (R) for a set of estimates and outcomes in the Brier score decomposition.

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates.
        outcomes (list or numpy.array): vector of trial by trial outcomes.

    Returns:
        float: resolution score
    """
    _, outcomes_binned, N_j = binEstimatesOutcomes(estimates,outcomes)
    R = np.sum(N_j * (outcomes_binned - np.mean(outcomes))**2) / sum(N_j)
    return R


def calcLogProb(estimates,outcomes,by_trial=False):
    """Calculate the log probability score of the estimates and outcomes

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates.
        outcomes (list or numpy.array): vector of trial by trial outcomes.
        by_trial (bool, optional): whether to return the mean log prob, or the log prob on each trial. Defaults to False.

    Raises:
        ValueError: check the variable by_trial

    Returns:
        float or numpy.array: log probability score
    """
    try:
        assert(type(by_trial)==bool)
    except:
        raise ValueError('by_trial must be a boolean variable')
    indx_occured = outcomes == 1
    if not by_trial:
        log_prob = np.mean(np.concatenate((-np.log(1 - estimates[indx_occured<1] + .0001),
                           -np.log(estimates[indx_occured] + .0001))))
    elif by_trial:
        log_prob = np.concatenate((-np.log(1 - estimates[indx_occured<1] + .0001),
                           -np.log(estimates[indx_occured] + .0001)))
    return log_prob


def multRespByTrialScore(estimates,outcomes,method='brier'):
    """Score the trial-by-trial confidence estimate using either the Brier score or the log probability score

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates.
        outcomes (list or numpy.array): vector of trial by trial outcomes.
        method (str, optional): 'brier' or 'log_prob'. Defaults to 'brier'.

    Raises:
        ValueError: checks to see if method is 'brier' or 'log_prob'.

    Returns:
        numpy.array: vector of scores
    """
    if method == 'brier':
        func = calcBrierScore
    elif method == 'log_prob':
        func = calcLogProb
    else:
        raise ValueError(f'{method} is invalid. Must be "brier" or "log_prob"')
    scores = np.zeros(estimates.shape[0])
    for t in range(estimates.shape[0]):
        scores[t] = func(estimates[t,:],outcomes[t,:])
    return scores


## QUESTIONNAIRE ANALYSIS FUNCTIONS ##


def createDemographicsDF(f_dir_demographics = 'prolific_demographics/'):
    """Loads the demographics csv provided by Prolific and turns it into a dataframe.

    Args:
        f_dir_demographics (str, optional): Location of Prolifics demographics file. Defaults to 'prolific_demographics/'.

    Returns:
        pandas.DataFrame: demographics dataframe
    """
    f_names = glob(os.path.join(f_dir_demographics,'*.csv'))
    demographics_list = []
    for f_name in f_names:
        demographics_list.append(pd.read_csv(f_name))
    demographics_df = pd.concat(demographics_list,axis=0)
    demographics_df.rename(columns={'Participant id':'external_id'},inplace=True)
    return demographics_df


def makeSubjectSubstancesDF(data,substance_options=None):
    """Takes the user's reported substances in the database and turns them into a dataframe

    Args:
        data (pandas.DataFrame): output from the database with the substances column as a list
        substance_options (list, optional): the possible substances. Defaults to None.

    Returns:
        pandas.DataFrame: dataframe where substances are specified in columns with boolean values. 
    """
    if substance_options is None:
        substance_options = ['caffeine','adhd_stimulants','alcohol','tobacco','marijuana','opioids',
        'illicit_stimulants']
    # Initialize the dictionary
    substance_check_dict = {'subject_id':[]}
    for substance in substance_options:
        substance_check_dict[substance] = []
    # Go through individual subjects
    for subject_id in data.subject_id.unique():
        subject_substances = data.substances[data.subject_id==subject_id].iloc[0]
        substance_check_dict['subject_id'].append(subject_id)
        for substance in substance_options:
            if data.substances[data.subject_id==subject_id].iloc[0] is None:
                substance_check_dict[substance].append(False)
                continue
            if substance in subject_substances:
                substance_check_dict[substance].append(True)
            else:
                substance_check_dict[substance].append(False)
    # Return DF
    substance_check_df = pd.DataFrame(substance_check_dict)
    return substance_check_df


def calcConnersInconsistency(data_overall):
    """Calculate the inconsistency score from Conners responses. Above 8 is considered inconsistent.

    Args:
        data_overall (pandas.DataFrame): dataframe of question-answer responses

    Returns:
        float: inconsistency score. 
    """
    conners_inconsistency_scores = [
        np.abs(data_overall.loc[data_overall['questionnaire_question_number']==11,'answer'].iloc[0] - 
            data_overall.loc[data_overall['questionnaire_question_number']==49,'answer'].iloc[0]),
        np.abs(data_overall.loc[data_overall['questionnaire_question_number']==40,'answer'].iloc[0] - 
            data_overall.loc[data_overall['questionnaire_question_number']==44,'answer'].iloc[0]),
        np.abs(data_overall.loc[data_overall['questionnaire_question_number']==20,'answer'].iloc[0] - 
            data_overall.loc[data_overall['questionnaire_question_number']==25,'answer'].iloc[0]),
        np.abs(data_overall.loc[data_overall['questionnaire_question_number']==30,'answer'].iloc[0] - 
            data_overall.loc[data_overall['questionnaire_question_number']==47,'answer'].iloc[0]),
        np.abs(data_overall.loc[data_overall['questionnaire_question_number']==19,'answer'].iloc[0] - 
            data_overall.loc[data_overall['questionnaire_question_number']==23,'answer'].iloc[0]),
        np.abs(data_overall.loc[data_overall['questionnaire_question_number']==6,'answer'].iloc[0] - 
            data_overall.loc[data_overall['questionnaire_question_number']==37,'answer'].iloc[0]),
        np.abs(data_overall.loc[data_overall['questionnaire_question_number']==26,'answer'].iloc[0] - 
            data_overall.loc[data_overall['questionnaire_question_number']==63,'answer'].iloc[0]),
    ]
    conners_inconsistency_score = np.sum(conners_inconsistency_scores)
    
    return conners_inconsistency_score


def appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex):
    """Helper function to append Conners T values to a dictionary.

    Args:
        T_conversion (dict): dictionary of responses
        subscale (vec of str): the specific subscale
        min_age (vec of int): minimum age for that t conversion
        max_age (vec of int): _description_
        raw (vec of int): the raw score
        T (vec of int): T corrected score
        sex (vec of str): subject sex

    Returns:
        dict: dictionary with new values appended
    """
    T_conversion['subscale'].extend(subscale)
    T_conversion['min_age'].extend(min_age)
    T_conversion['max_age'].extend(max_age)
    T_conversion['raw'].extend(raw)
    T_conversion['T'].extend(T)
    T_conversion['sex'].extend(sex)
    return T_conversion


def getConnersTValues():
    """Builds the conners conversion table

    Returns:
        dict:
    """
    T_conversion = {
        "subscale": [],
        "min_age": [],
        "max_age": [],
        "raw": [],
        'T': [],
        'sex': []
    }

    ## Male
    sex_label = 'male'

    # Inattention/Memory Problems
    subscale_name = 'inattention_memory_problems'
    # 18-29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [31,33,34,35,37,38,40,41,43,44,46,47,48,50,51,53,54,56,57,59,60,62,63,65,66,68,69,71,72,74,75,77,78,80,81,83,84]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    T = [33,35,36,37,39,41,42,44,46,47,49,50,52,54,55,57,59,60,62,64,65,67,69,70,72,74,75,77,79,80,82,84,85,87,89,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [32,33,34,36,38,39,41,43,44,46,47,49,51,52,54,56,57,59,60,62,64,65,67,69,70,72,74,75,77,78,80,82,83,85,87,88,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    T = [35,36,37,39,40,42,44,45,47,49,50,52,53,55,57,58,60,62,63,65,66,68,70,71,73,75,76,78,79,81,83,84,86,87,89]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # Hyperactivity/Restlessness
    subscale_name = 'hyperactivity_restlessness'
    # 18 - 29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [28,29,30,31,33,34,35,37,38,40,41,42,44,45,46,48,49,51,52,53,55,56,57,59,60,61,63,64,66,67,68,70,71,72,74,75,76]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [32,33,34,36,37,38,40,41,43,44,46,47,48,50,51,53,54,56,57,59,60,61,63,64,66,67,69,70,71,73,74,76,77,79,80,81,83]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [35,36,37,38,40,41,42,44,45,47,48,49,51,52,53,55,56,57,59,60,62,63,64,66,67,68,70,71,72,74,75,77,78,79,81,82,83]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [35,36,37,38,40,41,42,44,45,47,48,49,51,52,53,55,56,57,59,60,62,63,64,66,67,68,70,71,72,74,75,77,78,79,81,82,83]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # Impulsivity/Emotional Lability
    subscale_name = 'impulsivity_emotional_lability'
    # 18 - 29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [31,33,34,35,37,38,40,41,43,44,47,48,49,50,52,53,55,56,58,59,61,62,64,65,67,68,70,71,73,74,76,77,79,80,82,83,85]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [35,36,37,39,40,42,43,45,46,48,49,51,52,53,55,56,58,59,61,62,64,65,57,58,70,71,73,74,76,77,78,80,81,83,84,86,87]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    T = [35,36,37,36,38,39,41,43,45,47,49,51,53,54,56,58,60,62,64,66,68,70,71,73,75,77,79,81,83,85,87,88,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    T = [35,37,38,39,41,42,44,46,47,49,50,52,54,55,57,58,60,62,63,65,66,68,69,71,73,74,76,77,79,81,82,84,85,87,89,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # Problems with Self-Concept
    subscale_name = 'problems_with_self_concept'
    # 18 - 29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    T = [36,38,39,42,44,46,49,51,53,56,58,60,63,65,68,70,72,75,77]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    T = [38,40,42,44,46,49,51,53,56,58,61,63,65,68,70,72,75,77,79]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    T = [36,39,40,42,45,47,50,52,55,57,59,62,64,67,69,72,74,77,79]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    T = [37,40,41,44,47,50,53,56,60,63,66,69,72,75,78,81,84,87,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # DSM-IV Inattentive Symptoms
    subscale_name = 'dsm_iv_inattentive_symptoms'
    # 18 - 29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    T = [36,39,40,43,46,48,51,53,56,59,61,64,66,69,72,74,77,79,82,85,87,90]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    T = [36,39,40,43,46,48,51,53,56,59,61,64,66,69,72,74,77,79,82,85,87,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    T = [28,31,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74,77,80,85,87,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    T = [28,31,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74,77,80,84,87,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # DSM-IV Hyperactive-Impulsive Symptoms
    subscale_name = 'dsm_iv_hyperactive_impulsive_symptoms'
    # 18 - 29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    T = [33,35,36,39,41,44,46,49,51,54,56,59,61,64,66,69,71,74,76,79,81,84,86,89,90]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    T = [33,35,36,39,41,44,46,49,51,54,56,59,61,64,66,69,71,74,76,79,81,84,86,89,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    T = [32,34,36,38,40,43,45,48,50,52,55,57,59,62,64,67,69,71,74,76,78,81,83,86,88,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    T = [32,34,36,38,40,43,45,48,50,52,55,57,59,62,64,67,69,71,74,76,78,81,83,86,88,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # DSM-IV ADHD Symptoms Total
    subscale_name = 'dsm_iv_adhd_symptoms_total'
    # 18 - 29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
    T = [31,32,33,35,37,38,40,41,43,44,46,47,49,51,52,54,55,57,58,60,61,63,64,66,68,69,71,72,74,75,77,78,80,82,83,85,86,88,89,90]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
    T = [31,32,33,35,37,38,40,41,43,44,46,47,49,51,52,54,55,57,58,60,61,63,64,66,68,69,71,72,74,75,77,78,80,82,83,85,86,88,89,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
    T = [28,29,30,31,33,34,36,37,39,40,42,43,45,46,48,49,51,52,54,55,57,58,60,61,63,64,66,67,69,70,72,73,75,76,78,79,81,82,84,85,87,89,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
    T = [28,29,30,31,33,34,36,37,39,40,42,43,45,46,48,49,51,52,54,55,57,58,60,61,63,64,66,67,69,70,72,73,75,76,78,79,81,82,84,85,87,89,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # ADHD Index
    subscale_name = 'adhd_index'
    # 18-29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [30,31,32,34,36,37,39,40,42,44,45,47,48,50,52,53,55,57,58,60,61,63,65,66,68,69,71,73,74,76,77,79,84,82,84,85,87]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    T = [33,35,36,38,39,41,43,44,46,48,49,51,53,54,56,58,59,61,63,64,66,68,70,71,73,75,76,78,80,81,83,85,86,88,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    T = [34,35,36,38,40,41,43,44,46,48,49,51,53,54,56,57,59,61,62,64,66,67,69,70,72,74,75,77,79,80,82,83,85,87,88,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    T = [34,36,37,39,41,42,44,46,48,49,51,53,55,57,58,60,62,64,65,67,69,71,73,74,76,78,80,81,83,85,87,89,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    ## FEMALE
    sex_label = 'female'
    # Inattention/Memory Problems
    subscale_name = 'inattention_memory_problems'
    # 18-29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [35,36,37,38,40,42,43,45,46,48,49,51,52,54,56,57,59,60,62,63,65,66,68,70,71,73,74,76,77,79,80,82,84,85,87,88,90]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    T = [34,36,37,39,40,42,44,45,47,49,50,52,54,55,57,59,60,62,63,65,67,68,70,72,73,75,77,76,80,82,83,85,87,88,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    T = [35,36,37,39,41,42,44,45,48,49,51,53,54,56,58,60,61,63,65,66,68,70,72,73,75,77,78,80,82,84,85,87,89,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [36,34,38,39,41,43,44,46,47,49,50,52,53,55,57,58,60,61,63,64,66,67,69,70,72,74,75,77,78,80,81,83,84,86,88,89,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # Hyperactivity/Restlessness
    subscale_name = 'hyperactivity_restlessness'
    # 18-29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [32,33,34,35,37,38,39,41,42,44,45,46,48,49,51,52,53,55,56,57,59,60,62,63,64,66,67,69,70,71,73,74,76,77,78,80,81]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [33,35,36,37,38,40,41,43,44,46,47,48,50,51,53,54,56,57,59,60,61,63,64,66,67,69,70,71,73,74,76,77,79,80,81,83,84]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [35,36,37,38,40,41,43,44,45,47,48,50,51,53,54,55,57,58,60,61,62,64,65,67,68,69,71,72,74,75,77,78,79,81,82,84,85]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [35,36,37,38,40,41,43,44,45,47,48,50,51,53,54,55,57,58,60,61,62,64,65,67,68,69,71,72,74,75,77,78,79,81,82,84,85]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # Impulsivity Emotional Lability
    subscale_name = 'impulsivity_emotional_lability'
    # 18-29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    T = [33,35,36,37,39,40,42,44,45,47,48,50,52,53,55,56,58,60,61,63,64,66,68,69,71,72,74,76,77,79,80,82,84,85,87,88,90]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    T = [30,32,33,35,36,38,40,40,44,46,47,49,51,53,55,57,59,60,62,64,66,68,70,71,73,75,77,79,81,83,84,86,88,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    T = [33,34,35,37,39,41,43,44,46,48,50,52,53,55,57,59,61,63,64,66,68,70,72,73,75,77,79,81,83,84,83,88,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    T = [33,36,35,37,39,41,43,45,46,48,50,52,54,56,57,59,61,63,65,67,69,70,72,74,76,78,80,81,83,85,87,89,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # Problems with Self-Concept
    subscale_name = 'problems_with_self_concept'
    # 18 - 29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    T = [34,36,37,40,42,45,47,49,52,54,57,59,62,64,66,69,71,74,76]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    T = [33,35,37,39,42,44,47,50,52,55,58,60,63,66,68,71,73,76,79]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    T = [35,38,39,41,43,46,48,50,53,55,57,59,62,64,66,69,71,73,76]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    T = [36,38,39,42,44,47,49,52,54,57,59,61,64,66,69,71,74,76,79]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # DSM-IV Inattentive Symptoms
    subscale_name = 'dsm_iv_inattentive_symptoms'
    # 18 - 29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    T = [35,37,38,41,43,45,47,49,52,54,56,58,60,63,65,67,69,72,74,76,78,80,83,85,87,89,90]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    T = [35,37,38,41,43,45,47,49,52,54,56,58,60,63,65,67,69,72,74,76,78,80,83,85,87,89,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    T = [29,33,34,38,41,44,47,51,54,57,60,64,67,70,73,77,80,83,86,89,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    T = [29,33,34,38,41,44,47,51,54,57,60,64,67,70,73,77,80,83,86,89,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # DSM-IV Hyperactive-Impulsive Symptoms
    subscale_name = 'dsm_iv_hyperactive_impulsive_symptoms'
    # 18 - 29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
    T = [31,33,34,36,39,41,43,45,48,50,52,55,57,59,61,64,66,68,70,73,75,77,79,82,84,86,88,90]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
    T = [31,33,34,36,39,41,43,45,48,50,52,55,57,59,61,64,66,68,70,73,75,77,79,82,84,86,88,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25]
    T = [29,32,33,36,38,41,43,46,48,51,54,56,59,61,64,66,69,72,74,77,79,82,84,87,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25]
    T = [29,32,33,36,38,41,43,46,48,51,54,56,59,61,64,66,69,72,74,77,79,82,84,87,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # DSM-IV ADHD Symptoms Total
    subscale_name = 'dsm_iv_adhd_symptoms_total'
    # 18 - 29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]
    T = [30,31,32,34,35,36,37,39,40,41,43,44,45,46,48,49,50,51,53,54,55,57,58,59,60,62,63,64,65,67,68,69,71,72,73,74,76,77,78,79,81,82,83,85,86,87,88,90]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]
    T = [30,31,32,34,35,36,37,39,40,41,43,44,45,46,48,49,50,51,53,54,55,57,58,59,60,62,63,64,65,67,68,69,71,72,73,74,76,77,78,79,81,82,83,85,86,87,88,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
    T = [26,27,28,30,32,33,35,37,38,40,42,43,45,47,48,50,52,53,55,57,58,60,62,63,65,67,68,70,72,74,75,77,79,80,82,84,85,87,89,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
    T = [26,27,28,30,32,33,35,37,38,40,42,43,45,47,48,50,52,53,55,57,58,60,62,63,65,67,68,70,72,74,75,77,79,80,82,84,85,87,89,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    # ADHD Index
    subscale_name = 'adhd_index'
    # 18-29
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    T = [31,34,35,36,38,40,42,43,45,47,48,50,52,53,55,57,58,60,62,63,65,67,68,70,72,73,75,77,78,80,82,83,85,87,89,90]
    min_age = [18] * len(raw)
    max_age = [29] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 30-39
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    T = [31,32,33,35,37,39,41,43,44,46,48,50,52,54,55,57,59,61,63,65,66,68,70,72,74,75,77,79,81,83,85,86,88,90]
    min_age = [30] * len(raw)
    max_age = [39] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 40-49
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    T = [33,35,36,38,39,41,43,45,47,49,50,52,54,56,58,59,61,63,65,67,68,70,72,74,76,77,79,81,83,85,87,88,90]
    min_age = [40] * len(raw)
    max_age = [49] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)
    # 50+
    raw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    T = [33,35,36,38,40,41,43,45,47,49,51,52,54,56,58,60,62,63,65,67,69,71,73,75,76,78,80,82,84,86,87,89,90]
    min_age = [50] * len(raw)
    max_age = [np.inf] * len(raw)
    subscale = [subscale_name] * len(raw)
    sex = [sex_label] * len(raw)
    T_conversion = appendConnersTValues(T_conversion,subscale,min_age,max_age,raw,T,sex)

    T_conversion_df = pd.DataFrame(T_conversion)

    return T_conversion_df


def convertConnersT(subscale,sex,raw_score,age,T_conversion_df=None):
    """Converts a subjects' conners score to a T score

    Args:
        subscale (str): relevant subscale
        sex (str): sex of subject
        raw_score (int): subscale score
        age (int): _description_
        T_conversion_df (pandas.DataFrame, optional): conversion dataframe. Defaults to None. If none, it will be created

    Returns:
        int: converted T score
    """
    if T_conversion_df is None:
        T_conversion_df = getConnersTValues()
    T_conversion_df_sub = T_conversion_df.loc[(T_conversion_df.subscale == subscale) & \
        (T_conversion_df.sex == sex) & (T_conversion_df.min_age <= age) & (T_conversion_df.max_age >= age),:]
    # Because some of the T scores are for all above a raw score, we find the closest
    T = T_conversion_df_sub.loc[T_conversion_df_sub.raw==min(T_conversion_df_sub['raw'], key=lambda x:abs(x- raw_score)),'T'].iloc[0]
    return T


def getQuestionnaireSummaries(data_questionnaires,raw_scores=False,convert_education=True,convert_age=True,\
    screen_conners_inconsistency=False,verbose=False):
    """Converts raw subject questionnaire data into summary scores

    Args:
        data_questionnaires (pandas.DataFrame): contains individual responses from subjects
        raw_scores (bool, optional): whether to perform Conners conversion and others. Defaults to False.
        convert_education (bool, optional): Whether to convert string education to numerics. Defaults to True.
        convert_age (bool, optional): Whether to convert string age reports to ints (legacy). Defaults to True.
        verbose (bool, optional): Tell the user what is happening. Defaults to False.

    Returns:
        pandas.DataFrames: overall_scores_df, subscale_scores_df
    """
    #Copy before modifying
    data_questionnaires = copy.deepcopy(data_questionnaires)
    #Clean up the DF
    data_questionnaires.loc[data_questionnaires['questionnaire_name'] == 'phq9','subscale'] = 'NA'
    data_questionnaires = data_questionnaires.loc[data_questionnaires['subscale'] != 'Attention Check',:]
    data_questionnaires['questionnaire_name'] = data_questionnaires['questionnaire_name'].str.lower()
    data_questionnaires['subscale'] = data_questionnaires['subscale'].str.lower()
    data_questionnaires = data_questionnaires.loc[data_questionnaires.sex != '',:]
    data_questionnaires.loc[data_questionnaires.subscale=='hyperactive-impulsive symptoms','subscale'] = 'dsm-iv hyperactive-impulsive symptoms'
    if 'possible_answers' in data_questionnaires.columns:
        data_questionnaires = convertUnhashableDFcolumn(data_questionnaires,'possible_answers')
    if 'substances' in data_questionnaires.columns:
        data_questionnaires = convertUnhashableDFcolumn(data_questionnaires,'substances')
    if 'psych_history' in data_questionnaires.columns:
        data_questionnaires = convertUnhashableDFcolumn(data_questionnaires,'psych_history')
    data_questionnaires.drop_duplicates(inplace=True)
    #For the Conners, split the comma questions into two
    print('Splitting Conners questions')
    data_questionnaires_sub = data_questionnaires.assign(subscale=data_questionnaires['subscale'].str.split(',')).explode('subscale')
    data_questionnaires_sub = data_questionnaires_sub.reset_index(drop=True)
    print('Processing substances')
    substance_df = makeSubstancesDF(data_questionnaires,substances_list=['caffeine','adhd_stimulants','alcohol','tobacco',\
        'marijuana','opioids','illicit_stimulants'])
    print('Processing psych history')
    psych_history_df = makePsychHistoryDF(data_questionnaires,psych_history_list=['asd','adhd','ocd','depression',\
        'bipolar','schizophrenia','schizotypy','addiction'])
    
    #Prep dictionanries
    overall_scores = {
        'subject_id': [],
        'external_id': [],
        'questionnaire_name': [],
        'score': [],
        'education': [],
        'age': [],
        'sex': [],
        'gender': [],
        'passed_attention_check': [],
        'session_completed': []
    }

    subscale_scores = {
        'subject_id': [],
        'external_id': [],
        'questionnaire_name': [],
        'subscale': [],
        'score': [],
        'education': [],
        'age': [],
        'sex': [],
        'gender': [],
        'passed_attention_check': [],
        'session_completed': []
    }


    education_conversion = {
        'postgrad': 4,
        'college': 3, 
        '<college': 2, 
        'highschool': 1, 
        '<highschool': 0
    }

    age_conversion = {
        '10-20': 0,
        '21-30': 1,
        '31-45': 2,
        "46-65": 3,
        '>65': 4
    }

    if convert_education:
        for level in education_conversion.keys():
            data_questionnaires.loc[data_questionnaires['education'] == level,'education'] = education_conversion[level]
            data_questionnaires_sub.loc[data_questionnaires_sub['education'] == level,'education'] = education_conversion[level]

    if convert_age:
        for age in age_conversion.keys():
            data_questionnaires.loc[data_questionnaires['age'] == age,'age'] = age_conversion[age]
            data_questionnaires_sub.loc[data_questionnaires_sub['age'] == age,'age'] = age_conversion[age]
        data_questionnaires['age'] = data_questionnaires['age'].astype(int)
        data_questionnaires_sub['age'] = data_questionnaires_sub['age'].astype(int)
        
    # change the ASRS to the official scoring rubric
    if ('asrs_screener' in data_questionnaires['questionnaire_name'].unique()) and (raw_scores is False):
        data_questionnaires.loc[data_questionnaires['questionnaire_name']=='asrs_screener',['answer']] = \
            asrsScreenerAnswerConversion(data_questionnaires)
    # If conners, get ready to do some converting
    if 'conners_full' in data_questionnaires['questionnaire_name'].unique():
        T_conversion_df = getConnersTValues()
    # Loop through and calculate score for each subject
    for i, subject_id in enumerate(data_questionnaires['subject_id'].unique()):
        if verbose:
            print(f'Processing subject {i}/{data_questionnaires.subject_id.nunique()}')
        for questionnaire_name in data_questionnaires['questionnaire_name'].unique():
            if questionnaire_name in ['att_check','att_check_list']:
                continue
            # save overall scores
            data_overall = data_questionnaires.loc[(data_questionnaires['subject_id'] == subject_id) & 
                                                (data_questionnaires['questionnaire_name'] == questionnaire_name)]
            overall_scores['subject_id'].append(subject_id)
            try:
                overall_scores['external_id'].append(data_overall['external_id'].iloc[0])
            except:
                print('here')
            overall_scores['questionnaire_name'].append(questionnaire_name)
            if (raw_scores == False) and ((questionnaire_name == 'bapq') or (questionnaire_name == 'rbq2a')):
                overall_scores['score'].append(np.mean(data_overall['answer']))
            elif (questionnaire_name == 'cape_pos_neg') or (questionnaire_name == 'cape'):
                weighted_score, _, subscales_cape, _ = scoreSubjectCAPE(data_overall)
                overall_scores['score'].append(np.mean(weighted_score))
            else:
                overall_scores['score'].append(np.sum(data_overall['answer']))
            overall_scores['education'].append(data_overall['education'].iloc[0])
            overall_scores['age'].append(data_overall['age'].iloc[0])
            overall_scores['sex'].append(data_overall['sex'].iloc[0])
            overall_scores['gender'].append(data_overall['gender'].iloc[0])
            overall_scores['passed_attention_check'].append(data_overall['passed_attention_check'].iloc[0])
            overall_scores['session_completed'].append(data_overall['session_completed'].iloc[0])
            # Check for inconsistency in the Conners score
            if questionnaire_name == 'conners_full':
                conners_asd_inconsistency_score = calcConnersInconsistency(data_overall)
                subscale_scores['score'].append(conners_asd_inconsistency_score)
                subscale_scores['subject_id'].append(subject_id)
                subscale_scores['external_id'].append(data_overall['external_id'].iloc[0])
                subscale_scores['questionnaire_name'].append(questionnaire_name)
                subscale_scores['education'].append(data_overall['education'].iloc[0])
                subscale_scores['age'].append(data_overall['age'].iloc[0])
                subscale_scores['sex'].append(data_overall['sex'].iloc[0])
                subscale_scores['gender'].append(data_overall['gender'].iloc[0])
                subscale_scores['passed_attention_check'].append(data_overall['passed_attention_check'].iloc[0])
                subscale_scores['session_completed'].append(data_overall['session_completed'].iloc[0])
                subscale_scores['subscale'].append(f'{questionnaire_name}_inconsistency')
            # Save subscale scores
            subscales = data_questionnaires_sub.loc[(data_questionnaires_sub['subject_id'] == subject_id) & \
                (data_questionnaires_sub['questionnaire_name'] == questionnaire_name),['subscale']].drop_duplicates().to_numpy().flatten()
            for subscale in subscales:
                subscale_list = subscale.split(',')
                for subscale_ in subscale_list:
                    data_subscale = data_questionnaires_sub.loc[(data_questionnaires_sub['subject_id'] == subject_id) & 
                                                    (data_questionnaires_sub['questionnaire_name'] == questionnaire_name) & 
                                                    (data_questionnaires_sub['subscale'] == subscale)]
                    subscale_scores['subject_id'].append(subject_id)
                    subscale_scores['external_id'].append(data_subscale['external_id'].iloc[0])
                    subscale_scores['questionnaire_name'].append(questionnaire_name)
                    subscale_scores['education'].append(data_subscale['education'].iloc[0])
                    subscale_scores['age'].append(data_subscale['age'].iloc[0])
                    subscale_scores['sex'].append(data_subscale['sex'].iloc[0])
                    subscale_scores['gender'].append(data_subscale['gender'].iloc[0])
                    subscale_scores['passed_attention_check'].append(data_subscale['passed_attention_check'].iloc[0])
                    subscale_scores['session_completed'].append(data_subscale['session_completed'].iloc[0])
                    subscale_scores['subscale'].append(f'{questionnaire_name}_{subscale_.replace(" ","_").replace("-","_")}'.lower())
                    if (raw_scores == False) and ((questionnaire_name == 'bapq') or (questionnaire_name == 'rbq2a')):
                        subscale_scores['score'].append(np.mean(data_subscale['answer']))
                    elif (questionnaire_name == 'cape_pos_neg') or (questionnaire_name == 'cape'):
                        subscale_scores['score'].append(np.mean(weighted_score[[subscale_ == sub_cape for sub_cape in subscales_cape]]))
                    elif (questionnaire_name == 'conners_full') and (subscale_scores['sex'][-1] in ['male','female']):
                        subscale_scores['score'].append(convertConnersT(subscale_.replace(" ","_").replace("-","_").lower(),subscale_scores['sex'][-1],\
                                np.sum(data_subscale['answer']),age=subscale_scores['age'][-1]))
                    else:
                        subscale_scores['score'].append(np.sum(data_subscale['answer']))

    if verbose:
        print('All subjects processed. Now formatting DF')
    # Merge the score dataframes
    overall_scores_df = pd.DataFrame(overall_scores).drop_duplicates()
    subscale_scores_df = pd.DataFrame(subscale_scores).drop_duplicates()

    if screen_conners_inconsistency:
        subscale_df_conners_inconsistency = subscale_scores_df.loc[subscale_scores_df.subscale == \
            'conners_full_inconsistency',:]
        subj_id_conners_screen = subscale_df_conners_inconsistency.loc[subscale_df_conners_inconsistency.score<8,\
            'subject_id'].to_numpy()
        overall_scores_df = overall_scores_df.loc[overall_scores_df.subject_id.isin(subj_id_conners_screen), :]
        subscale_scores_df = subscale_scores_df.loc[subscale_scores_df.subject_id.isin(subj_id_conners_screen), :]
        data_questionnaires = data_questionnaires.loc[data_questionnaires.subject_id.isin(subj_id_conners_screen), :]
        
    overall_scores_df = ft.reduce(lambda left, right: pd.merge(left, right, on='subject_id'),\
        [overall_scores_df,substance_df,psych_history_df])
    subscale_scores_df = ft.reduce(lambda left, right: pd.merge(left, right, on='subject_id'),\
        [subscale_scores_df,substance_df,psych_history_df])

    return overall_scores_df, subscale_scores_df


def includeQuestionnaireCutoffs(overall_scores_df,subscale_scores_df):
    """Adds the standard questionniare cutoffs to the subscale and overall scores dataframes

    Args:
        overall_scores_df (pandas.DataFrame): subscale scores for each subject
        subscale_scores_df (pandas.DataFrame): overall scores for each subject

    Returns:
        pandas.DataFrames: overall_scores_df, subscale_scores_df
    """
    bapq_cutoff_df = passBAPQcutoff(overall_scores_df,male_thresh=3.55,female_thresh=3.17)
    cape_cutoff_df = passCAPECutoff(subscale_scores_df)
    conners_adhd_cutoff_df = passConnersADHDindexCutoff(subscale_scores_df,thresh=60)
    phq9_cutoff_df = passPHQ9Cutoffs(overall_scores_df)
    overall_scores_df = ft.reduce(lambda left, right: pd.merge(left, right, on='subject_id'),\
        [overall_scores_df,bapq_cutoff_df,conners_adhd_cutoff_df,phq9_cutoff_df,cape_cutoff_df])
    subscale_scores_df = ft.reduce(lambda left, right: pd.merge(left, right, on='subject_id'),\
        [subscale_scores_df,bapq_cutoff_df,conners_adhd_cutoff_df,phq9_cutoff_df,cape_cutoff_df])
    return overall_scores_df, subscale_scores_df


def convertUnhashableDFcolumn(data_questionnaires,column):
    """Converts an unhashable column to something parsable by pandas. Often needed when building a DF from server data

    Args:
        data_questionnaires (pandas.DataFrame): the dataframe to convert
        column (str): specific column to be converted

    Returns:
        pandas.DataFrame: data_questionnaires
    """
    if column not in data_questionnaires.columns:
        print(f'Column {column} not in data_questionnaires, so no unhashable conversion needed')
        return data_questionnaires
    if type(data_questionnaires[column].iloc[0]) is tuple:
        print(f'Column {column} already converted to unhashable')
        return data_questionnaires
    # Convert to tuples (avoid unhashable type error)
    vec = []
    for i in range(len(data_questionnaires[column])):
        if (data_questionnaires[column].iloc[i] == None) or (data_questionnaires[column].iloc[i] is np.nan):
            vec.append(())
        elif type(data_questionnaires[column].iloc[i]) is str:
            vec.append(tuple(eval(data_questionnaires[column].iloc[i])))
        else:
            vec.append(tuple(data_questionnaires[column].iloc[i]))
    data_questionnaires[column] = vec

    return data_questionnaires


def passBAPQcutoff(input_df,male_thresh=3.55,female_thresh=3.17):
    """Determine which participants passed the BAPQ threshold. It is sex-specific. 

    Args:
        input_df (pandas.DataFrame): questionnaire responses
        male_thresh (float, optional): cutoff for male subjects. Defaults to 3.55.
        female_thresh (float, optional): cutoff for female subjects. Defaults to 3.17.

    Returns:
        pandas.DataFrame: DF with boolean values reporting whether a subject passed.
    """
    if 'external_session_ID' in input_df:
        # Get the relevant subset of the data
        data_questionnaires_bapq = input_df.loc[(input_df['questionnaire_name']=='bapq') & \
            input_df['passed_attention_check'],['passed_attention_check', 'session_completed', 'external_id',
            'questionnaire_name', 'subscale','possible_answers', 'question','answer', 'questionnaire_question_number','gender']]
        # Create a dictionary for the return values
        assessment_dict = {
            'overall_score': [],
            'external_id': [],
            'subject_id': [],
            'bapq_cutoff': [],
            'group': []
        }
        # Loop through the participants
        for external_id in data_questionnaires_bapq['external_id'].unique():
            assessment_dict['external_id'].append(external_id)
            overall_score = np.mean(data_questionnaires_bapq.loc[data_questionnaires_bapq['external_id'] == external_id,'answer'])
            if 'sex' in data_questionnaires_bapq.columns:
                group = data_questionnaires_bapq.loc[data_questionnaires_bapq['external_id'] == external_id,'sex'].iloc[0]
            else:
                group = data_questionnaires_bapq.loc[data_questionnaires_bapq['external_id'] == external_id,'gender'].iloc[0]
            subject_id = data_questionnaires_bapq.loc[data_questionnaires_bapq['external_id'] == external_id,'subject_id'].iloc[0]
            if group == 'male':
                pass_thresh = overall_score >= male_thresh
            elif group == 'female':
                pass_thresh = overall_score >= female_thresh
            assessment_dict['bapq_cutoff'].append(pass_thresh)
            assessment_dict['overall_score'].append(overall_score)
            assessment_dict['group'].append(group)
            assessment_dict['subject_id'].append(subject_id)
        # Convert to a dataframe
        bapq_cutoff_df = pd.DataFrame(assessment_dict)

    else:
        if 'sex' in input_df.columns:
            bapq_cutoff_df = input_df.loc[input_df.questionnaire_name=='bapq',['score','subject_id','sex']]
        else:
            bapq_cutoff_df = input_df.loc[input_df.questionnaire_name=='bapq',['score','subject_id','gender']]
        bapq_cutoff_df['bapq_cutoff'] = np.zeros(len(bapq_cutoff_df)).astype(bool)
        if 'sex' in bapq_cutoff_df.columns:
            bapq_cutoff_df.loc[(bapq_cutoff_df['score']>=3.55) & (bapq_cutoff_df['sex']=='male'),'bapq_cutoff'] = True
            bapq_cutoff_df.loc[(bapq_cutoff_df['score']>=3.17) & (bapq_cutoff_df['sex']=='female'),'bapq_cutoff'] = True
        else:
            bapq_cutoff_df.loc[(bapq_cutoff_df['score']>=3.55) & (bapq_cutoff_df['gender']=='male'),'bapq_cutoff'] = True
            bapq_cutoff_df.loc[(bapq_cutoff_df['score']>=3.17) & (bapq_cutoff_df['gender']=='female'),'bapq_cutoff'] = True
        if 'sex' in bapq_cutoff_df.columns:
            bapq_cutoff_df = bapq_cutoff_df.drop(columns=['score','sex'])
        else:
            bapq_cutoff_df = bapq_cutoff_df.drop(columns=['score','gender'])
    # return the result
    return bapq_cutoff_df


def passConnersADHDindexCutoff(subscale_scores_df,thresh=60):
    """Determins if subjects are above the Conners ADHD index cutoff. This indicates they have ADHD

    Args:
        subscale_scores_df (pandas.DataFrame): the subscale scores for each subject
        thresh (int, optional): threshold for being above. Defaults to 60.

    Returns:
        pandas.DataFrame: dataframe with boolean values for each subjects indicating if they passes or not. 
    """
    conners_index_scores = subscale_scores_df.loc[subscale_scores_df.subscale=='conners_full_adhd_index',['score','subject_id']]
    conners_index_scores['conners_adhd_cutoff'] = False
    conners_index_scores.loc[conners_index_scores.score>=thresh,'conners_adhd_cutoff'] = True
    conners_index_scores_df = conners_index_scores.loc[:,['subject_id','conners_adhd_cutoff']]
    return conners_index_scores_df


def passPHQ9Cutoffs(overall_scores_df):
    """Depression cutoffs from the PHQ-9

    Args:
        overall_scores_df (pandas.DataFrame): dataframe with overall scores for each subject

    Returns:
        pandas.DataFrame: dataframe with booleans for whether the subject was above the cutoff
    """
    phq9_df = overall_scores_df.loc[overall_scores_df.questionnaire_name=='phq9',['score','subject_id']]
    phq9_df['phq9_mild_cutoff'] = np.zeros(len(phq9_df)).astype(bool)
    phq9_df['phq9_moderate_cutoff'] = np.zeros(len(phq9_df)).astype(bool)
    phq9_df['phq9_severe_cutoff'] = np.zeros(len(phq9_df)).astype(bool)
    phq9_df['phq9_very_severe_cutoff'] = np.zeros(len(phq9_df)).astype(bool)
    phq9_df.loc[(phq9_df['score']>=5) & (phq9_df['score']<10),'phq9_mild_cutoff'] = True
    phq9_df.loc[(phq9_df['score']>=10) & (phq9_df['score']<15),'phq9_moderate_cutoff'] = True
    phq9_df.loc[(phq9_df['score']>=15) & (phq9_df['score']<20),'phq9_severe_cutoff'] = True
    phq9_df.loc[phq9_df['score']>=20,'phq9_very_severe_cutoff'] = True
    phq9_df = phq9_df.drop(columns=['score'])
    return phq9_df


def passCAPECutoff(subscale_scores_df):
    """CAPE cutoff for psychotic traits

    Args:
        subscale_scores_df (pandas.DataFrame): each subject's subscale scores

    Returns:
        pandas.DataFrame: boolean values whether the subject was above those scores. 
    """
    cape_df = subscale_scores_df.loc[subscale_scores_df.subscale=='cape_pos_neg_positive_symptoms',['score','subject_id']]
    cape_df['cape_cutoff'] = np.zeros(len(cape_df)).astype(bool)
    cape_df.loc[cape_df['score']>=1.7,'cape_cutoff'] = True
    cape_df = cape_df.loc[:,['cape_cutoff','subject_id']]
    return cape_df


def makeSubstancesDF(data_questionnaires,substances_list=None):
    """Makes a substaces dataframe including all subjects

    Args:
        data_questionnaires (pandas.DataFrame): questionnaire responses 
        substances_list (_type_, optional): list of possible substances. Defaults to None.

    Returns:
        pandas.DataFrame: columns are substances, rows subjects and values indicate whether they reported using. 
    """
    # Convert to tuples (avoid unhashable type error)
    data_questionnaires = convertUnhashableDFcolumn(data_questionnaires,'substances')
    # Make the list of substances
    substances_list = []
    substances_details_list = []
    for subject_substances in data_questionnaires.loc[:,['substances']].drop_duplicates()['substances']:
        if type(subject_substances) is str:
                subject_substances = eval(subject_substances)
        for substance in subject_substances:
            if ('-NA' in substance) or ('other_detail' in substance):
                continue
            if len(substance.split('_')) > 1:
                substances_details_list.append(substance)
            else:
                substances_list.append(substance)
    substances_list = np.unique(substances_list)
    substances_details_list = np.unique(substances_details_list)
    # Create a dictionary
    substances_subset_df = data_questionnaires.loc[:,['subject_id','substances']].drop_duplicates()
    substances_dict = {}
    substances_dict['subject_id'] = substances_subset_df['subject_id'].to_numpy().astype(int)
    for substance in np.concatenate((substances_list,substances_details_list)):
        substances_dict[substance] = np.zeros(len(substances_subset_df)).astype(bool)
    # Record whether each subject has used the substance
    for i, substances in enumerate(substances_subset_df['substances']):
        if type(substances) is str:
            substances = eval(substances)
        if len(substances) > 0:
            for substance in substances:
                if ('nicotine' in substance.lower()) and ('tobacco' in substances_list):
                    substances_dict['tobacco'][i] = True
                    continue
                if substance in np.concatenate((substances_list,substances_details_list)):
                    substances_dict[substance][i] = True
                    continue
    substances_df = pd.DataFrame(substances_dict)
    
    return substances_df


def makePsychHistoryDF(data_questionnaires,psych_history_list=None):
    """Create a dataframe of subject's reported psych history.

    Args:
        data_questionnaires (pandas.DataFrame): DF of questionnaire responses
        psych_history_list (_type_, optional): possible psych diagnoses. Defaults to None.

    Returns:
        pandas.DataFrame: Columns are psychiatric conditions, rows subjects and values whether they report diagnosis. 
    """
    # Convert to tuples (avoid unhashable type error)
    data_questionnaires = convertUnhashableDFcolumn(data_questionnaires,'psych_history')
    # get list of psych_history
    if psych_history_list is None:
        psych_history_list = []
        for val in data_questionnaires.loc[:,['psych_history']].drop_duplicates()['psych_history']:
            if len(val) > 0:
                for val_ in val:
                    psych_history_list += [val_.split('-')[0]]
        psych_history_list = np.unique(psych_history_list)
    # Create dictionary to use for psych_history recording
    psych_history_subset_df = data_questionnaires.loc[:,['subject_id','psych_history']].drop_duplicates()
    psych_history_dict = {}
    psych_history_dict['subject_id'] = psych_history_subset_df['subject_id'].to_numpy().astype(int)
    for psych_history in psych_history_list:
        psych_history_dict[psych_history] = np.zeros(len(psych_history_subset_df)).astype(bool)
        psych_history_dict[f'{psych_history}_age'] = [np.nan] * len(psych_history_dict['subject_id'])
    # Record whether each subject has used the psych_history
    for psych_history, i in zip(psych_history_subset_df['psych_history'],np.arange(len(psych_history_subset_df))):
        if len(psych_history) > 0:
            for psych_history_ in psych_history:
                if psych_history_.split('-')[0] not in psych_history_list:
                    print(f'{psych_history_list} not in list')
                    continue
                psych_history_dict[psych_history_.split('-')[0]][i] = True
                if len(psych_history_.split('-')) > 1:
                    psych_history_dict[f"{psych_history_.split('-')[0]}_age"][i] = int(psych_history_.split('-')[1])
                else:
                    psych_history_dict[f"{psych_history_.split('-')[0]}_age"][i] = np.nan
    # Convert to dataframe
    psych_history_df = pd.DataFrame(psych_history_dict)

    return psych_history_df


def asrsScreenerAnswerConversion(data_questionnaires):
    """Conterts ASRS screener answers to the official scoring rubric

    Args:
        data_questionnaires (pandas.DataFrame): trial-by-trial questionnaire responses

    Returns:
        numpy.array: array of revised answers
    """
    revised_answers = np.zeros(np.sum(data_questionnaires['questionnaire_name']=='asrs'))
    asrs_score_key = np.array([
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1] 
    ])

    i=0
    for _, row in data_questionnaires.loc[data_questionnaires['questionnaire_name']=='asrs',['answer','questionnaire_question_number']].iterrows():
        revised_answers[i] = asrs_score_key[row['questionnaire_question_number']-1,row['answer']]
        i+=1
        
    return revised_answers


def getSubdimMatrixDF(source='individual',subscale_scores_df=None,data_questionnaires=None):
    """Creates matrix from dataframe of questionniare rsponses to be used in factor analysis and other. 

    Args:
        source (str, optional): 'subscale' or 'individual'. Whether to create the matrix on individual questions or subscale scores. Defaults to 'individual'.
        subscale_scores_df (_type_, optional): summarized subscale scores. Defaults to None.
        data_questionnaires (_type_, optional): individual-question responses. Defaults to None.

    Raises:
        ValueError: if an invalid source is specified

    Returns:
        _type_: _description_
    """
    # The source of the values for the matrix
    if source == 'subscale':
        vals = subscale_scores_df.pivot(index='subject_id',columns='subscale',values='score').apply(zscore)
    elif source == 'individual':
        data_questionnaires['subscale'][data_questionnaires['subscale'].isnull()] = 'NA'
        data_questionnaires['questionnaire_name_number'] = data_questionnaires['questionnaire_name'].str.lower() + ' ' + \
            data_questionnaires['subscale'].str.lower() + ' ' + data_questionnaires['questionnaire_question_number'].astype(str)
        if 'possible_answers' in data_questionnaires.columns:
            data_questionnaires = convertUnhashableDFcolumn(data_questionnaires,'possible_answers')
        if 'substances' in data_questionnaires.columns:
            data_questionnaires = convertUnhashableDFcolumn(data_questionnaires,'substances')
        if 'psych_history' in data_questionnaires.columns:
            data_questionnaires = convertUnhashableDFcolumn(data_questionnaires,'psych_history')
        data_questionnaires.drop_duplicates(inplace=True)
        vals = data_questionnaires.pivot_table(values='answer', index='subject_id', columns='questionnaire_name_number',\
            aggfunc='first').apply(zscore)
        if 'att_check na 1' in vals.columns:
            vals.drop(columns=['att_check na 1'],inplace=True)
        if 'att_check na 2' in vals.columns:
            vals.drop(columns=['att_check na 2'],inplace=True)
        if 'bapq attention check 0' in vals.columns:
            vals.drop(columns=['bapq attention check 0'],inplace=True)
        if 'att_check_list na 0' in vals.columns:
            vals.drop(columns=['att_check_list na 0'],inplace=True)
    else:
        raise ValueError('Invalid source')
    return vals


## PLOTTING FUNCTIONS ##

def decompositionFigure(estimates,outcomes,bins=None,ttl='',fontsize=12,save_fig=False):
    """Plots the Brier decomposition for a subject's responses

    Args:
        estimates (vec): trial-by-trial estimates
        outcomes (vec): trial-by-trial outcomes
        bins (_type_, optional): bins for summary. Defaults to None.
        ttl (str, optional): plot title. Defaults to ''.
        fontsize (int, optional): font size. Defaults to 12.
        save_fig (bool, optional): whether to save figure. Defaults to False.

    Returns:
        fig, ax
    """
    #Calculate values
    estimates_binned, outcomes_binned, _ = binEstimatesOutcomes(estimates,outcomes)
    O = calcOutcomeVariance(outcomes)
    C = calcCalibration(estimates,outcomes,bins=bins)
    R = calcResolution(estimates,outcomes,bins=bins)
    PS = calcBrierScore(estimates,outcomes,by_trial=False)
    #Organize data
    df = pd.DataFrame({
        'Value': [O,C,R,O+C-R,PS],
        'labels': ['O','C','R','O + C - R','Brier PS']

    })
    #Plot it
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(estimates_binned,outcomes_binned,color='k')
    ax[0].axis('square')
    ax[0].set_xlim([-0.05,1.05])
    ax[0].set_ylim([-0.05,1.05])
    ax[0].set_title('Calibration Plot',fontsize=fontsize+1)
    ax[0].set_xlabel('Estimated Likelihood',fontsize=fontsize)
    ax[0].set_ylabel('Actual Occurance',fontsize=fontsize)

    ax[1] = sns.barplot(data=df,ax=ax[1],x='labels',y='Value')
    ax[1].tick_params(labelrotation=45)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Value',fontsize=fontsize)
    fig.suptitle(ttl,fontsize=fontsize+2,fontweight='bold')
    plt.tight_layout()
    if save_fig:
        fig.savefig(f'{ttl}.png',dpi=300)
    return fig, ax