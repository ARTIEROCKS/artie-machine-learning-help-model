import csv
import json
import sys
import numpy as np
from datetime import datetime


# Function to load the json file data
def loadjsondata(filepath):
    data = []
    for line in open(filepath, 'r'):
        data.append(json.loads(line))

    sortedData = sorted(data, key=lambda x: (x['student']['_id'], x['lastLogin'], x['dateTime']))
    return sortedData


# Function to get the first actions of the exercise
def getfirstaction(interventions):
    first_actions = {}
    student_id = None
    date_time = None
    last_login = None
    exercise_id = None
    for element in interventions:
        if 'student' in element:
            if '_id' in element['student']:
                student_id = element['student']['_id']
        if 'dateTime' in element:
            date_time = datetime.strptime(element['dateTime'], '%Y-%m-%d %H:%M:%S.%f')
        if 'lastLogin' in element:
            last_login = element['lastLogin']
        if 'exercise' in element:
            if '_id' in element['exercise']:
                exercise_id = element['exercise']['_id']

        if student_id is not None and date_time is not None and last_login is not None and exercise_id is not None:
            if student_id + '_' + exercise_id + '_' + last_login in first_actions.keys():
                if date_time < first_actions[student_id + '_' + exercise_id + '_' + last_login]:
                    first_actions[student_id + '_' + exercise_id + '_' + last_login] = date_time
            else:
                first_actions[student_id + '_' + exercise_id + '_' + last_login] = date_time

    return first_actions

def check_groupid_timestamp(group_id_timestamp, group_id, timestamp):

    #Checks if the group id exists in the dictionary
    if group_id in group_id_timestamp.keys():
        arr_timestamps = group_id_timestamp[group_id]
        if timestamp in arr_timestamps:
            return True, group_id_timestamp
        else:
            np.append(arr_timestamps, timestamp)
            group_id_timestamp[group_id] = arr_timestamps
            return False, group_id_timestamp
    else:
        group_id_timestamp[group_id] = np.array(timestamp)
        return False, group_id_timestamp


# Function to write the software interventions in csv format
def writepedagogicalsoftwareinterventionscsv(interventions, first_actions):
    row_list = []
    group_id_timestamp = {}
    row_list.append(['group_id','date_time','student_gender', 'student_mother_tongue', 'student_age', 'student_competence',
                     'student_motivation', 'exercise_skill_parallelism', 'exercise_skill_logical_thinking',
                     'exercise_skill_flow_control', 'exercise_skill_user_interactivity',
                     'exercise_skill_information_representation',
                     'exercise_skill_abstraction', 'exercise_skill_synchronization', 'exercise_valid_solution',
                     'exercise_is_evaluation',
                     'exercise_level', 'solution_distance_family_distance', 'solution_distance_element_distance',
                     'solution_distance_position_distance', 'solution_distance_input_distance',
                     'solution_distance_total_distance', 'seconds_help_open', 'finished_exercise', 'valid_solution', 'grade',
                     'total_seconds','request_help'])

    for element in interventions:

        student_gender = None
        student_age = None
        total_seconds = None
        student_mother_tongue = 0
        student_competence = 0
        student_motivation = 0

        exercise_skill_parallelism = 0
        exercise_skill_logical_thinking = 0
        exercise_skill_flow_control = 0
        exercise_skill_user_interactivity = 0
        exercise_skill_information_representation = 0
        exercise_skill_abstraction = 0
        exercise_skill_synchronization = 0
        exercise_valid_solution = 0
        exercise_level = 0
        solution_distance_family_distance = 0
        solution_distance_element_distance = 0
        solution_distance_position_distance = 0
        solution_distance_input_distance = 0
        solution_distance_total_distance = 0
        seconds_help_open = 0
        valid_solution = 0
        grade = 0

        exercise_is_evaluation = False
        request_help = False
        finished_exercise = False

        student_id = None
        exercise_id = None
        last_login = None

        date_time = None
        group_id = None

        if 'student' in element:
            if '_id' in element['student']:
                student_id = element['student']['_id']

        if 'exercise' in element:
            if '_id' in element['exercise']:
                exercise_id = element['exercise']['_id']

        if 'lastLogin' in element:
            last_login = element['lastLogin']

        # Time calculation between the first action of the exercise and the current action
        if student_id is not None and last_login is not None and exercise_id is not None:
            if student_id + '_' + exercise_id + '_' + last_login in first_actions.keys():
                if 'dateTime' in element:
                    date_time_obj = datetime.strptime(element['dateTime'], '%Y-%m-%d %H:%M:%S.%f')
                    first_action = first_actions[student_id + '_' + exercise_id + '_' + last_login]
                    difference = (date_time_obj - first_action)
                    total_seconds = difference.total_seconds()
                    date_time = date_time_obj
                    group_id = student_id + '_' + exercise_id

        # Student information
        if 'student' in element:
            if 'gender' in element['student']:
                student_gender = element['student']['gender']
            if 'age' in element['student']:
                student_age = element['student']['age']
            if 'motherTongue' in element['student']:
                student_mother_tongue = element['student']['motherTongue']
            if 'competence' in element['student']:
                student_competence = element['student']['competence']
            if 'motivation' in element['student']:
                student_motivation = element['student']['motivation']

        # Exercise information
        if 'exercise' in element:
            if 'skills' in element['exercise']:
                for skill in element['exercise']['skills']:
                    if skill['name'] == 'Paralelismo':
                        exercise_skill_parallelism = skill['score']
                    elif skill['name'] == 'Pensamiento lógico':
                        exercise_skill_logical_thinking = skill['score']
                    elif skill['name'] == 'Control de flujo':
                        exercise_skill_flow_control = skill['score']
                    elif skill['name'] == 'Interactividad con el usuario':
                        exercise_skill_user_interactivity = skill['score']
                    elif skill['name'] == 'Representación de la información':
                        exercise_skill_information_representation = skill['score']
                    elif skill['name'] == 'Abstracción':
                        exercise_skill_abstraction = skill['score']
                    elif skill['name'] == 'Sincronización':
                        exercise_skill_synchronization = skill['score']
            if 'valid_solution' in element['exercise']:
                exercise_valid_solution = element['exercise']['valid_solution']
            if 'isEvaluation' in element['exercise']:
                exercise_is_evaluation = element['exercise']['isEvaluation']
            if 'level' in element['exercise']:
                exercise_level = element['exercise']['level']

        # Solution distance information
        if 'solutionDistance' in element:
            if 'familyDistance' in element['solutionDistance']:
                solution_distance_family_distance = element['solutionDistance']['familyDistance']
            if 'elementDistance' in element['solutionDistance']:
                solution_distance_element_distance = element['solutionDistance']['elementDistance']
            if 'positionDistance' in element['solutionDistance']:
                solution_distance_position_distance = element['solutionDistance']['positionDistance']
            if 'inputDistance' in element['solutionDistance']:
                solution_distance_input_distance = element['solutionDistance']['inputDistance']
            if 'totalDistance' in element['solutionDistance']:
                solution_distance_total_distance = element['solutionDistance']['totalDistance']

        if 'requestHelp' in element:
            request_help = element['requestHelp']
        if 'secondsHelpOpen' in element:
            seconds_help_open = element['secondsHelpOpen']
        if 'finishedExercise' in element:
            finished_exercise = element['finishedExercise']
        if 'validSolution' in element:
            valid_solution = element['validSolution']
        if 'grade' in element:
            grade = element['grade']


        #Checks if the group_id has already the same timestamp, to avoid including it
        result, group_id_timestamp = check_groupid_timestamp(group_id_timestamp, group_id, date_time)

        if not result:
            # Creating  the row of the csv
            row_list.append([group_id, date_time, student_gender, student_mother_tongue, student_age, student_competence,
                             student_motivation, exercise_skill_parallelism, exercise_skill_logical_thinking,
                             exercise_skill_flow_control, exercise_skill_user_interactivity,
                             exercise_skill_information_representation,
                             exercise_skill_abstraction, exercise_skill_synchronization, exercise_valid_solution,
                             int(exercise_is_evaluation),
                             exercise_level, solution_distance_family_distance, solution_distance_element_distance,
                             solution_distance_position_distance, solution_distance_input_distance,
                             solution_distance_total_distance,
                             seconds_help_open, int(finished_exercise), valid_solution, grade, total_seconds,
                             int(request_help)])

    return row_list


# 1- Gets the json data
data = loadjsondata(sys.argv[1])

# 2- Get the first action of each exercise
actions = getfirstaction(data)

# 3- Creating the csv rows
rowList = writepedagogicalsoftwareinterventionscsv(data, actions)

# 4- Writing the csv file
with open(sys.argv[2], 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rowList)
