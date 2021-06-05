import csv
import json
import sys
from datetime import datetime

# Function to load the json file data
def loadjsondata(filepath):

    data = []
    for line in open(filepath, 'r'):
        data.append(json.loads(line))
    return data


# Function to get the different
def getdifferentskills(tmpdata):
    skillslist = []

    # 1- Walk through the data and get the different skills
    for element in tmpdata:
        if 'exercise' in element and 'skills' in element['exercise']:
            for skill in element['exercise']['skills']:
                if 'name' in skill:
                    if skill['name'] not in skillslist:
                        skillslist.append(skill['name'])
    return skillslist


def writepedagogicalsoftwareinterventionscsv(interventions):

    row_list=[]
    row_list.append(['student_gender', 'student_mother_tongue', 'student_age', 'student_competence',
                    'student_motivation', 'exercise_skill_paralellism', 'exercise_skill_logical_thinking',
                    'exercise_skill_flow_control', 'exercise_skill_user_interactivity', 'exercise_skill_information_representation',
                    'exercise_skill_abstraction', 'exercise_skill_syncronization', 'exercise_valid_solution', 'exercise_is_evaluation',
                    'exercise_level', 'solution_distance_family_distance', 'solution_distance_element_distance',
                    'solution_distance_position_distance', 'solution_distance_input_distance', 'solution_distance_total_distance',
                    'request_help', 'seconds_help_open', 'finished_exercise', 'valid_solution', 'grade', 'totalSeconds'])

    for element in interventions:

        student_gender = None
        student_age = None
        total_seconds = None
        student_mother_tongue = 0
        student_competence = 0
        student_motivation = 0

        exercise_skill_paralellism = 0
        exercise_skill_logical_thinking = 0
        exercise_skill_flow_control = 0
        exercise_skill_user_interactivity = 0
        exercise_skill_information_representation = 0
        exercise_skill_abstraction = 0
        exercise_skill_syncronization = 0
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

        # Time calculation between the last login and the current action
        if 'dateTime' in element and 'lastLogin' in element:
            date_time_obj = datetime.strptime(element['dateTime'], '%Y-%m-%d %H:%M:%S.%f')
            last_login_obj = datetime.strptime(element['lastLogin'], '%Y-%m-%d %H:%M:%S')
            difference = (date_time_obj - last_login_obj)
            total_seconds = difference.total_seconds()


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
                        exercise_skill_paralellism = skill['score']
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
                        exercise_skill_syncronization = skill['score']
            if 'valid_solution' in element['exercise']:
                exercise_valid_solution = element['exercise']['valid_solution']
            if 'isEvaluation' in element['exercise']:
                exercise_is_evaluation = element['exercise']['isEvaluation']
            if 'level' in element['exercise']:
                exercise_level = element['exercise']['level']

        #Solution distance information
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

        if 'request_help' in element:
            request_help = element['request_help']
        if 'seconds_help_open' in element:
            seconds_help_open = element['seconds_help_open']
        if 'finished_exercise' in element:
            finished_exercise = element['finished_exercise']
        if 'valid_solution' in element:
            valid_solution = element['valid_solution']
        if 'grade' in element:
            grade = element['grade']

        # Creating  the row of the csv
        row_list.append([student_gender, student_mother_tongue, student_age, student_competence,
                         student_motivation, exercise_skill_paralellism, exercise_skill_logical_thinking,
                         exercise_skill_flow_control, exercise_skill_user_interactivity, exercise_skill_information_representation,
                         exercise_skill_abstraction, exercise_skill_syncronization, exercise_valid_solution, exercise_is_evaluation,
                         exercise_level, solution_distance_family_distance, solution_distance_element_distance,
                         solution_distance_position_distance, solution_distance_input_distance, solution_distance_total_distance,
                         request_help, seconds_help_open, finished_exercise, valid_solution, grade, total_seconds])

    return row_list


# 1- Gets the json data
data = loadjsondata(sys.argv[1])

# 2-Get the different skills from the data
skills = getdifferentskills(data)

# 3- Creating the csv rows
rowList = writepedagogicalsoftwareinterventionscsv(data)

# 4- Writing the csv file
with open(sys.argv[2], 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rowList)