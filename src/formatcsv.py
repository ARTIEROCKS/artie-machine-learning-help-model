import csv
import json
import sys

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

    rowList=[]
    rowList.append(['studentGender', 'studentMotherTongue', 'studentAge', 'studentCompetence',
                    'studentMotivation', 'exerciseSkillParalellism', 'exerciseSkillLogicalThinking',
                    'exerciseSkillFlowControl', 'exerciseSkillUserInteractivity', 'exerciseSkillInformationRepresentation',
                    'exerciseSkillAbstraction', 'exerciseSkillSyncronization', 'exerciseValidSolution', 'exerciseIsEvaluation',
                    'exerciseLevel', 'solutionDistanceFamilyDistance', 'solutionDistanceElementDistance',
                    'solutionDistancePositionDistance', 'solutionDistanceInputDistance', 'solutionDistanceTotalDistance',
                    'dateTime', 'requestHelp', 'secondsHelpOpen', 'finishedExercise', 'validSolution', 'grade',
                    'lastLogin'])

    for element in interventions:

        dateTime = None
        lastLogin = None
        studentGender = None
        studentAge = None
        studentMotherTongue = 0
        studentCompetence = 0
        studentMotivation = 0

        exerciseSkillParalellism = 0
        exerciseSkillLogicalThinking = 0
        exerciseSkillFlowControl = 0
        exerciseSkillUserInteractivity = 0
        exerciseSkillInformationRepresentation = 0
        exerciseSkillAbstraction = 0
        exerciseSkillSyncronization = 0
        exerciseValidSolution = 0
        exerciseLevel = 0
        solutionDistanceFamilyDistance = 0
        solutionDistanceElementDistance = 0
        solutionDistancePositionDistance = 0
        solutionDistanceInputDistance = 0
        solutionDistanceTotalDistance = 0
        secondsHelpOpen = 0
        validSolution = 0
        grade = 0

        exerciseIsEvaluation = False
        requestHelp = False
        finishedExercise = False

        # Student information
        if 'student' in element:
            if 'gender' in element['student']:
                studentGender = element['student']['gender']
            if 'age' in element['student']:
                studentAge = element['student']['age']
            if 'motherTongue' in element['student']:
                studentMotherTongue = element['student']['motherTongue']
            if 'competence' in element['student']:
                studentCompetence = element['student']['competence']
            if 'motivation' in element['student']:
                studentMotivation = element['student']['motivation']

        # Exercise information
        if 'exercise' in element:
            if 'skills' in element['exercise']:
                for skill in element['exercise']['skills']:
                    if skill['name'] == 'Paralelismo':
                        exerciseSkillParalellism = skill['score']
                    elif skill['name'] == 'Pensamiento lógico':
                        exerciseSkillLogicalThinking = skill['score']
                    elif skill['name'] == 'Control de flujo':
                        exerciseSkillFlowControl = skill['score']
                    elif skill['name'] == 'Interactividad con el usuario':
                        exerciseSkillUserInteractivity = skill['score']
                    elif skill['name'] == 'Representación de la información':
                        exerciseSkillInformationRepresentation = skill['score']
                    elif skill['name'] == 'Abstracción':
                        exerciseSkillAbstraction = skill['score']
                    elif skill['name'] == 'Sincronización':
                        exerciseSkillSyncronization = skill['score']
            if 'validSolution' in element['exercise']:
                exerciseValidSolution = element['exercise']['validSolution']
            if 'isEvaluation' in element['exercise']:
                exerciseIsEvaluation = element['exercise']['isEvaluation']
            if 'level' in element['exercise']:
                exerciseLevel = element['exercise']['level']

        #Solution distance information
        if 'solutionDistance' in element:
            if 'familyDistance' in element['solutionDistance']:
                solutionDistanceFamilyDistance = element['solutionDistance']['familyDistance']
            if 'elementDistance' in element['solutionDistance']:
                solutionDistanceElementDistance = element['solutionDistance']['elementDistance']
            if 'positionDistance' in element['solutionDistance']:
                solutionDistancePositionDistance = element['solutionDistance']['positionDistance']
            if 'inputDistance' in element['solutionDistance']:
                solutionDistanceInputDistance = element['solutionDistance']['inputDistance']
            if 'totalDistance' in element['solutionDistance']:
                solutionDistanceTotalDistance = element['solutionDistance']['totalDistance']

        if 'dateTime' in element:
            dateTime = element['dateTime']
        if 'requestHelp' in element:
            requestHelp = element['requestHelp']
        if 'secondsHelpOpen' in element:
            secondsHelpOpen = element['secondsHelpOpen']
        if 'finishedExercise' in element:
            finishedExercise = element['finishedExercise']
        if 'validSolution' in element:
            validSolution = element['validSolution']
        if 'grade' in element:
            grade = element['grade']
        if 'lastLogin' in element:
            lastLogin = element['lastLogin']

        # Creating  the row of the csv
        rowList.append([studentGender, studentMotherTongue, studentAge, studentCompetence,
                        studentMotivation, exerciseSkillParalellism, exerciseSkillLogicalThinking,
                        exerciseSkillFlowControl, exerciseSkillUserInteractivity, exerciseSkillInformationRepresentation,
                        exerciseSkillAbstraction, exerciseSkillSyncronization, exerciseValidSolution, exerciseIsEvaluation,
                        exerciseLevel, solutionDistanceFamilyDistance, solutionDistanceElementDistance,
                        solutionDistancePositionDistance, solutionDistanceInputDistance, solutionDistanceTotalDistance,
                        dateTime, requestHelp, secondsHelpOpen, finishedExercise, validSolution, grade,
                        lastLogin])

    return rowList


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