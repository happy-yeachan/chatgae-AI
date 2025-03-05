import shutil

from flask import Flask, request, jsonify, render_template,Blueprint
import datetime
import os
import random
import subprocess
import pymysql
import sys


bp= Blueprint('main',__name__,url_prefix='/')

#path adjusting
def get_path(path):
    change_path = path.replace("\\",'/')
    return change_path


# [등록 API]
@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        print(request.form)
        dogNose2 = request.files['dogNose2']
        dogNose3 = request.files['dogNose3']
        dogNose4 = request.files['dogNose4']
        dogNose5 = request.files['dogNose5']
        global details
        details = request.form
        profile = request.files['dogProfile']
        forlookup = request.files['dogNose1']
        now = datetime.datetime.now()
        formoment = str(now.year) + str(now.month) + str(now.hour) + str(now.minute) + str(now.second)
        formoment1 = str(formoment)
        print("formoment1 = " + formoment1)
   
    try:
        # 경로 설정
        createFolder('nose/SVM-Classifier/testimage/%s' % (formoment1))
        forlookup.save('nose/SVM-Classifier/testimage/%s/%s.jpg' % (formoment1, formoment1))
        
       
        print("--- register start ---")
        # 5장 중 1장만 ml코드 돌리기
        result = getSVMResultForRegister(formoment1)
        compare = result.split(',')
        print(compare)
        if compare == ['']:
            raise Exception('error')

    except Exception as e:
        print("ML코드가 안돌아가서 등록 실패", e)
        return jsonify({'message': 'fail'
                        })

    isRegistered = None
    if compare[1] == '등록된강아지':
        return jsonify({'message': 'fail', 'status': 0, 'id': compare[0] })
    else:
        return jsonify({'message': 'success', 'status': 1,  'id': compare[0]})

# [조회 API]
@bp.route('/lookup', methods=['GET', 'POST'])
def lookup():
    if request.method == 'POST':
        lookupimg = request.files['dogNose']
        now1 = datetime.datetime.now()
        formomentLookup = str(now1.year) + str(now1.month) + str(now1.hour) + str(now1.minute) + str(now1.second)
        formomentLookup1 = str(formomentLookup)
        print(formomentLookup1)

        # 경로 설정
        path = get_path('nose/SVM-Classifier')
        print(path)
        createFolder(path+'/testimage/%s' % (formomentLookup1))
        lookupimg.save(path+'/testimage/%s/%s.jpg' % (formomentLookup1, formomentLookup1))

        try:
            print("--- lookup start ---")
            print("조회 경로 = " + os.getcwd())
            result = getSVMResult(formomentLookup1)
            SVMresult = result.split(',')
            if SVMresult == ['']:
                raise Exception('attribute error')
        except Exception as e:
            print("[조회] ML코드가 작동하지 않아 조회가 되지 않습니다", e)
            return jsonify({"message": "fail"})

        if SVMresult[1] == '미등록강아지':
            return jsonify({'isSuccess': False, 'message': '조회된 강아지가 없습니다'})

        # 조회 성공한 경우
        else:
                return jsonify({ 'isSuccess': True, 'id': SVMresult[0], 'message': '조회를 성공했습니다'})


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def uniquenumber(details):
    date_time = datetime.datetime.now()
    alist=[]
    for i in range(1):
        a = random.randint(1, 100)
        alist.append(a)
        while a in alist:
            a = random.randint(1, 100)
    alist.append(a)

    unique = str(details)

    reg_num = (str(date_time.year) + str(date_time.month) + str(date_time.day) +str(date_time.second)+ str(a)+unique) #  년도 + 월 + 일 + 초 + 1~100 난수 + 뒷자리
    return reg_num




def getSVMResult(formomentLookup):
    cmd =['python','nose/SVM-Classifier/Classifier.py','--test','%s' %(formomentLookup) , '--option','%s' %('getpost')]
    fd_popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True).stdout
    data = fd_popen.read().strip()
    print(data)
    fd_popen.close()
    # os.chdir('../')
    return data

def getSVMResultForRegister(formoment):    
    cmd =['python','nose/SVM-Classifier/Classifier.py','--test','%s' %(formoment) , '--option'  , '%s' %('getpost')]
    fd_popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True).stdout
    data = fd_popen.read().strip()
    print(" data 는 = " +data)
    fd_popen.close()
    # os.chdir('../')
    return data


def createProfileFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

