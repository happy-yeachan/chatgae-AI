import shutil

from flask import Flask, request, jsonify, render_template,Blueprint
import datetime
import os
import random
import subprocess
import pymysql
import sys

from nose.learning import learning


bp= Blueprint('main',__name__,url_prefix='/')

#path adjusting
def get_path(path):
    change_path = path.replace("\\",'/')
    return change_path


# [등록 API]
@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        dogNose2 = request.files['dogNose2']
        dogNose3 = request.files['dogNose3']
        dogNose4 = request.files['dogNose4']
        dogNose5 = request.files['dogNose5']
        forlookup = request.files['dogNose1']

        # 스트림을 메모리에 저장 후 다시 사용 가능하도록 설정
        dogNose1_bytes = forlookup.read()
        forlookup.seek(0)  # 스트림 초기화

        now = datetime.datetime.now()
        formoment1 = f"{now.year}{now.month}{now.hour}{now.minute}{now.second}"
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
    print(compare)
    if compare[0][-2:] != '-1':
        return jsonify({'message': '이미등록된강아지', 'status': 0, 'id': compare[0][-1:] })
    else:
        image_base_path = "nose/SVM-Classifier/image"
        new_folder_number = get_next_folder_number(image_base_path)
        new_folder_path = f"{image_base_path}/{new_folder_number}"

        # 폴더 생성
        createFolder(new_folder_path)

        # 5장의 사진 저장
        # 1.jpg 저장 (바이너리 파일로 직접 저장)
        with open(f"{new_folder_path}/1.jpg", "wb") as f:
            f.write(dogNose1_bytes)
        dogNose2.save(f"{new_folder_path}/2.jpg")
        dogNose3.save(f"{new_folder_path}/3.jpg")
        dogNose4.save(f"{new_folder_path}/4.jpg")
        dogNose5.save(f"{new_folder_path}/5.jpg")

        learning(new_folder_path)

        return jsonify({'message': '등록완료', 'status': 1,  'id': new_folder_path[26:]})

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
        print(SVMresult)
        if SVMresult[0][-2:] == '-1':
            return jsonify({'isSuccess': False,  'message': '조회된 강아지가 없습니다'})

        # 조회 성공한 경우
        else:
            return jsonify({ 'isSuccess': True, 'id': SVMresult[0][-1:], 'message': '조회를 성공했습니다'})


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

# 가장 큰 폴더 번호 찾기 + 1
def get_next_folder_number(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return 0  # 첫 번째 폴더는 0부터 시작

    # 폴더 내 모든 숫자 폴더 찾기
    existing_numbers = []
    for folder_name in os.listdir(base_path):
        if folder_name.isdigit():  # 숫자로 된 폴더만 찾기
            existing_numbers.append(int(folder_name))

    # 가장 큰 숫자의 다음 숫자 반환
    return max(existing_numbers, default=-1) + 1

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

