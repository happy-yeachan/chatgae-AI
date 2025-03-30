# 🐶 찾개

"찾개"는 유실견을 쉽고 빠르게 신고하고, 반려동물 정보를 체계적으로 관리할 수 있는 서비스입니다. <br/>
반려동물을 잃어버린 경우 빠른 조회와 신고가 가능하며, 주변의 유실견 정보도 확인할 수 있습니다.

<br />

## 👩🏻‍💻 Developer
| danny.oh (오예찬) | sando.gang (강산아)|
|:---:|:---:|
|  <a href="https://github.com/miginho12"> <img src="https://avatars.githubusercontent.com/u/21968811?v=4" width=100px alt="_"/> </a> | <a href="https://github.com/miginho12"> <img src="https://avatars.githubusercontent.com/u/98865571?v=4" width=100px alt="_"/> </a> | 
|<a href="https://github.com/happy-yeachan">@happy-yeachan</a> |<a href="https://github.com/gsandoo">@gsandoo</a> |

<br />

## 🛠️ Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white)  
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=Flask&logoColor=white)  


<br />

## ✨ Main Feature

### 1. 비문인식 기반 유실견 조회
- 클라이언트가 촬영한 비문 이미지를 `multipart/form-data` 형식으로 서버에 전송합니다.
- Flask 서버는 업로드된 이미지를 OpenCV로 읽어 `numpy array`로 변환하고, `grayscale`로 전처리합니다.
- 특징 추출에는 **SIFT** 알고리즘을 활용하며, 추출된 벡터는 **BoW** 방식으로 변환됩니다.
- 변환된 벡터는 학습된 **SVM** 모델(`bow.pkl`)에 입력되어, 기존 등록된 비문과의 유사도를 기반으로 유실견 여부를 판단합니다.

### 2. 새로운 비문 등록 및 모델 학습
- `/register` 엔드포인트를 통해 5장의 비문 이미지를 입력받습니다 (`dogNose1`~`dogNose5`).
- 먼저 `dogNose1` 이미지를 기반으로 기존 등록 여부를 확인하고, 미등록 시 나머지 4장과 함께 총 5장을 저장합니다.
- 비문 이미지는 `nose/SVM-Classifier/image/{id}/1~5.jpg` 경로에 저장되며, 특징 벡터를 추출한 후 SVM 모델을 전체 재학습합니다.
- 모델은 `learning()` 함수를 통해 BoW 기반 벡터를 학습하고, 최종적으로 `bow.pkl`로 저장됩니다.

### 📦 이미지 처리 포맷 및 전처리 흐름
1. 클라이언트는 `multipart/form-data` 형식으로 5장의 이미지를 업로드합니다.
2. 서버는 파일 스트림을 바이너리로 읽고, 필요한 경우 `.read()` 후 `.seek(0)`으로 복원합니다.
3. 저장 경로를 생성한 후 OpenCV를 이용해 `.jpg` 포맷으로 저장 및 전처리를 진행합니다.
4. SIFT로 특징 추출 → BoW로 변환 → SVM으로 분류/학습.




<br />
 
## 💻 Screen Preview

| 앱 진입 화면 | 스플래시 화면 | 홈 화면 |
|:-----: | :-----: | :-----: |
| ![image](https://github.com/user-attachments/assets/f9effaf0-6e5f-4b66-bbd5-ee9e198129b4) | ![image](https://github.com/user-attachments/assets/88dd7314-404e-4354-b7d0-b7036611cfd1) | ![image](https://github.com/user-attachments/assets/17b3182a-db98-4229-8494-db5c7bb84ee2) |


| 가이드 화면 | 비문 촬영 카메라 화면 | 강아지 조회 성공 화면 | 강아지 조회 실패 화면 | 
| :-----: | :-----: | :-----:| :-----:|
| ![image](https://github.com/user-attachments/assets/154e4038-fea4-4f1a-8ec6-8e467fdcdfa4) | ![image](https://github.com/user-attachments/assets/74468f4a-afff-4aa8-ae81-d1ce55fd3414) | ![image](https://github.com/user-attachments/assets/99bfb799-98e7-4f11-9001-ae6bb37c1f0b) | ![image](https://github.com/user-attachments/assets/8cb092af-694c-41e3-9f44-e0d1445a62e6) |

| 반려견 등록 화면 1 (이름, 프로필 사진) | 반려견 등록 화면 2 (비문 사진 등록) | 반려견 등록 화면 3 (견종 등록) | 변려견 등록 완료 화면 |
|:-----: | :-----:|:-----: | :-----:|
| ![image](https://github.com/user-attachments/assets/2ec32688-94c6-4a79-aaa8-b332091988a7) | ![image](https://github.com/user-attachments/assets/369a033e-f194-4251-b5eb-0246c404ffe2) | ![image](https://github.com/user-attachments/assets/eea08704-76d5-4c51-ba93-34d8484ce91f) | ![image](https://github.com/user-attachments/assets/71ab0384-1f62-4e85-b082-aaa8e6fa92cf) |
