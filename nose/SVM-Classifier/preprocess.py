import os
import cv2
import numpy as np

def create_folder(directory):
    """ 디렉토리가 없으면 생성하는 함수 """
    if not os.path.exists(directory):
        print("Creating directory:", directory)
        os.makedirs(directory)

def histo_clahe(img_path):
    """
    이미지 경로를 입력받아 CLAHE를 적용한 이미지를 반환하는 함수
    """
    print(f"Processing: {img_path}")  # 디버깅용 출력
    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다. 경로를 확인하세요: {img_path}")

    height, width, _ = img.shape

    # 이미지 크기가 너무 크면 절반으로 줄이기
    while height >= 600 or width >= 600:
        img = cv2.resize(img, (width // 2, height // 2))
        height, width, _ = img.shape

    # YUV 컬러 스페이스 변환
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 밝기 채널에 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])  # CLAHE 적용

    # 다시 BGR로 변환
    img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_clahe

# 이미지 폴더 경로
image_dir = './image/'
file_lists = os.listdir(image_dir)

# 회전 각도 설정
rotate_angles = [0, 15, 30, 45, -15, -30, -45]

# 이미지 처리 및 저장
for folder in file_lists:
    file_lists = os.listdir(image_dir + folder)
    for file in file_lists:
        file_path = os.path.join(image_dir, folder, file)
        print(file_path)
        # 파일이 아니면 건너뜀 (폴더 제외)
        if not os.path.isfile(file_path):
            continue

        file_name, file_ext = os.path.splitext(file)
        
        save_dir = os.path.join('Dog-Data', 'train', folder)
        create_folder(save_dir)

        saved_files = []  # 저장한 파일 목록

        # CLAHE 적용
        img = histo_clahe(file_path)
        height, width, _ = img.shape

        # 원본 크기에서 회전하여 저장
        for i, angle in enumerate(rotate_angles[:5]):  # 처음 5개 각도만 적용
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            dst = cv2.warpAffine(img, matrix, (width, height))
            save_path = os.path.join(save_dir, f"{file_name}-{i}{file_ext}")
            cv2.imwrite(save_path, dst)
            saved_files.append(save_path)

        # 이미지 크기를 절반으로 줄이기
        img_resized = cv2.resize(img, (width // 2, height // 2))
        height, width, _ = img_resized.shape

        # 줄인 크기에서 회전하여 저장
        for i, angle in enumerate(rotate_angles):
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            dst = cv2.warpAffine(img_resized, matrix, (width, height))
            save_path = os.path.join(save_dir, f"{file_name}-{i+5}{file_ext}")
            cv2.imwrite(save_path, dst)
            saved_files.append(save_path)

        print(f"Saved {len(saved_files)} images to {save_dir}")
