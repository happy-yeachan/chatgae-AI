import os
import numpy as np
import cv2

import pickle
import sklearn
import argparse
import time


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from histo_clahe import histo_clahe

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--dir', default='Dog-Data',help='dataset directory')
parser.add_argument('--test', default='pedestrian.png',help='test image data')
parser.add_argument('--option',default='test',help='test or register')
opt = parser.parse_args()


#get_path
def get_path(path):
    change_path = path.replace("\\",'/')
    return change_path

# test path
train_path = get_path(os.getcwd()+"/nose/SVM-Classifier/Dog-Data/train")
dog_data_path = get_path(os.getcwd()+"/nose/SVM-Classifier/Dog-Data")



#read data
def read_data(label2id):
    X = []
    Y = []
    for label in os.listdir(train_path):
        if os.path.isdir(os.path.join(train_path, label)):
            for img_file in os.listdir(os.path.join(train_path, label)):
                img_path = str(train_path+"/"+label+"/"+img_file)
                img_array = np.fromfile(img_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                X.append(img)
                Y.append(label2id[label])
    return X, Y

# feature exraction with SIFT
def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.xfeatures2d.SIFT_create(nfeatures =200, nOctaveLayers =3, contrastThreshold=0.0005)

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)

    return image_descriptors

#Kmeans bow
def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_
    return bow_dict

#image to vector.
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis=1)
            for j in argmin:
                features[j] += 1
        X_features.append(features)
    return X_features

def main():
    start = time.time()
    # Label to id
   
    file_list = os.listdir(train_path)
    label2id = {}
    idx = 0
    for i, label in enumerate(file_list):
        if label == ".DS_Store":
            continue
        label2id[label] = idx
        idx += 1
    X, Y = read_data(label2id)

    image_descriptors = extract_sift_features(X)

    # all descriptors
    all_descriptors = []
    for descriptors in image_descriptors:
        if descriptors is not None:
            for des in descriptors:
                all_descriptors.append(des)


    num_clusters = 500

    if not os.path.isfile(dog_data_path+'/bow.pkl'):
        BoW = kmeans_bow(all_descriptors, num_clusters)
        pickle.dump(BoW, open(dog_data_path+'/bow.pkl', 'wb'))
    else:
        BoW = pickle.load(open(dog_data_path+'/bow.pkl', 'rb'))

    X_features = create_features_bow(image_descriptors, BoW, num_clusters)

    #Set model
    X_train = [] 
    X_test = []
    Y_train = []
    Y_test = []
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42, shuffle=True)

    # #Train SVM
    # svm = sklearn.svm.SVC(C = 100, probability=True)
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    param_grid = {'C': [1, 10, 100, 1000], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    grid_search.fit(X_train, Y_train)

    svm = grid_search.best_estimator_  # 최적의 SVM 모델 적용
    svm.fit(X_train, Y_train)

    #Train KNN
    knn = KNeighborsClassifier(n_neighbors=2,  weights='distance', leaf_size=200, p=2)
    knn.fit(X_train, Y_train)

    #predict
    #img_test = cv2.imread(opt.dir + '/test/' + opt.test)
    if opt.option == 'test':
        img_test = histo_clahe(dog_data_path + '/test/' + opt.test)
    elif opt.option == 'getpost':
        path = os.getcwd()+'/nose/SVM-Classifier/testimage/'
        register_path = get_path(path)
        img_test = histo_clahe(register_path + '/' + opt.test + '/' + opt.test + '.jpg')
    img = [img_test]
    
    img_sift_feature = extract_sift_features(img)
    img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)

    #---------------------------------
    #-----------evaluation------------
    #---------------------------------

    #predict SVM
    img_predict = svm.predict(img_bow_feature)
    #predict KNN
    img_predict2 = knn.predict(img_bow_feature)

    
    #prediction probability
    svm_prob = svm.predict_proba(img_bow_feature)[0][img_predict[0]]
    knn_prob = knn.predict_proba(img_bow_feature)[0][img_predict2[0]]


    for key, value in label2id.items():
        if value == img_predict[0]:
            svm_k = key
        if value == img_predict2[0]:
            knn_k = key

    result =""

    if svm_prob > 0.7 and knn_prob > 0.7 and svm_k == knn_k:
        result = f"{svm_k}, 등록된강아지, {max(svm_prob, knn_prob)}"
    elif svm_prob > 0.5 and knn_prob > 0.5 and svm_k == knn_k:
        result = f"{svm_k}, 등록된강아지 (낮은 확률), {max(svm_prob, knn_prob)}"
    else:
        result = "-1, 미등록강아지, " + str(max(svm_prob, knn_prob))

    #Accuracy
    if svm_prob > knn_prob:
        result =result+str(svm_prob)
    else :
        result =result+str(knn_prob)

    
    print(result)
    return result
if __name__ == "__main__":
    main()