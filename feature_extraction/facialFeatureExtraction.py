import tensorflow as tf
import numpy as np
import os
import cv2

# 특징 추출 AI 불러오기
with open(os.path.join('model', 'featureAI', 'featureAI.json'), 'r') as json_file:
    json_saved_model = json_file.read()

aimodel = tf.keras.models.model_from_json(json_saved_model)
aimodel.load_weights(os.path.join('model', 'featureAI', 'weights.hdf5')
aimodel.compile(optimizer='Adam')

# 이미지 특징 추출 (이미지가 있는 디렉토리 경로를 넣어야함)
def facialFeature(path):

    # 이미지 리스트
    imageList = os.listdir(path)
    
    # 특징 추출
    prediction = []
    for i in range(len(imageList)):
        image = cv2.imread(os.path.join(path, imageList[i]))
        image = cv2.resize(image, (64, 64))
        image = image / 255
        
        feature = aimodel.predict(np.expand_dims(image, axis=0))
    
        prediction.append(feature)
     
    return prediction


# 동영상 특징 추출 (동영상이 있는 디렉토리 경로와 자르고 싶은 ms단위 입력)
def facialVideoFeature(video_path, ms):
    video = cv2.VideoCapture(video_path)
    
    count = ms
    prediction = []
    
    while True:
        success, image = video.read()
        if success == False:
            break
        
        if (int(video.get(cv2.CAP_PROP_POS_MSEC)/ 100)) == count:
            # 프레임 전처리
            image = cv2.resize(image, (64, 64))
            image = image / 255.0  # 이미지 정규화
            
            # 표정 특징 추출
            feature = aimodel.predict(np.expand_dims(image, axis=0))
            
            prediction.append(feature)
            
            count += ms
        
    # 표정 특징 벡터를 numpy 배열로 변환
    prediction = np.array(prediction)
        
    # 동영상 파일 닫기
    video.release()
        
    return prediction