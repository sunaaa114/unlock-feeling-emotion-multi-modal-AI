import argparse
import whisper
import pandas as pd
import librosa
import os
import numpy as np
import facialFeatureExtraction
import moviepy.editor as mp
from moviepy.editor import *
import cv2
import torch
import torch.nn.functional as F
from python_speech_features import mfcc
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
from MultimodalClassifier import MultimodalClassifier
import boto3

os.chdir("/home/ec2-user/multi_modal")

# 비디오 경로명 넣으면 대화 내용 텍스트로 데이터프레임 반환
def video_to_text(video_path):
    stt_model = whisper.load_model("medium")

    result = stt_model.transcribe(video_path)

    df = pd.DataFrame(result['segments'])

    df = df[['start', 'end', 'text']]
    
    return df

# 텍스트 데이터프레임과 텍스트 열이름을 넣으면 텍스트 특징 추출
def add_kobert_embeddings(df, combined_text):
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    embeddings_numpy = []
    for text in df[combined_text]:
        tokens = tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([token_ids])
        
        model = BertModel.from_pretrained("monologg/kobert")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            model.to(device)
            tokens_tensor = tokens_tensor.to(device)
            outputs = model(tokens_tensor)
            sentence_embedding = outputs[0][:, 0, :].squeeze().cpu().numpy()
        embeddings_numpy.append(sentence_embedding)

    return embeddings_numpy

# 동영상 경로명과 시작 시간(초), 끝나는 시간(초)를 넣으면 해당 부분의 오디오 반환
def extract_video_audio(video_path, start_time_seconds, end_time_seconds):
    video_clip = mp.VideoFileClip(video_path)

    # Extract the audio from the video clip
    audio = video_clip.subclip(start_time_seconds, end_time_seconds).audio

    return audio

# 비디오의 오디오 부분 넣으면 오디오 특징 추출
def audio_feature_extraction(audio):
    audio_data = audio.to_soundarray()
    sample_rate = audio.fps
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=25), axis=0)

    return mfccs
    
# 오디오 특징 87개로 고정
def create_padded_array(data):
    if len(data) < 87:
        padded_data = list(data) + [0] * (87 - len(data))
    else:        
        padded_data = data[:87]
    
    return np.array(padded_data)

# 오디오 특징 정규화
def normalize_audio_features(audio_df):
  scaler = StandardScaler()

  audio_df2 = pd.DataFrame(columns=['normalized'])
  for k in range(len(audio_df)):
      features = audio_df.loc[k, 'feature'].reshape(1, -1)
      features_transposed = features.T

      # 정규화 적용
      scaled_features_transposed = scaler.fit_transform(features_transposed)

      # 다시 전치하여 shape을 (n_samples, n_features)로 변경
      scaled_features = scaled_features_transposed.T
      scaled_features = scaled_features.flatten()
      scaled_features.tolist()
      audio_df2.loc[k, 'normalized'] = scaled_features.tolist()

  audio_3 = pd.DataFrame(audio_df2['normalized'].values.tolist())
  audio_3 = audio_3.fillna(0)

  # 모든 행을 배열로 변환
  rows_as_arrays = []
  for index, row in audio_3.iterrows():
      row_array = np.array(row)
      rows_as_arrays.append(create_padded_array(row_array).astype('float32'))

  return rows_as_arrays

# 동영상 0.1초당 특징 추출
def faceFT(file):

    if file.endswith(".mp4") or file.endswith(".m2ts"):
        # 표정 특징 벡터
        feature = facialFeatureExtraction.facialVideoFeature(file, 1)
        feature = np.array(feature)
        
        return feature

# 원하는 시간대의 특징 평균
def faceMeanFT(feature, start, end):
    if start < 1:
        start = 1
    if end > len(feature):
        end = len(feature)
        
    feature = feature[start - 1:end]
    feature = np.mean(feature, axis=0)
    
    return feature

if __name__ == "__main__":
    print("\n\n\n\n\nMain\n")
    parser = argparse.ArgumentParser(description="Emotion Analysis")
    parser.add_argument("--video", required=True, help="Path to the video file for analysis")
    args = parser.parse_args()
    
    video_path = args.video
    
    ### AWS 연결 ###    
    aws_access_key_id = 'AWS 키(config.json이나 .env를 이용하여 보안 유지하기)'
    aws_secret_access_key = 'AWS 시크릿 키'
    bucket_name = 's3 bucket name'
    
    # AWS S3 클라이언트 생성
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    try:
        s3.download_file(bucket_name, video_path, video_path)
        print("Download Successful")
    except Exception as e:
        print(f"failed to download video file : {str(e)}")
    
    # 동영상에서 텍스트 추출
    try:
        text_df = video_to_text(video_path)
        print("Text extraction Successful")
    except Exception as e:
        print(f"Failed to extract text from video : {str(e)}")
        
    # 텍스트 특징 추출
    try:
        text_features = add_kobert_embeddings(text_df, 'text')
        print("Text features exctraction Successful")
    except Exception as e:
        print(f"Failed to extract text features : {str(e)}")
    
    # 동영상 전체 이미지 특징 추출
    try:
        total_image_features = faceFT(video_path)
        print("Total image features extraction Successful")
    except Exception as e:
        print(f"Failed to extract total image features : {str(e)}")
    
    audios = []
    image_features = []
    for i in range(len(text_df)):
        start = text_df['start'][i]
        end = text_df['end'][i]
        
        # 특정 시간대 오디오 추출
        audio = extract_video_audio(video_path, start, end)
        audios.append(audio)
            
        # 특정 시간대 이미지 특징 평균 추출
        image_feat = faceMeanFT(total_image_features, int(start*10), int(end*10))
        image_features.append(image_feat.flatten())
           
    try:
        # 오디오 특징 추출
        audio_features = pd.DataFrame(columns=['feature'])
        i = 0
        for audio in audios:
            audio_features.loc[i] = [audio_feature_extraction(audio)]
            i += 1
            
        # 오디오 특징 정규화
        audio_features = normalize_audio_features(audio_features)
        print("Audio features extraction Successful")
    except Exception as e:
        print(f"Failed to extract audio features : {str(e)}")
    
    # 동영상 파일 삭제
    os.remove(video_path)
    
    ### 감정 분석 ###
    input_dim_text = 768
    input_dim_image = 400
    input_dim_audio = 87
    num_classes = 7
    
    try:
        model = MultimodalClassifier(input_dim_text, input_dim_image, input_dim_audio, num_classes)
        model.load_state_dict(torch.load("model/text_image_audio_model.pth", map_location=torch.device('cpu')))
        model.eval()
        print("Model load Successful")
    except Exception as e:
        print(f"Failed to load model : {str(e)}")
    
    emotion_labels = ["Happy", "Surprise", "Expressionless", "Fear", "Aversion", "Angry", "Sad"]

    happy = []
    surprise = []
    expressionless = []
    fear = []
    aversion = []
    angry = []
    sad = []
    
    try:
        for i in range(len(text_features)):
              # 데이터를 텐서로 변환하여 모델에 입력
              text_input = torch.tensor(text_features[i]).unsqueeze(0)
              image_input = torch.tensor(image_features[i]).unsqueeze(0)
              audio_input = torch.tensor(audio_features[i]).unsqueeze(0)
        
              # 모델 예측 수행
              with torch.no_grad():
                  output = model(text_input, image_input, audio_input)
                  probabilities = F.softmax(output, dim=1).squeeze().tolist()
        
              # 예측 결과를 DataFrame에 저장
              for label, prob in zip(emotion_labels, probabilities):
                  percentage = round(prob * 100, 1)
                  print(f"{label}: {percentage}%")
                  if label == "Happy":
                    happy.append(percentage)
                  elif label == "Surprise":
                    surprise.append(percentage)
                  elif label == "Expressionless":
                    expressionless.append(percentage)
                  elif label == "Fear":
                    fear.append(percentage)
                  elif label == "Aversion":
                    aversion.append(percentage)
                  elif label == "Angry":
                    angry.append(percentage)
                  else:
                    sad.append(percentage)
        
        text_df['Happy'] = happy
        text_df['Surprise'] = surprise
        text_df['Expressionless'] = expressionless
        text_df['Fear'] = fear
        text_df['Aversion'] = aversion
        text_df['Angry'] = angry
        text_df['Sad'] = sad
        print("Emotion analysis Successfull")
    except Exception as e:
        print(f"Emotion analysis failed : {str(e)}")
            
    
    # CSV 파일
    csv_content = text_df.to_csv(index=False)
    
    # 결과 파일을 S3 버킷에 업로드
    try:
        file_name = os.path.basename(video_path)
        file_name, file_extension = os.path.splitext(file_name)
        s3.put_object(Bucket=bucket_name, Key=os.path.join("emotion_result", file_name + '_emotion_results.csv'), Body=csv_content)
        print("Successfully upload result to s3")
    except Exception as e:
        print(f"Failed to upload to s3 : {str(e)}")
