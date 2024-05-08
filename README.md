# unlock-feeling-emotion-multi-modal-AI
- 텍스트(text), 이미지(image), 음성(audio)을 이용하여 감정 분석하는 AI
- 감정 종류 : 행복(Happy), 놀람(Surprise), 중립(Expressionless), 공포(Fear), 혐오(Aversion), 분노(Angry), 슬픔(Sad)
- validation 정확도(validation Accuracy) : 75.41%
- 텍스트, 이미지, 음성의 특징을 따로 먼저 추출하여 저장한 다음, 추출한 특징을 input값으로 넣어 만든 multi modal AI입니다.\
훈련 속도를 절감하고 메모리 절약을 위해 이와같이 진행하였습니다.
- 해당 AI는 2023년 덕성여자대학교 컴퓨터공학 졸업 프로젝트를 위해 만들어졌습니다.
## 만든 이
- 조선아(Cho Suna) : 이미지 특징 추출, 멀티 모달, 파인튜닝 작업
- 이예원(Lee Yewon) : 텍스트와 음성 특징 추출, 멀티 모달
## 사용법
### feature_extraction
해당 폴더는 특징 추출했던 폴더입니다. 본래의 해당 폴더가 없었고 특징 추출했던 파일과 모델 훈련한 파일이 섞여있어 따로 폴더를 나눴을 뿐, 해당 폴더에 중복되어있는 model폴더와 facialFeatureExtraction.py파일은 그 밖에 있는 것들과 같은 것입니다.
#### facialFeatureAI.ipynb
- 표정 특징을 추출하는 AI를 만든 파일입니다. 단순히 특징을 추출할 뿐이라 훈련과정이 없습니다.
- Tensorflow, Keras를 사용한 AI라 구조 파일과 가중치 파일, 2개로 나옵니다.
- 완성된 모델 구조 파일 : model/featureAI/featureAI.json
- 완성된 모델 가중치 파일 : model/featureAI/weights.hdf5
#### facialFeatureExtraction.py
- 표정 특징을 추출해주는 파일입니다.
- model/featureAI에 있는 표정 특징 추출 AI를 사용하여 이미지 폴더 경로나 동영상의 경로를 넣으면 특징을 추출해 반환해줍니다.
#### textAudioFeaturesSave.ipynb
- 텍스트와 음성 특징을 추출하여 저장한 파일입니다.
#### threeFeaturesSave.ipynb
- 텍스트, 이미지, 음성 특징을 추출하여 저장한 파일입니다.
---
### text_audio_train.ipynb
- 텍스트, 음성 특징을 이용하여 감정 분석하는 AI를 훈련한 파일입니다. (Pytorch)
- 완성된 모델 가중치 파일 : model/text_audio_model_53.pth
- 훈련 데이터셋 개수 : 260171개
- validation 정확도 : 52.21%
### additional_train_addImageLayer.ipynb
- 텍스트, 음성에서 이미지 레이어를 추가하여 AI를 파인튜닝한 파일입니다. (Pytorch)
- 완성된 모델 가중치 파일 : text_image_audio_model.pth
- 훈련 데이터셋 개수 : 10350개
- validation 정확도 : 75.41%

### final_emotion_analysis.py
- 완성된 AI를 이용하여 동영상을 넣으면 감정분석해주는 파일입니다.
- 해당 파일은 AWS ec2에서 사용됐던 파일로 로컬에서 사용하기엔 적절하지 않습니다. AWS와 관련된 코드는 모두 지워주시고 아래와 같이 수정해주세요. 결과값인 text_df를 print를 하거나 csv로 변환하여 로컬에서 저장 후 확인하시면 됩니다.
```python
video_path = args.video
```
- 해당 파일을 실행하는 방법은 cmd에서 아래와 같이 작성하면 됩니다.
```bash
python final_emotion_analysis.py --video=동영상_경로
```

## Data 출처
- AIHub 감성 및 발화 스타일별 음성합성 데이터 (링크 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=466)
- AIHub 감정 분류용 데이터셋 (링크 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=259)
