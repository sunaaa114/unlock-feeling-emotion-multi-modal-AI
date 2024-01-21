import torch
import torch.nn as nn
from TextAudioMultimodalClassifier import TextAudioMultimodalClassifier

class MultimodalClassifier(nn.Module):
    def __init__(self, input_dim_text, input_dim_image, input_dim_audio, num_classes):
        super(MultimodalClassifier, self).__init__()
        self.text_layer = nn.Linear(input_dim_text, 512)
        self.image_layer = nn.Linear(input_dim_image, 512)
        self.audio_layer = nn.Linear(input_dim_audio, 512)
        
        self.pretrained_model = TextAudioMultimodalClassifier(input_dim_text, input_dim_audio, num_classes)
        
        self.fc = nn.Linear(1543, num_classes)

    def forward(self, text, image, audio):
        text_features = self.text_layer(text)
        image_features = self.image_layer(image)
        audio_features = self.audio_layer(audio)
        
        combined_features = torch.cat((text_features, image_features, audio_features), dim=1)
        pretrained_output = self.pretrained_model(text, audio)  # 기존 텍스트와 오디오 특징을 입력으로 사용
        combined_output = combined_features + pretrained_output

        output = self.fc(combined_output)

        return output
