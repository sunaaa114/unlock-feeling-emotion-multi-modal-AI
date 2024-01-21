import torch
import torch.nn as nn

class TextAudioMultimodalClassifier(nn.Module):
    def __init__(self, input_dim_text, input_dim_audio, num_classes):
        super(TextAudioMultimodalClassifier, self).__init__()
        self.text_layer = nn.Linear(input_dim_text, 512)
        self.audio_layer = nn.Linear(input_dim_audio, 512)
        
        self.fc = nn.Linear(1024, num_classes)  # 텍스트와 오디오 특징을 합친 후 분류
        
    def forward(self, text, audio):
        text_features = self.text_layer(text)
        audio_features = self.audio_layer(audio)
        
        combined_features = torch.cat((text_features, audio_features), dim=1)
        output = self.fc(combined_features)
        
        return output

