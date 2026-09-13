"""One modality-aware architecture for all corrected supervised scenarios."""
import torch
import torch.nn as nn
from transformers import AutoModel
from src.baseline.encoder import ResNet50Encoder
from src.baseline.aggregator import MeanImageAggregator

IMAGE_DIM = 2048
HIDDEN_DIM = 512
SMALL_FUSION_HIDDEN_DIM = 256

class FinalSupervisedBaseline(nn.Module):
    def __init__(self, modalities, text_model_name, num_labels, pretrained_image=True, freeze_image_encoder=True, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.modalities = tuple(modalities)
        if "photograph" in self.modalities:
            self.photograph_aggregator = MeanImageAggregator(ResNet50Encoder(pretrained_image, freeze_image_encoder))
        if "radiograph" in self.modalities:
            self.radiograph_aggregator = MeanImageAggregator(ResNet50Encoder(pretrained_image, freeze_image_encoder))
        text_dim = 0
        if "text" in self.modalities:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            text_dim = self.text_encoder.config.hidden_size
        input_dim = (IMAGE_DIM if "photograph" in self.modalities else 0) + (IMAGE_DIM if "radiograph" in self.modalities else 0) + text_dim
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, num_labels))

    def _images(self, batch_images, aggregator):
        return torch.stack([aggregator(x) for x in batch_images])

    def forward(self, images=None, radiographs=None, input_ids=None, attention_mask=None):
        features = []
        if "photograph" in self.modalities:
            features.append(self._images(images, self.photograph_aggregator))
        if "radiograph" in self.modalities:
            features.append(self._images(radiographs, self.radiograph_aggregator))
        if "text" in self.modalities:
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            features.append(outputs.last_hidden_state[:, 0])
        return self.classifier(torch.cat(features, dim=1))
