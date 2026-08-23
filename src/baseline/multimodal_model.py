import torch
import torch.nn as nn

from transformers import AutoModel

from src.baseline.encoder import ResNet50Encoder
from src.baseline.aggregator import MeanImageAggregator


class FullMultimodalBaseline(nn.Module):
    """
    Full supervised multimodal baseline.

    Modalities:
        - Photograph
        - Radiograph
        - Clinical text

    Architecture:

        Photograph
            ResNet50
            Mean pooling
            2048

        Radiograph
            ResNet50
            Mean pooling
            2048

        Text
            DistilBERT
            CLS embedding
            768

        Concatenate
            4864

        MLP classifier
            4864 -> 512 -> labels
    """


    def __init__(
        self,
        text_model_name="distilbert-base-uncased",
        num_labels=6,
        pretrained=True,
        freeze_image_encoder=False,
    ):

        super().__init__()


        # -------------------------
        # Image encoders
        # -------------------------

        photograph_encoder = ResNet50Encoder(
            pretrained=pretrained,
            freeze=freeze_image_encoder,
        )


        radiograph_encoder = ResNet50Encoder(
            pretrained=pretrained,
            freeze=freeze_image_encoder,
        )


        self.photograph_aggregator = MeanImageAggregator(
            photograph_encoder
        )


        self.radiograph_aggregator = MeanImageAggregator(
            radiograph_encoder
        )


        # -------------------------
        # Text encoder
        # -------------------------

        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )


        text_hidden_size = (
            self.text_encoder.config.hidden_size
        )


        # -------------------------
        # Fusion classifier
        # -------------------------

        fusion_dim = (
            2048
            +
            2048
            +
            text_hidden_size
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                512,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                512,
                num_labels,
            ),
        )



    def forward(
        self,
        images,
        radiographs,
        input_ids,
        attention_mask,
    ):


        # -------------------------
        # Photograph branch
        # -------------------------

        image_features = []


        for sample_images in images:

            feature = self.photograph_aggregator(
                sample_images
            )

            image_features.append(
                feature
            )


        image_features = torch.stack(
            image_features
        )



        # -------------------------
        # Radiograph branch
        # -------------------------

        radiograph_features = []


        for sample_radiographs in radiographs:

            feature = self.radiograph_aggregator(
                sample_radiographs
            )

            radiograph_features.append(
                feature
            )


        radiograph_features = torch.stack(
            radiograph_features
        )



        # -------------------------
        # Text branch
        # -------------------------

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


        text_features = (
            text_outputs
            .last_hidden_state[:, 0]
        )



        # -------------------------
        # Fusion
        # -------------------------

        fused_features = torch.cat(
            [
                image_features,
                radiograph_features,
                text_features,
            ],
            dim=1,
        )


        logits = self.classifier(
            fused_features
        )


        return logits