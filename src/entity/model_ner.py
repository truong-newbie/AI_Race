"""
NER Model Module

XLM-RoBERTa base model with NER classification head.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from src.entity.labels import NUM_LABELS, ID2LABEL


class XLMRobertaForNER(PreTrainedModel):
    """XLM-RoBERTa model for Named Entity Recognition.

    Adds a token classification head on top of XLM-RoBERTa.
    """

    base_model_prefix = "xlm-roberta"

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = NUM_LABELS,
        hidden_dropout_prob: float = 0.1,
        classifier_dropout: float = 0.1,
        ignore_index: int = -100,
    ):
        """Initialize NER model.

        Args:
            model_name: Pretrained model name or path
            num_labels: Number of NER labels
            hidden_dropout_prob: Dropout probability for hidden layers
            classifier_dropout: Dropout probability for classifier
            ignore_index: Label index to ignore in loss
        """
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)

        self.num_labels = num_labels
        self.ignore_index = ignore_index

        # Load pretrained model
        self.xlm_roberta = AutoModel.from_pretrained(
            model_name,
            config=config,
        )

        # Classifier dropout
        self.dropout = nn.Dropout(classifier_dropout)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        """Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (not used for XLM-R)
            labels: Token labels for computing loss

        Returns:
            TokenClassifierOutput with loss, logits, hidden_states, attentions
        """
        # Get transformer outputs
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # Get sequence output
        sequence_output = outputs.last_hidden_state

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # Get logits
        logits = self.classifier(sequence_output)

        # Compute loss
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Model logits (batch_size, seq_len, num_labels)
            labels: Token labels (batch_size, seq_len)

        Returns:
            Scalar loss
        """
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
        )

        # Flatten tensors
        # logits: (batch * seq_len, num_labels)
        # labels: (batch * seq_len)
        batch_size, seq_len, num_labels = logits.shape
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        loss = loss_fct(logits_flat, labels_flat)
        return loss

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get predictions without computing loss.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Predicted label IDs
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions

    def get_label_ids(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get label IDs from logits.

        Args:
            logits: Model logits
            attention_mask: Attention mask

        Returns:
            Predicted label IDs
        """
        predictions = torch.argmax(logits, dim=-1)

        # Mask padding tokens
        predictions = predictions.masked_fill(attention_mask == 0, self.ignore_index)

        return predictions


def create_ner_model(
    model_name: str = "xlm-roberta-base",
    num_labels: int = NUM_LABELS,
    **kwargs,
) -> XLMRobertaForNER:
    """Create NER model.

    Args:
        model_name: Pretrained model name
        num_labels: Number of NER labels
        **kwargs: Additional arguments for model

    Returns:
        Initialized NER model
    """
    model = XLMRobertaForNER(
        model_name=model_name,
        num_labels=num_labels,
        **kwargs,
    )
    return model


def load_ner_model(
    model_path: str,
    model_name: str = "xlm-roberta-base",
    num_labels: int = NUM_LABELS,
) -> XLMRobertaForNER:
    """Load NER model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        model_name: Base model name (needed for config)
        num_labels: Number of labels

    Returns:
        Loaded model
    """
    model = create_ner_model(
        model_name=model_name,
        num_labels=num_labels,
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model
