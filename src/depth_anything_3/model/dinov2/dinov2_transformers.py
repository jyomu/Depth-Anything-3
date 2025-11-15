# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DinoV2 backbone implementation using Hugging Face Transformers.

This module provides a wrapper around Hugging Face's official DinoV2 models
to maintain API compatibility with the custom implementation.
"""

from typing import List, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel, Dinov2Model


class DinoV2Transformers(nn.Module):
    """
    DinoV2 backbone using Hugging Face Transformers library.
    
    This wrapper provides the same interface as the custom DinoV2 implementation
    but uses Hugging Face's official models under the hood.
    
    Args:
        name: Model size variant ('vits', 'vitb', 'vitl', 'vitg')
        out_layers: List of layer indices to extract features from
        alt_start: Starting layer for alternative processing (compatibility parameter)
        qknorm_start: Starting layer for QK normalization (compatibility parameter)
        rope_start: Starting layer for RoPE (compatibility parameter)
        cat_token: Whether to concatenate token information (compatibility parameter)
    """
    
    # Mapping from custom names to Hugging Face model IDs
    MODEL_MAP = {
        "vits": "facebook/dinov2-small",
        "vitb": "facebook/dinov2-base",
        "vitl": "facebook/dinov2-large",
        "vitg": "facebook/dinov2-giant",
    }
    
    def __init__(
        self,
        name: str,
        out_layers: List[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert name in self.MODEL_MAP, f"Model {name} not supported. Choose from {list(self.MODEL_MAP.keys())}"
        
        self.name = name
        self.out_layers = out_layers
        self.cat_token = cat_token
        
        # Load the Hugging Face model
        model_id = self.MODEL_MAP[name]
        self.model = AutoModel.from_pretrained(
            model_id,
            output_hidden_states=True,
        )
        
        # Get model configuration
        self.embed_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        
        # Validate out_layers
        for layer_idx in out_layers:
            assert 0 <= layer_idx < self.num_layers, \
                f"Layer index {layer_idx} out of range [0, {self.num_layers})"
        
        # Note: alt_start, qknorm_start, rope_start are custom parameters
        # that don't directly map to transformers models. They are kept for
        # API compatibility but may not have the same effect.
        
    def forward(
        self,
        x: torch.Tensor,
        cam_token: torch.Tensor = None,
        export_feat_layers: List[int] = [],
        **kwargs,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], dict]:
        """
        Forward pass through the DinoV2 model.
        
        Args:
            x: Input images (B*N, 3, H, W)
            cam_token: Optional camera token (for compatibility)
            export_feat_layers: Additional layers to export features from
            
        Returns:
            Tuple of:
            - List of (camera_token, features) tuples for each output layer
            - Dictionary of auxiliary features for export layers
        """
        # Run the model
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        
        # Extract hidden states (includes embedding + all layer outputs)
        # hidden_states[0] is the embedding, hidden_states[i+1] is layer i output
        hidden_states = outputs.hidden_states
        
        # Process output layers
        results = []
        for layer_idx in self.out_layers:
            # Get the hidden state for this layer (+1 because hidden_states[0] is embedding)
            layer_output = hidden_states[layer_idx + 1]
            
            # Extract CLS token and patch tokens
            # Shape: (B*N, num_patches + 1, embed_dim)
            cls_token = layer_output[:, 0:1, :]  # (B*N, 1, embed_dim)
            patch_tokens = layer_output[:, 1:, :]  # (B*N, num_patches, embed_dim)
            
            # If cam_token is provided, use it; otherwise use CLS token
            if cam_token is not None and self.cat_token:
                # Concatenate cam_token with patch tokens
                # This maintains compatibility with the original implementation
                camera_token = cam_token if cam_token.dim() == 3 else cam_token.unsqueeze(1)
            else:
                camera_token = cls_token
            
            results.append((camera_token, patch_tokens))
        
        # Process auxiliary features for export layers
        aux_features = {}
        for layer_idx in export_feat_layers:
            if 0 <= layer_idx < len(hidden_states) - 1:
                layer_output = hidden_states[layer_idx + 1]
                aux_features[layer_idx] = layer_output[:, 1:, :]  # Only patch tokens
        
        return results, aux_features
