# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""
import math

import torch
from utils import sde_lib
import numpy as np

import models.Encoders as Encoders
import models.Decoders as Decoders


_MODELS = {}


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    # assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float() * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
        'beta_min': beta_start * (num_diffusion_timesteps - 1),
        'beta_max': beta_end * (num_diffusion_timesteps - 1),
        'num_diffusion_timesteps': num_diffusion_timesteps
    }


def create_model(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    return score_model


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model_fn(x, labels)
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def extract_encoder_params(config):
    """Extract encoder parameters from config object."""
    model_config = config.model
    
    # Base encoder parameters
    encoder_params = {
        'max_feat_num': model_config.max_feat_num,
        'hidden_dim': model_config.hidden_dim,
        'dim': model_config.dim
    }
    
    # Add specific parameters based on encoder type
    encoder_type = model_config.encoder
    if encoder_type in ['GCN', 'HGCN']:
        encoder_params.update({
            'enc_layers': model_config.enc_layers,
            'layer_type': model_config.layer_type,
            'dropout': model_config.dropout,
            'edge_dim': model_config.edge_dim,
            'normalization_factor': model_config.normalization_factor,
            'aggregation_method': model_config.aggregation_method,
            'msg_transform': model_config.msg_transform
        })
        
        if encoder_type == 'HGCN':
            encoder_params.update({
                'manifold': model_config.manifold,
                'c': model_config.c,
                'learnable_c': model_config.learnable_c,
                'sum_transform': model_config.sum_transform,
                'use_norm': model_config.use_norm
            })
    
    return encoder_params

def extract_decoder_params(config):
    """Extract decoder parameters from config object."""
    model_config = config.model
    
    # Base decoder parameters
    decoder_params = {
        'max_feat_num': model_config.max_feat_num,
        'hidden_dim': model_config.hidden_dim
    }
    
    # Add specific parameters based on decoder type
    decoder_type = model_config.decoder
    if decoder_type in ['GCN', 'HGCN']:
        decoder_params.update({
            'dec_layers': model_config.dec_layers,
            'layer_type': model_config.layer_type,
            'dropout': model_config.dropout,
            'edge_dim': model_config.edge_dim,
            'normalization_factor': model_config.normalization_factor,
            'aggregation_method': model_config.aggregation_method,
            'msg_transform': model_config.msg_transform
        })
        
        if decoder_type == 'HGCN':
            decoder_params.update({
                'manifold': model_config.manifold,
                'c': model_config.c,
                'learnable_c': model_config.learnable_c,
                'sum_transform': model_config.sum_transform,
                'use_norm': model_config.use_norm,
                'use_centroid': getattr(model_config, 'use_centroid', False)
            })
    elif decoder_type == 'CentroidDecoder':
        decoder_params.update({
            'dim': model_config.dim,
            'manifold': None,  # Will be set during instantiation
            'dropout': model_config.dropout
        })
    
    return decoder_params

def extract_hvae_params(config):
    """Extract HVAE parameters from config object."""
    model_config = config.model
    train_config = config.train
    
    return {
        'device': config.device,
        'encoder_class': getattr(Encoders, model_config.encoder),
        'encoder_params': extract_encoder_params(config),
        'decoder_class': getattr(Decoders, model_config.decoder),
        'decoder_params': extract_decoder_params(config),
        'manifold_type': model_config.manifold,
        'train_class_num': train_config.class_num,
        'dim': model_config.dim,
        'pred_node_class': getattr(model_config, 'pred_node_class', True),
        'use_kl_loss': getattr(model_config, 'use_kl_loss', True),
        'use_base_proto_loss': getattr(model_config, 'use_base_proto_loss', True),
        'use_sep_proto_loss': getattr(model_config, 'use_sep_proto_loss', True),
        'pred_edge': getattr(model_config, 'pred_edge', False),
        'pred_graph_class': getattr(model_config, 'pred_graph_class', False),
        'classifier_dropout': getattr(model_config, 'classifier_dropout', 0.0),
        'classifier_bias': getattr(model_config, 'classifier_bias', True)
    }

def create_encoder_from_config(config):
    """Create encoder instance from config."""
    encoder_class = getattr(Encoders, config.model.encoder)
    encoder_params = extract_encoder_params(config)
    return encoder_class(**encoder_params)

def create_decoder_from_config(config):
    """Create decoder instance from config."""
    decoder_class = getattr(Decoders, config.model.decoder)
    decoder_params = extract_decoder_params(config)
    return decoder_class(**decoder_params)

def create_hvae_from_config(config):
    """Create HVAE instance from config."""
    from models.HVAE import HVAE
    hvae_params = extract_hvae_params(config)
    return HVAE(**hvae_params)