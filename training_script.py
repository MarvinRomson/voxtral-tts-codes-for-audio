#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Training script for voice cloning/conversion using Voxtral TTS decoder.

This script trains learnable audio codes [T, 37] that are decoded to waveforms:
- Semantic (1 code): discrete selection via softmax + STE + embedding lookup
- Acoustic (36 codes): continuous FSQ values with tanh + STE

Architecture:
    codes [T, 37] → quantizer.decode() → embeddings [1, 292, T] → decoder → waveform

Training objective:
    - Minimize reconstruction loss between generated and reference audio
"""

"""
RUN:

!download weights to the voxtral-tts-weights (mistralai/Voxtral-4B-TTS-2603) folder 

python training_script.py \
  --reference-audio casual_female_clean.wav \
  --model-path voxtral-tts-weights \
  --num-epochs 15000 \
  --device mps --learning-rate 0.1 --reconstruction-weight 0.5 --speaker-weight 0.5 --num-frames 60
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator
from speechbrain.inference.classifiers import EncoderClassifier
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

# Local imports
from voice_to_audio import (
    load_decoder_and_embtable,
    save_wav,
    SAMPLING_RATE,
    FRAME_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model paths
    model_path: str = "../voxtral-tts-weights"
    reference_audio: str = "casual_female.wav"
    
    # Training parameters
    num_frames: int = 100  # Number of audio frames
    learning_rate: float = 1e-3
    num_epochs: int = 100
    max_steps: Optional[int] = None
    warmup_epochs: int = 10  # LR warmup to escape local minima
    
    # Loss weights
    reconstruction_weight: float = 1.0
    speaker_embedding_weight: float = 0.5
    perceptual_weight: float = 1.0 #0.3  # Multi-resolution STFT loss for better convergence
    mel_weight: float = 1.0 #0.1  # Mel spectrogram loss (full spectral detail)
    mfcc_weight: float = 1.0 #0.05  # MFCC loss (decorrelated cepstral features)
    
    # Optimization
    optimizer: str = "adam"
    grad_clip: float = 1.0
    use_cosine_restarts: bool = False  # Cosine annealing with warm restarts (helps escape local minima!). But it is better to avoid it according to experiments
    restart_period: int = 50  # Restart LR every 50 epochs
    
    # Gumbel-Softmax temperature for semantic codes
    temperature: float = 2.0  # Higher initial temperature for exploration
    temperature_decay: float = 0.99
    min_temperature: float = 0.3
    
    # Initialization
    init_from_codes: bool = True  # Initialize from actual encoded codes
    init_noise_scale: float = 0.1  # Add noise to initialization
    
    # Device and dtype
    device: str = "cpu"
    dtype: str = "float32"
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10
    log_every: int = 1
    sample_every: int = 10
    
    # Speaker model
    speaker_model_name: str = "speechbrain/spkrec-ecapa-voxceleb" # also tested a smaller - "speechbrain/spkrec-xvect-voxceleb"
    speaker_device: str = "cpu"


# ===========================================================================
# Learnable Codes Model
# ===========================================================================

class LearnableCodesModel(nn.Module):
    """Learnable audio codes [T, 37] with gradient flow.
    
    Architecture:
        - Semantic (code 0): logits [T, 8194] → Gumbel-Softmax → code index
        - Acoustic (codes 1-36): continuous [T, 36] → tanh/FSQ → quantized values
    
    Gradient flow via:
        - Semantic: STE (hard selection forward, soft probabilities backward)
        - Acoustic: STE (quantized forward, continuous backward)
    """
    
    def __init__(
        self,
        num_frames: int,
        semantic_vocab: int = 8192,  # Actual codebook size (no special tokens)
        acoustic_levels: int = 21,    # Actual codebook size (no special tokens)
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.semantic_vocab = semantic_vocab
        self.acoustic_levels = acoustic_levels
        self.temperature = temperature
        
        # Semantic: learnable logits for discrete code selection
        self.semantic_logits = nn.Parameter(
            torch.randn(num_frames, semantic_vocab)
        )
        
        # Acoustic: learnable continuous values (initialize larger for better gradient flow)
        self.acoustic_values = nn.Parameter(
            torch.randn(num_frames, 36) * 0.1  # Larger init so tanh doesn't saturate at 0
        )
        
    def forward(self, temperature: Optional[float] = None, tokenizer=None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            temperature: Gumbel-Softmax temperature
            tokenizer: If provided, returns embeddings; otherwise returns codes
            
        Returns:
            codes: [T, 37] quantized codes (integer for forward, continuous gradient)
            embeddings: [1, 292, T] if tokenizer provided, else None
        """
        if temperature is None:
            temperature = self.temperature
        
        T = self.num_frames
        
        # ===== Semantic part: Gumbel-Softmax + STE =====
        if self.training:
            # Add Gumbel noise for exploration
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.semantic_logits) + 1e-10) + 1e-10)
            logits_with_noise = (self.semantic_logits + gumbel_noise) / temperature
        else:
            logits_with_noise = self.semantic_logits / temperature
        
        # Soft probabilities (for gradients)
        probs = F.softmax(logits_with_noise, dim=-1)  # [T, 8192]
        
        # Hard selection (for forward pass)
        hard_codes = probs.argmax(dim=-1)  # [T] integer indices in range 0-8191
        
        # For semantic, we use the embedding table directly (no padding needed!)
        if tokenizer is not None:
            # Get semantic embedding table - exactly 8192 entries, matches our vocab
            sem_embedding = tokenizer.quantizer.semantic_codebook.embedding  # [8192, 256]
            
            # Soft embedding (weighted sum for gradients)
            soft_emb = torch.matmul(probs, sem_embedding)  # [T, 256]
            
            # Hard embedding (discrete lookup for forward)
            hard_emb = F.embedding(hard_codes, sem_embedding)  # [T, 256]
            
            # STE: forward uses hard, backward uses soft
            semantic_emb = soft_emb + (hard_emb - soft_emb).detach()  # [T, 256]
        else:
            # Just return codes without embedding
            semantic_emb = None
        
        # For compatibility with quantizer.decode(), use integer codes
        semantic_codes = hard_codes.float()  # [T]

        logger.info(f"Semantic codes: {hard_codes}")
        
        # ===== Acoustic part: FSQ with tanh + STE =====
        # Apply tanh normalization
        acoustic_normalized = torch.tanh(self.acoustic_values)  # [T, 36] in [-1, 1]
        
        # Quantize to discrete levels
        acoustic_scaled = ((acoustic_normalized + 1) / 2) * (self.acoustic_levels - 1)
        acoustic_quantized = acoustic_scaled.round()
        
        # STE: forward uses quantized, backward uses continuous
        acoustic_codes = acoustic_scaled + (acoustic_quantized - acoustic_scaled).detach()  # [T, 36]

        num_extreme_acoustic = torch.sum(acoustic_codes==0)+torch.sum(acoustic_codes==21)

        logger.info(f"Number of extreme acoustic tokens: {num_extreme_acoustic}")
        
        # ===== Combine codes: [T, 1] + [T, 36] = [T, 37] =====
        codes = torch.cat([semantic_codes.unsqueeze(1), acoustic_codes], dim=1)  # [T, 37]
        
        # ===== If tokenizer provided, build full embeddings =====
        if tokenizer is not None:
            # Acoustic decoding: rescale codes to [-1, 1]
            acoustic_emb = ((acoustic_codes * 2 / (self.acoustic_levels - 1)) - 1)  # [T, 36]
            
            # Combine: [T, 256] + [T, 36] = [T, 292]
            full_emb = torch.cat([semantic_emb, acoustic_emb], dim=1)  # [T, 292]
            
            # Reshape to [1, 292, T] for decoder
            embeddings = full_emb.unsqueeze(0).transpose(1, 2)  # [1, 292, T]
            return codes, embeddings
        
        return codes, None
    
    def get_discrete_codes(self) -> torch.Tensor:
        """Get purely discrete codes (for saving/inspection)."""
        semantic_codes = self.semantic_logits.argmax(dim=-1)  # [T]
        
        acoustic_normalized = torch.tanh(self.acoustic_values)
        acoustic_scaled = ((acoustic_normalized + 1) / 2) * (self.acoustic_levels - 1)
        acoustic_codes = acoustic_scaled.round().long()  # [T, 36]
        
        codes = torch.cat([semantic_codes.unsqueeze(1), acoustic_codes], dim=1)  # [T, 37]
        return codes


# ===========================================================================
# Training utilities
# ===========================================================================

class SpeakerEmbeddingExtractor:
    """Extract speaker embeddings using SpeechBrain."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": device},
            savedir=os.path.join("/tmp", model_name.replace("/", "_")),
        )
        self.device = device
        # Pre-create resampler (it's differentiable and preserves gradients)
        self.resampler_24_to_16 = torchaudio.transforms.Resample(24000, 16000)
        
    def extract(self, waveform: torch.Tensor, sample_rate: int = 16000, requires_grad: bool = False) -> torch.Tensor:
        """Extract speaker embedding from waveform.
        
        Args:
            waveform: audio waveform tensor
            sample_rate: sample rate of the audio
            requires_grad: if True, compute with gradients; if False, use no_grad
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample if needed (resampler is differentiable, gradients flow through)
        if sample_rate == 24000:
            waveform = self.resampler_24_to_16(waveform)
        elif sample_rate != 16000:
            # Fallback for other sample rates
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        if requires_grad:
            # Compute WITH gradients for training
            embeddings = self.model.encode_batch(waveform)
            embeddings = F.normalize(embeddings, dim=2)
        else:
            # Compute without gradients for inference/target
            with torch.no_grad():
                embeddings = self.model.encode_batch(waveform)
                embeddings = F.normalize(embeddings, dim=2)
        
        return embeddings


def load_reference_audio(
    path: str,
    target_duration: Optional[float] = None,
) -> tuple[torch.Tensor, int]:
    """Load reference audio file."""
    waveform, sample_rate = torchaudio.load(path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
    else:
        waveform = waveform.squeeze(0)
    
    if target_duration is not None:
        target_samples = int(target_duration * sample_rate)
        current_samples = waveform.shape[0]
        
        if current_samples > target_samples:
            waveform = waveform[:target_samples]
        elif current_samples < target_samples:
            padding = target_samples - current_samples
            waveform = F.pad(waveform, (0, padding))
    
    return waveform, sample_rate


def multi_resolution_stft_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT loss with 8 different FFT sizes.
    
    Uses FFT sizes: [2296, 1418, 876, 542, 334, 206, 126, 76]
    Similar to discriminator-based codec training.
    """
    # Ensure same length
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    
    # Multi-resolution FFT sizes (as suggested)
    fft_sizes = [2296, 1418, 876, 542, 334, 206, 126, 76]
    
    total_loss = 0.0
    for n_fft in fft_sizes:
        # Skip if signal too short
        if min_len < n_fft:
            continue
            
        try:
            window = torch.hann_window(n_fft, device=y_pred.device)
            hop_length = n_fft // 4
            
            stft_pred = torch.stft(y_pred, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, 
                                   window=window, return_complex=True)
            stft_true = torch.stft(y_true, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                                   window=window, return_complex=True)
            
            mag_pred = torch.abs(stft_pred)
            mag_true = torch.abs(stft_true)
            
            # Spectral convergence
            sc_loss = torch.norm(mag_true - mag_pred, p="fro") / (torch.norm(mag_true, p="fro") + 1e-8)
            
            # Log magnitude loss
            log_mag_loss = F.l1_loss(torch.log(mag_pred + 1e-5), torch.log(mag_true + 1e-5))
            
            total_loss += (sc_loss + log_mag_loss)
        except Exception as e:
            # Skip this resolution if it fails
            continue
    
    # Average over resolutions
    return total_loss / len(fft_sizes)


def mel_spectrogram_loss(y_pred: torch.Tensor, y_true: torch.Tensor, n_mels: int = 128) -> torch.Tensor:
    """Mel spectrogram loss for perceptual quality.
    
    Mel spectrograms preserve more information than MFCCs and are commonly used
    in neural vocoder training (HiFi-GAN, MelGAN, etc).
    """
    try:
        # Ensure same length
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
        
        # Compute Mel spectrograms
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLING_RATE,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
            power=1.0,  # Magnitude spectrogram (not power)
        ).to(y_pred.device)
        
        mel_pred = mel_transform(y_pred.unsqueeze(0))  # [1, n_mels, T]
        mel_true = mel_transform(y_true.unsqueeze(0))  # [1, n_mels, T]
        
        # L1 loss on log mel spectrograms (common in vocoder training)
        log_mel_pred = torch.log(mel_pred + 1e-5)
        log_mel_true = torch.log(mel_true + 1e-5)
        
        return F.l1_loss(log_mel_pred, log_mel_true)
    except Exception as e:
        # Fallback to zero if mel spectrogram computation fails
        return torch.tensor(0.0, device=y_pred.device)


def mfcc_loss(y_pred: torch.Tensor, y_true: torch.Tensor, n_mfcc: int = 40) -> torch.Tensor:
    """MFCC loss for perceptual quality.
    
    MFCCs capture decorrelated cepstral features via DCT, providing
    complementary gradient signal to mel spectrograms.
    """
    try:
        # Ensure same length
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
        
        # Compute MFCCs
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLING_RATE,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 2048,
                'hop_length': 512,
                'n_mels': 128,
            }
        ).to(y_pred.device)
        
        mfcc_pred = mfcc_transform(y_pred.unsqueeze(0))  # [1, n_mfcc, T]
        mfcc_true = mfcc_transform(y_true.unsqueeze(0))  # [1, n_mfcc, T]
        
        # L1 loss on MFCCs
        return F.l1_loss(mfcc_pred, mfcc_true)
    except Exception as e:
        # Fallback to zero if MFCC computation fails
        return torch.tensor(0.0, device=y_pred.device)


def get_lr_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """Calculate learning rate warmup factor."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return 1.0
    return (epoch + 1) / warmup_epochs


# ===========================================================================
# Training loop
# ===========================================================================

def train(config: TrainingConfig):
    """Main training function."""
    
    # Setup accelerator
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Training on device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = checkpoint_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    # Load reference audio
    logger.info(f"Loading reference audio: {config.reference_audio}")
    target_duration = config.num_frames/ FRAME_RATE # (config.num_frames - 1) / FRAME_RATE - before it includes special EOA
    ref_waveform, ref_sr = load_reference_audio(
        config.reference_audio,
        target_duration=target_duration,
    )
    logger.info(f"Reference audio: {len(ref_waveform)} samples @ {ref_sr}Hz ({len(ref_waveform)/ref_sr:.2f}s)")
    
    if ref_sr != SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(ref_sr, SAMPLING_RATE)
        ref_waveform = resampler(ref_waveform)
    
    ref_waveform = ref_waveform.to(device)
    
    # Truncate reference waveform to match expected duration based on num_frames
    # This ensures consistency between reconstruction loss and speaker embedding
    expected_duration = config.num_frames * 80 * 0.001  # num_frames * 80ms per frame
    expected_samples = int(expected_duration * SAMPLING_RATE)
    # print(expected_samples)
    ref_waveform = ref_waveform[:expected_samples]
    
    # Extract target speaker embedding (for monitoring only)
    speaker_extractor = None
    target_speaker_emb = None
    if config.speaker_embedding_weight > 0:
        logger.info("Extracting target speaker embedding...")
        speaker_extractor = SpeakerEmbeddingExtractor(
            config.speaker_model_name,
            device=config.speaker_device,
        )
        target_speaker_emb = speaker_extractor.extract(ref_waveform.cpu(), SAMPLING_RATE)
        target_speaker_emb = target_speaker_emb.to(device)
    
    # Load decoder and embedding table
    logger.info(f"Loading decoder from: {config.model_path}")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[config.dtype]
    
    tokenizer, emb_table = load_decoder_and_embtable(
        Path(config.model_path),
        dtype=dtype,
        device=device,
    )
    tokenizer.eval()  # Decoder is frozen
    logger.info("Decoder loaded successfully")
    
    # Initialize model with actual codebook sizes (no special tokens)
    model = LearnableCodesModel(
        num_frames=config.num_frames,
        semantic_vocab=8192,  # Actual semantic codebook size
        acoustic_levels=21,    # Actual acoustic codebook size
        temperature=config.temperature,
    ).to(device)
    
    logger.info(f"Model initialized with {config.num_frames} frames")
    logger.info(f"  Semantic logits: {model.semantic_logits.shape}")
    logger.info(f"  Acoustic values: {model.acoustic_values.shape}")
    
    # Setup optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Prepare with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # Setup learning rate scheduler (cosine annealing with optional warm restarts)
    if config.use_cosine_restarts:
        min_lr = config.learning_rate * 0.1 # reduced
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=config.restart_period,  # First restart after N epochs
            T_mult=1,  # Keep same period for all restarts
            eta_min=min_lr
        )
        logger.info(f"Using Cosine Annealing with Warm Restarts (T_0={config.restart_period})")
    else:
        min_lr = config.learning_rate * 0.01  # Further Decay introduced
        # Use max_steps if specified, otherwise use num_epochs for scheduler T_max
        T_max = config.max_steps if config.max_steps else config.num_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)
        logger.info(f"Using Cosine Annealing (T_max={T_max})")
    
    # Training loop
    logger.info("Starting training...")
    logger.info(f"LR schedule: {config.learning_rate:.6f} → {min_lr:.6f} (cosine)")
    current_temperature = config.temperature
    
    for epoch in range(config.num_epochs):
        model.train()
        
        # Forward pass - get embeddings with gradients via STE
        # Now the model directly computes embeddings using weighted sum of embedding table
        codes, embeddings = model(temperature=current_temperature, tokenizer=tokenizer)  # [T, 37], [1, 292, T]
        
        # Decode to waveform via frozen decoder
        # Gradients flow through embeddings even though decoder is frozen
        generated_waveform_batch = tokenizer._forward_decoder(embeddings)  # [1, 1, T_samples]
        generated_waveform = generated_waveform_batch.squeeze()  # [T_samples]
        
        # Compute losses
        min_len = min(len(generated_waveform), len(ref_waveform))
        print(len(generated_waveform), len(ref_waveform)) # initially they had different length here
        recon_loss = F.l1_loss(
            generated_waveform[:min_len],
            ref_waveform[:min_len],
        )
        
        # Multi-resolution STFT loss for better convergence
        perceptual_loss = torch.tensor(0.0, device=device)
        if config.perceptual_weight > 0:
            try:
                perceptual_loss = multi_resolution_stft_loss(
                    generated_waveform[:min_len],
                    ref_waveform[:min_len],
                )
            except Exception as e:
                logger.warning(f"Multi-resolution STFT loss failed: {e}")
        
        # Mel spectrogram loss for perceptual quality (full spectral detail)
        mel_loss_val = torch.tensor(0.0, device=device)
        if config.mel_weight > 0:
            try:
                mel_loss_val = mel_spectrogram_loss(
                    generated_waveform[:min_len],
                    ref_waveform[:min_len],
                )
            except Exception as e:
                logger.warning(f"Mel spectrogram loss failed: {e}")
        
        # MFCC loss for perceptual quality (decorrelated cepstral features)
        mfcc_loss_val = torch.tensor(0.0, device=device)
        if config.mfcc_weight > 0:
            try:
                mfcc_loss_val = mfcc_loss(
                    generated_waveform[:min_len],
                    ref_waveform[:min_len],
                )
            except Exception as e:
                logger.warning(f"MFCC loss failed: {e}")
        
        # Speaker embedding loss (WITH gradients!)
        speaker_loss = torch.tensor(0.0, device=device)
        if config.speaker_embedding_weight > 0 and speaker_extractor is not None:
            # Extract speaker embedding WITH gradients - gradients flow through speaker model
            # Move waveform to speaker model's device while preserving gradients
            waveform_for_speaker = generated_waveform.to(config.speaker_device)
            generated_speaker_emb = speaker_extractor.extract(
                waveform_for_speaker,  # No .detach() - gradients flow!
                SAMPLING_RATE,
                requires_grad=True,  # Enable gradients!
            ).to(device)
            speaker_loss = 1.0 - F.cosine_similarity(
                generated_speaker_emb.squeeze(),
                target_speaker_emb.squeeze(),
                dim=-1,
            ).mean()
        
        total_loss = (config.reconstruction_weight * recon_loss + 
                     config.speaker_embedding_weight * speaker_loss +
                     config.perceptual_weight * perceptual_loss +
                     config.mel_weight * mel_loss_val +
                     config.mfcc_weight * mfcc_loss_val)
        
        # Backward pass
        optimizer.zero_grad()
        accelerator.backward(total_loss)
        
        # Check if gradients are flowing (debug)
        if (epoch + 1) % config.log_every == 0:
            unwrapped = accelerator.unwrap_model(model)
            sem_grad = unwrapped.semantic_logits.grad
            aco_grad = unwrapped.acoustic_values.grad
            if sem_grad is not None:
                logger.info(f"  Semantic grad norm: {sem_grad.norm().item():.6f}")
            else:
                logger.warning("  No gradient for semantic_logits!")
            if aco_grad is not None:
                logger.info(f"  Acoustic grad norm: {aco_grad.norm().item():.6f}")
            else:
                logger.warning("  No gradient for acoustic_values!")
        
        # Gradient clipping
        if config.grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        
        # Update learning rate (cosine annealing)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update temperature
        current_temperature = max(
            config.min_temperature,
            current_temperature * config.temperature_decay
        )
        
        # Logging
        if (epoch + 1) % config.log_every == 0:
            log_msg = (
                f"Epoch {epoch+1}/{config.num_epochs} | "
                f"Loss: {total_loss.item():.10f} | "
                f"Recon: {recon_loss.item():.10f}"
            )
            if config.perceptual_weight > 0:
                log_msg += f" | Percept: {perceptual_loss.item():.6f}"
            if config.mel_weight > 0:
                log_msg += f" | Mel: {mel_loss_val.item():.6f}"
            if config.mfcc_weight > 0:
                log_msg += f" | MFCC: {mfcc_loss_val.item():.6f}"
            if config.speaker_embedding_weight > 0:
                log_msg += f" | Speaker: {speaker_loss.item():.4f}"
            log_msg += f" | LR: {current_lr:.6f} | Temp: {current_temperature:.3f}"
            logger.info(log_msg)
        
        # Save audio samples
        if (epoch + 1) % config.sample_every == 0 or epoch<10:
            sample_path = samples_dir / f"epoch_{epoch+1:04d}.wav"
            save_wav(sample_path, generated_waveform.detach().cpu())
            logger.info(f"Saved audio sample: {sample_path}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pt"
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch + 1,
                'semantic_logits': unwrapped_model.semantic_logits.data,
                'acoustic_values': unwrapped_model.acoustic_values.data,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
                'temperature': current_temperature,
                'config': config,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        if config.max_steps and epoch + 1 >= config.max_steps:
            logger.info(f"Reached max steps ({config.max_steps}), stopping training")
            break
    
    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    unwrapped_model = accelerator.unwrap_model(model)
    discrete_codes = unwrapped_model.get_discrete_codes()
    torch.save({
        'epoch': epoch + 1,
        'semantic_logits': unwrapped_model.semantic_logits.data,
        'acoustic_values': unwrapped_model.acoustic_values.data,
        'discrete_codes': discrete_codes,
        'config': config,
    }, final_path)
    logger.info(f"Training complete! Final model saved to: {final_path}")
    
    # Generate final audio sample
    final_audio_path = checkpoint_dir / "final_output.wav"
    save_wav(final_audio_path, generated_waveform.detach().cpu())
    logger.info(f"Final audio saved to: {final_audio_path}")
    
    # Save final codes
    codes_path = checkpoint_dir / "final_codes.pt"
    torch.save(discrete_codes, codes_path)
    logger.info(f"Final codes shape: {discrete_codes.shape}, saved to: {codes_path}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train voice cloning model using Voxtral TTS decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--model-path", default="../voxtral-tts-weights",
                       help="Path to Voxtral TTS model directory")
    parser.add_argument("--reference-audio", required=True,
                       help="Path to reference audio file")
    
    parser.add_argument("--num-frames", type=int, default=100,
                       help="Number of audio frames")
    parser.add_argument("--learning-rate", type=float, default=1e-2,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum number of training steps")
    
    parser.add_argument("--reconstruction-weight", type=float, default=1.0,
                       help="Weight for reconstruction loss")
    parser.add_argument("--speaker-weight", type=float, default=0.3,
                       help="Weight for speaker loss (monitoring only)")
    
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"],
                       help="Optimizer type")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                       help="Gradient clipping")
    
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Initial Gumbel-Softmax temperature")
    parser.add_argument("--temperature-decay", type=float, default=0.995,
                       help="Temperature decay per epoch")
    parser.add_argument("--min-temperature", type=float, default=0.5,
                       help="Minimum temperature")
    
    parser.add_argument("--device", default="cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--dtype", default="float32",
                       choices=["bfloat16", "float16", "float32"],
                       help="Model dtype")
    
    parser.add_argument("--checkpoint-dir", default="./checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=500,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--log-every", type=int, default=1,
                       help="Log every N epochs")
    parser.add_argument("--sample-every", type=int, default=500,
                       help="Generate audio sample every N epochs")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_path=args.model_path,
        reference_audio=args.reference_audio,
        num_frames=args.num_frames,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        reconstruction_weight=args.reconstruction_weight,
        speaker_embedding_weight=args.speaker_weight,
        optimizer=args.optimizer,
        grad_clip=args.grad_clip,
        temperature=args.temperature,
        temperature_decay=args.temperature_decay,
        min_temperature=args.min_temperature,
        device=args.device,
        dtype=args.dtype,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        sample_every=args.sample_every,
    )
    
    train(config)


if __name__ == "__main__":
    main()
