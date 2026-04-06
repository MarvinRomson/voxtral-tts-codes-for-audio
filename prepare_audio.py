#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Prepare audio file for training by resampling to 24kHz and converting to mono WAV.

This script preprocesses raw audio recordings to be compatible with the
Voxtral TTS training script. It:
1. Loads audio in any format (mp3, wav, flac, m4a, etc.)
2. Converts to mono if stereo
3. Resamples to 24kHz (Voxtral's sample rate)
4. Saves as 16-bit WAV file
5. Optionally trims silence from start/end

Usage:
    # Basic conversion
    python prepare_audio.py \
        --input my_recording.mp3 \
        --output prepared_audio.wav
    
    # With silence trimming
    python prepare_audio.py \
        --input my_recording.mp3 \
        --output prepared_audio.wav \
        --trim-silence \
        --silence-threshold -40
    
    # Specify duration
    python prepare_audio.py \
        --input my_recording.mp3 \
        --output prepared_audio.wav \
        --duration 8.0  # seconds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F

logger = logging.getLogger(__name__)

VOXTRAL_SAMPLE_RATE = 24000  # Voxtral TTS sample rate


def load_audio(
    path: str | Path,
    target_sr: int = VOXTRAL_SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """Load audio file and resample if needed.
    
    Args:
        path: Path to audio file (any format supported by torchaudio)
        target_sr: Target sample rate (default: 24000)
        
    Returns:
        waveform: 1D tensor of audio samples
        sample_rate: Sample rate of the audio
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    logger.info(f"Loading audio from: {path}")
    waveform, sample_rate = torchaudio.load(str(path))
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        logger.info(f"Converting from {waveform.shape[0]} channels to mono")
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Ensure shape is [1, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Resample if needed
    if sample_rate != target_sr:
        logger.info(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr
    
    logger.info(f"Loaded audio: {waveform.shape[1]} samples @ {sample_rate}Hz ({waveform.shape[1]/sample_rate:.2f}s)")
    
    return waveform.squeeze(0), sample_rate


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """Trim silence from start and end of audio.
    
    Args:
        waveform: 1D audio tensor
        sample_rate: Sample rate
        threshold_db: Silence threshold in dB (default: -40)
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation
        
    Returns:
        trimmed_waveform: Audio with silence trimmed
    """
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform, got {waveform.dim()}D")
    
    # Convert to 2D for processing
    waveform_2d = waveform.unsqueeze(0)
    
    # Compute frame-wise energy
    energy = torch.nn.functional.unfold(
        waveform_2d.unsqueeze(1),
        kernel_size=(1, frame_length),
        stride=(1, hop_length),
    ).pow(2).sum(dim=1).squeeze(0)
    
    # Convert to dB
    energy_db = 10 * torch.log10(energy + 1e-10)
    
    # Find non-silent frames
    non_silent = energy_db > threshold_db
    
    if not non_silent.any():
        logger.warning("No non-silent frames found, returning original audio")
        return waveform
    
    # Find first and last non-silent frame
    non_silent_indices = non_silent.nonzero(as_tuple=True)[0]
    start_frame = non_silent_indices[0].item()
    end_frame = non_silent_indices[-1].item() + 1
    
    # Convert frame indices to sample indices
    start_sample = start_frame * hop_length
    end_sample = min(end_frame * hop_length + frame_length, len(waveform))
    
    trimmed = waveform[start_sample:end_sample]
    
    trimmed_duration = len(trimmed) / sample_rate
    original_duration = len(waveform) / sample_rate
    logger.info(f"Trimmed silence: {original_duration:.2f}s → {trimmed_duration:.2f}s")
    
    return trimmed


def apply_duration_limit(
    waveform: torch.Tensor,
    sample_rate: int,
    max_duration: float | None,
) -> torch.Tensor:
    """Limit audio duration by trimming or padding.
    
    Args:
        waveform: 1D audio tensor
        sample_rate: Sample rate
        max_duration: Maximum duration in seconds (None = no limit)
        
    Returns:
        waveform: Audio trimmed/padded to duration
    """
    if max_duration is None:
        return waveform
    
    target_samples = int(max_duration * sample_rate)
    current_samples = len(waveform)
    
    if current_samples > target_samples:
        logger.info(f"Trimming audio to {max_duration}s")
        return waveform[:target_samples]
    elif current_samples < target_samples:
        logger.info(f"Padding audio to {max_duration}s")
        padding = target_samples - current_samples
        return torch.nn.functional.pad(waveform, (0, padding))
    
    return waveform


def normalize_audio(
    waveform: torch.Tensor,
    target_peak: float = 0.95,
) -> torch.Tensor:
    """Normalize audio to target peak amplitude.
    
    Args:
        waveform: 1D audio tensor
        target_peak: Target peak amplitude (0.0 to 1.0)
        
    Returns:
        normalized_waveform: Audio normalized to target peak
    """
    current_peak = waveform.abs().max()
    
    if current_peak < 1e-6:
        logger.warning("Audio appears to be silent (peak < 1e-6)")
        return waveform
    
    scale = target_peak / current_peak
    normalized = waveform * scale
    
    logger.info(f"Normalized audio: peak {current_peak:.4f} → {normalized.abs().max():.4f}")
    
    return normalized


def save_wav(
    path: str | Path,
    waveform: torch.Tensor,
    sample_rate: int = VOXTRAL_SAMPLE_RATE,
    bits_per_sample: int = 16,
) -> None:
    """Save audio as WAV file.
    
    Args:
        path: Output path
        waveform: 1D audio tensor
        sample_rate: Sample rate
        bits_per_sample: Bits per sample (8, 16, or 24)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure shape is [1, T] for torchaudio.save
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Save as WAV
    torchaudio.save(
        str(path),
        waveform,
        sample_rate,
        bits_per_sample=bits_per_sample,
    )
    
    logger.info(f"✓ Saved audio to: {path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare audio file for Voxtral TTS training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", required=True, metavar="AUDIO_FILE",
        help="Input audio file (any format: mp3, wav, flac, m4a, etc.)",
    )
    p.add_argument(
        "--output", "-o", required=True, metavar="WAV_FILE",
        help="Output WAV file path",
    )
    p.add_argument(
        "--sample-rate", type=int, default=VOXTRAL_SAMPLE_RATE,
        help="Target sample rate (Hz)",
    )
    p.add_argument(
        "--duration", type=float, default=None,
        help="Maximum duration in seconds (trim or pad to this length)",
    )
    p.add_argument(
        "--trim-silence", action="store_true",
        help="Trim silence from start and end",
    )
    p.add_argument(
        "--silence-threshold", type=float, default=-40.0,
        help="Silence threshold in dB (used with --trim-silence)",
    )
    p.add_argument(
        "--normalize", action="store_true",
        help="Normalize audio to target peak amplitude",
    )
    p.add_argument(
        "--target-peak", type=float, default=0.95,
        help="Target peak amplitude for normalization (0.0 to 1.0)",
    )
    p.add_argument(
        "--bits-per-sample", type=int, default=16, choices=[8, 16, 24],
        help="Bits per sample in output WAV",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    
    # Load audio
    waveform, sample_rate = load_audio(args.input, args.sample_rate)
    
    # Trim silence if requested
    if args.trim_silence:
        waveform = trim_silence(
            waveform,
            sample_rate,
            threshold_db=args.silence_threshold,
        )
    
    # Apply duration limit if specified
    if args.duration is not None:
        waveform = apply_duration_limit(waveform, sample_rate, args.duration)
    
    # Normalize if requested
    if args.normalize:
        waveform = normalize_audio(waveform, args.target_peak)
    
    # Save as WAV
    save_wav(args.output, waveform, sample_rate, args.bits_per_sample)
    
    # Print summary
    duration = len(waveform) / sample_rate
    print(f"\n{'='*60}")
    print(f"Audio prepared successfully!")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Samples: {len(waveform)}")
    print(f"Peak amplitude: {waveform.abs().max():.4f}")
    print(f"{'='*60}\n")
    print(f"Ready to use with training_script.py:")
    print(f"  python training_script.py --reference-audio {args.output} ...")


if __name__ == "__main__":
    main(sys.argv[1:])
