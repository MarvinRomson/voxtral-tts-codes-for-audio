#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compare statistics between two audio files to identify preprocessing differences.

Usage:
    python compare_audio.py \
        --audio1 casual_female_clean.wav \
        --audio2 my_reference.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio file."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
    else:
        waveform = waveform.squeeze(0)
    return waveform, sr


def compute_statistics(waveform: torch.Tensor, sr: int, name: str):
    """Compute comprehensive audio statistics."""
    waveform_np = waveform.numpy()
    
    stats = {
        "name": name,
        "sample_rate": sr,
        "duration": len(waveform) / sr,
        "num_samples": len(waveform),
        "channels": 1,  # already converted to mono
        
        # Amplitude statistics
        "min": float(waveform.min()),
        "max": float(waveform.max()),
        "mean": float(waveform.mean()),
        "std": float(waveform.std()),
        "abs_max": float(waveform.abs().max()),
        
        # RMS (loudness)
        "rms": float(torch.sqrt((waveform ** 2).mean())),
        
        # Dynamic range
        "dynamic_range_db": 20 * np.log10(waveform.abs().max() / (waveform.std() + 1e-8)),
        
        # Zero crossings (speech characteristic)
        "zero_crossings": int(((waveform_np[:-1] * waveform_np[1:]) < 0).sum()),
        "zcr_rate": ((waveform_np[:-1] * waveform_np[1:]) < 0).sum() / len(waveform),
        
        # Clipping detection
        "clipped_samples": int((waveform.abs() >= 0.999).sum()),
        "clipping_ratio": float((waveform.abs() >= 0.999).sum() / len(waveform)),
        
        # Silence detection
        "silent_samples": int((waveform.abs() < 0.001).sum()),
        "silence_ratio": float((waveform.abs() < 0.001).sum() / len(waveform)),
        
        # DC offset
        "dc_offset": float(waveform.mean()),
        
        # Peak-to-peak
        "peak_to_peak": float(waveform.max() - waveform.min()),
    }
    
    return stats


def print_comparison(stats1: dict, stats2: dict):
    """Print side-by-side comparison of statistics."""
    
    print("\n" + "="*80)
    print(f"AUDIO COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<25} {stats1['name']:<25} {stats2['name']:<25}")
    print("-"*80)
    
    # Basic info
    print(f"{'Sample Rate':<25} {stats1['sample_rate']:<25} {stats2['sample_rate']:<25}")
    print(f"{'Duration (s)':<25} {stats1['duration']:<25.3f} {stats2['duration']:<25.3f}")
    print(f"{'Num Samples':<25} {stats1['num_samples']:<25} {stats2['num_samples']:<25}")
    
    print("\n" + "-"*80)
    print("AMPLITUDE STATISTICS")
    print("-"*80)
    
    print(f"{'Min':<25} {stats1['min']:<25.6f} {stats2['min']:<25.6f}")
    print(f"{'Max':<25} {stats1['max']:<25.6f} {stats2['max']:<25.6f}")
    print(f"{'Mean':<25} {stats1['mean']:<25.6f} {stats2['mean']:<25.6f}")
    print(f"{'Std Dev':<25} {stats1['std']:<25.6f} {stats2['std']:<25.6f}")
    print(f"{'Abs Max (Peak)':<25} {stats1['abs_max']:<25.6f} {stats2['abs_max']:<25.6f}")
    print(f"{'Peak-to-Peak':<25} {stats1['peak_to_peak']:<25.6f} {stats2['peak_to_peak']:<25.6f}")
    
    print("\n" + "-"*80)
    print("LOUDNESS & DYNAMICS")
    print("-"*80)
    
    print(f"{'RMS (Loudness)':<25} {stats1['rms']:<25.6f} {stats2['rms']:<25.6f}")
    print(f"{'Dynamic Range (dB)':<25} {stats1['dynamic_range_db']:<25.2f} {stats2['dynamic_range_db']:<25.2f}")
    
    print("\n" + "-"*80)
    print("QUALITY METRICS")
    print("-"*80)
    
    print(f"{'DC Offset':<25} {stats1['dc_offset']:<25.6f} {stats2['dc_offset']:<25.6f}")
    print(f"{'Zero Crossing Rate':<25} {stats1['zcr_rate']:<25.6f} {stats2['zcr_rate']:<25.6f}")
    print(f"{'Clipped Samples':<25} {stats1['clipped_samples']:<25} {stats2['clipped_samples']:<25}")
    print(f"{'Clipping Ratio':<25} {stats1['clipping_ratio']:<25.6f} {stats2['clipping_ratio']:<25.6f}")
    print(f"{'Silent Samples':<25} {stats1['silent_samples']:<25} {stats2['silent_samples']:<25}")
    print(f"{'Silence Ratio':<25} {stats1['silence_ratio']:<25.6f} {stats2['silence_ratio']:<25.6f}")
    
    print("\n" + "="*80)
    print("KEY DIFFERENCES")
    print("="*80)
    
    issues = []
    
    # Check sample rate
    if stats1['sample_rate'] != stats2['sample_rate']:
        issues.append(f"❌ Sample rate mismatch: {stats1['sample_rate']} vs {stats2['sample_rate']}")
    
    # Check duration
    duration_diff = abs(stats1['duration'] - stats2['duration'])
    if duration_diff > 0.1:
        issues.append(f"⚠️  Duration differs by {duration_diff:.2f}s")
    
    # Check peak amplitude
    peak_diff = abs(stats1['abs_max'] - stats2['abs_max'])
    if peak_diff > 0.1:
        issues.append(f"⚠️  Peak amplitude differs by {peak_diff:.3f}")
        if stats2['abs_max'] < 0.5:
            issues.append(f"   → Audio 2 may be too quiet (peak={stats2['abs_max']:.3f})")
        if stats2['abs_max'] > 0.99:
            issues.append(f"   → Audio 2 may be clipping (peak={stats2['abs_max']:.3f})")
    
    # Check RMS (loudness)
    rms_ratio = stats2['rms'] / (stats1['rms'] + 1e-10)
    if abs(rms_ratio - 1.0) > 0.3:
        issues.append(f"⚠️  RMS loudness differs by {abs(rms_ratio - 1.0)*100:.1f}%")
        issues.append(f"   → Audio 2 is {rms_ratio:.2f}x {'louder' if rms_ratio > 1 else 'quieter'}")
    
    # Check DC offset
    if abs(stats2['dc_offset']) > 0.01:
        issues.append(f"⚠️  Audio 2 has DC offset: {stats2['dc_offset']:.6f}")
    
    # Check clipping
    if stats2['clipping_ratio'] > 0.001:
        issues.append(f"❌ Audio 2 is clipping! {stats2['clipping_ratio']*100:.2f}% of samples")
    
    # Check silence
    silence_ratio_diff = abs(stats1['silence_ratio'] - stats2['silence_ratio'])
    if silence_ratio_diff > 0.1:
        issues.append(f"⚠️  Silence ratio differs: {stats1['silence_ratio']:.3f} vs {stats2['silence_ratio']:.3f}")
    
    if not issues:
        print("✅ No significant differences detected!")
    else:
        for issue in issues:
            print(issue)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Normalization
    if stats2['abs_max'] < 0.7:
        recommendations.append("🔧 Increase normalization: --target-peak 0.95")
    elif stats2['abs_max'] > 0.99:
        recommendations.append("🔧 Reduce normalization: --target-peak 0.90")
    
    # Duration
    if stats1['duration'] != stats2['duration']:
        recommendations.append(f"🔧 Match duration: --duration {stats1['duration']:.2f}")
    
    # DC offset
    if abs(stats2['dc_offset']) > 0.01:
        recommendations.append("🔧 Remove DC offset (add high-pass filter)")
    
    # Silence
    if stats2['silence_ratio'] > stats1['silence_ratio'] + 0.1:
        recommendations.append("🔧 Trim silence: --trim-silence")
    
    if not recommendations:
        print("✅ Audio appears properly preprocessed!")
    else:
        print("\nSuggested fixes for Audio 2:")
        for rec in recommendations:
            print(rec)
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare two audio files")
    parser.add_argument("--audio1", required=True, help="First audio file (reference)")
    parser.add_argument("--audio2", required=True, help="Second audio file (to compare)")
    args = parser.parse_args()
    
    # Load audios
    print(f"Loading {args.audio1}...")
    waveform1, sr1 = load_audio(args.audio1)
    
    print(f"Loading {args.audio2}...")
    waveform2, sr2 = load_audio(args.audio2)
    
    # Compute statistics
    stats1 = compute_statistics(waveform1, sr1, Path(args.audio1).name)
    stats2 = compute_statistics(waveform2, sr2, Path(args.audio2).name)
    
    # Print comparison
    print_comparison(stats1, stats2)


if __name__ == "__main__":
    main()
