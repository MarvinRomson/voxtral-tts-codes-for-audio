#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert trained discrete codes to voice embeddings.

This script takes trained audio codes [T, 37] and converts them back to
voice embeddings [T, 3072] by:
1. Adding special token offsets to raw codes
2. Looking up embeddings from the embedding table
3. Summing across all 37 codebooks

The output format matches the voice_embeddings/*.pt files and can be used
with the Voxtral TTS model.

Usage:
    python codes_to_embeddings.py \
        --codes checkpoints/final_codes.pt \
        --embedding-weight voxtral-tts-weights/consolidated.safetensors \
        --output voice_embeddings/my_trained_voice.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from reconstruct_codes import (
    OFFSETS,
    _CODEBOOK_SIZES,
    load_embedding_table,
    N_SPECIAL_TOKENS,
)

logger = logging.getLogger(__name__)


def codes_to_embeddings(
    codes: torch.Tensor,      # [T, 37] raw codes (no offset)
    emb_table: torch.Tensor,  # [V, D] embedding table
    offsets: torch.Tensor,    # [37] per-codebook start indices
) -> torch.Tensor:
    """Convert discrete codes to voice embeddings via table lookup and sum.
    
    Args:
        codes: [T, 37] int64 tensor of raw codes (0-8191 for semantic, 0-20 for acoustic)
        emb_table: [V, D] embedding table (V=9088, D=3072)
        offsets: [37] start index of each codebook in the embedding table
        
    Returns:
        voice_emb: [T, D] float32 tensor - sum of 37 embeddings per timestep
    """
    T, K = codes.shape
    D = emb_table.shape[1]
    
    # Validate input
    if K != 37:
        raise ValueError(f"Expected 37 codebooks, got {K}")
    
    # Initialize output
    voice_emb = torch.zeros(T, D, dtype=torch.float32)
    
    # Debug: Print code ranges
    logger.debug(f"Code ranges (raw, before any offset):")
    logger.debug(f"  Semantic (cb0): [{codes[:, 0].min()}, {codes[:, 0].max()}]")
    logger.debug(f"  Acoustic (cb1): [{codes[:, 1].min()}, {codes[:, 1].max()}]")
    
    # For each codebook, look up and sum embeddings
    for k in range(K):
        # Get codebook offset in embedding table
        codebook_offset = int(offsets[k].item())
        
        # Add N_SPECIAL_TOKENS (2) to shift raw codes to embedding table indices
        # Raw codes: 0-8191 → Table indices: 2-8193 (skip EMPTY=0, END=1)
        # Then add codebook offset to get final embedding table position
        adjusted_codes = codes[:, k] + N_SPECIAL_TOKENS + codebook_offset
        
        # Validate adjusted codes are in range
        if adjusted_codes.max() >= len(emb_table):
            raise ValueError(
                f"Codebook {k}: adjusted codes [{adjusted_codes.min()}, {adjusted_codes.max()}] "
                f"out of range for embedding table size {len(emb_table)}"
            )
        
        # Look up embeddings and add to sum
        embeddings = emb_table[adjusted_codes.long()]  # [T, D]
        voice_emb += embeddings
        
        if k == 0:  # Debug first codebook
            logger.debug(f"  Codebook {k}: +{N_SPECIAL_TOKENS} (special tokens) +{codebook_offset} (codebook offset)")
            logger.debug(f"    → Final embedding indices: [{adjusted_codes.min()}, {adjusted_codes.max()}]")
    
    return voice_emb


def prepare_codes_for_model(
    codes: torch.Tensor,  # [T, 37] raw discrete codes
) -> torch.Tensor:
    """Prepare codes for model by adding special token offsets.
    
    Converts raw codes (0-8191 for semantic, 0-20 for acoustic) to 
    model format (2+ for real codes, 0=EMPTY, 1=END_AUDIO).
    
    Args:
        codes: [T, 37] raw codes as output by training or coordinate descent
        
    Returns:
        model_codes: [T, 37] codes with special token offset (+2)
    """
    # Add special token offset: raw codes start at 0, model expects 2+
    return codes + N_SPECIAL_TOKENS


def add_end_audio_token(
    voice_emb: torch.Tensor,  # [T, D]
    emb_table: torch.Tensor,  # [V, D]
    offsets: torch.Tensor,    # [37]
) -> torch.Tensor:
    """Add END_AUDIO frame to the end of voice embedding.
    
    The END_AUDIO marker is code 1 in the first codebook (semantic),
    and code 0 (EMPTY) in all other codebooks.
    
    Args:
        voice_emb: [T, D] voice embedding
        emb_table: [V, D] embedding table
        offsets: [37] per-codebook offsets
        
    Returns:
        voice_emb_with_end: [T+1, D] with END_AUDIO frame appended
    """
    D = emb_table.shape[1]
    
    # END_AUDIO: semantic=1, all acoustic=0
    end_frame = torch.zeros(1, D, dtype=torch.float32)
    
    # Semantic codebook: code 1 (END_AUDIO)
    semantic_offset = int(offsets[0].item())
    end_frame += emb_table[semantic_offset + 1].unsqueeze(0)  # +1 for END_AUDIO
    
    # Acoustic codebooks: code 0 (EMPTY_AUDIO) for each
    for k in range(1, len(offsets)):
        offset = int(offsets[k].item())
        end_frame += emb_table[offset + 0].unsqueeze(0)  # +0 for EMPTY_AUDIO
    
    # Concatenate
    return torch.cat([voice_emb, end_frame], dim=0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert trained codes to voice embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--codes", required=True, metavar="PT_FILE",
        help="Path to trained codes file (.pt) - e.g., checkpoints/final_codes.pt",
    )
    p.add_argument(
        "--embedding-weight", required=True, metavar="PATH",
        help="Path to embedding table (.safetensors or .pt/.bin)",
    )
    p.add_argument(
        "--output", required=True, metavar="PT_FILE",
        help="Output path for voice embedding (.pt file)",
    )
    p.add_argument(
        "--add-end-token", action="store_true",
        help="Add END_AUDIO token at the end (recommended for compatibility)",
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
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    # Load embedding table
    logger.info(f"Loading embedding table from: {args.embedding_weight}")
    emb_table = load_embedding_table(args.embedding_weight)
    logger.info(f"Embedding table shape: {list(emb_table.shape)}")
    
    # Load trained codes
    logger.info(f"Loading codes from: {args.codes}")
    codes = torch.load(args.codes, map_location="cpu", weights_only=True)
    
    if isinstance(codes, dict):
        # Handle checkpoint format
        if 'discrete_codes' in codes:
            codes = codes['discrete_codes']
        else:
            raise ValueError(f"Checkpoint does not contain 'discrete_codes' key. Keys: {list(codes.keys())}")
    
    if not isinstance(codes, torch.Tensor):
        raise ValueError(f"Expected tensor, got {type(codes)}")
    
    logger.info(f"Codes shape: {list(codes.shape)}")
    
    if codes.dim() != 2 or codes.shape[1] != 37:
        raise ValueError(f"Expected codes shape [T, 37], got {list(codes.shape)}")
    
    T, K = codes.shape
    
    # Convert codes to embeddings
    logger.info("Converting codes to voice embeddings...")
    voice_emb = codes_to_embeddings(
        codes=codes,
        emb_table=emb_table,
        offsets=OFFSETS,
    )
    
    logger.info(f"Voice embedding shape: {list(voice_emb.shape)}")
    
    # Add END_AUDIO token if requested
    if args.add_end_token:
        logger.info("Adding END_AUDIO token...")
        voice_emb = add_end_audio_token(voice_emb, emb_table, OFFSETS)
        logger.info(f"Final shape with END_AUDIO: {list(voice_emb.shape)}")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(voice_emb, str(output_path))
    
    logger.info(f"✓ Saved voice embedding to: {output_path}")
    logger.info(f"  Shape: {list(voice_emb.shape)} ({T} frames, {voice_emb.shape[0]/12.5:.2f}s @ 12.5Hz)")
    logger.info(f"  Can be used with voice_to_audio.py to generate audio")


if __name__ == "__main__":
    main(sys.argv[1:])
