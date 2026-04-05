# Voxtral TTS Audio Autoencoder - Gradient-Based Discrete Code Reconstruction

![Approach illustration. Generated with Gemini based on the schema from the Voxtral TTS paper](Generated_Image_wdsro4wdsro4wdsr.png)

**Disclaimer**: It is a research experiment!

Reconstruct audio by training AE discrete bottleneck codes [num_frames, 37] using gradient descent through Voxtral's frozen decoder. This approach replaces the missing encoder with optimization-based code learning for a specific audio. Here we directly train codes for the select audio during many single-sample epochs.

**Key Idea**: Given target audio and a frozen decoder, find the optimal discrete codes that reconstruct it using differentiable quantization (Gumbel-Softmax + STE) and multi-objective losses. 

**⚠️ Note**: ~1 hour training on Mac M-series yields good results (5000 epochs). Works for single audio reconstruction.

## Files

- **`training_script.py`** - Train learnable codes to reconstruct audio
- **`voice_to_audio.py`** - Decode voice embeddings -> waveforms
- **`audio_tokenizer.py`** - Voxtral codec
- **`reconstruct_codes.py`** - Invert embeddings to discrete codes
- **`audio_generation.py`** - Components related to autoregressive (LLM) transformer for audio generation (some parts used in audio autoencoder - tokenizer)
- **`voice_embeddings/`** - Pre-computed embeddings (19 voices) - original from Voxtral-4B-TTS-2603
- **`codecs2audio.ipynb`** - Jupyter Notebook with audio reconstruction from embeddings, codes corruption experiments
- **`casual_female_clean.wav`** - Training reference audio (reconstructed from the embeddings `voice_embeddings/casual_female.pt`)
- **`casual_female_corrupted.wav`** - Corrupted through randomization of the semantic and some acoustic codes
- **`casual_female_reconstructed.wav`** - Reconstructed output using the training (after 5000 epochs)

## Quick Start

### Installation

```bash
pip install -r requirements.txt

# Download Voxtral-4B-TTS weights to voxtral-tts-weights/
# From: https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
hf download mistralai/Voxtral-4B-TTS-2603 --local-dir "voxtral-tts-weights"
```

### Train Audio Reconstruction

```bash
python training_script.py \
  --reference-audio casual_female_clean.wav \
  --model-path voxtral-tts-weights \
  --num-epochs 5000 \
  --learning-rate 0.1 \
  --device mps  # or cuda/cpu
```

Output saved to: `checkpoints/final_output.wav`
Saved trained codes: `checkpoints/final_codes.pt` - example how to read them at the bottom of `codecs2audio.ipynb`
Example result (5000 epochs): `casual_female_reconstructed.wav` in root folder

### Decode Voice Embeddings

Check `codecs2audio.ipynb` notebook for details.

## Requirements

Full list of requirements is in `requirements.txt` file.

## License (following Voxtral license)

Creative Commons Attribution Non Commercial 4.0 (CC BY-NC 4.0)
