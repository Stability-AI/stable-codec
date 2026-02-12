# Stable Codec

This repository contains training and inference scripts for models in the Stable Codec series, starting with `stable-codec-speech-16k` - introduced in the paper titled Scaling Transformers for Low-bitrate High-Quality Speech Coding.

Paper: https://arxiv.org/abs/2411.19842

Sound demos: https://stability-ai.github.io/stable-codec-demo/

Model weights: https://huggingface.co/stabilityai/stable-codec-speech-16k

## Changelog

### [v0.1.3] TBD
- __Fix__ restricted Python version to <3.12 due to dependency incompatibilities
- __Fix__ clarified installation instructions regarding Python version requirements
### [v0.1.2] 14-01-25
- __New__ added hooks for `stable-codec-speech-16k-base`. 
- __Fix__ fixed major issue with precision in FSQ token calculation, which was degrading results. Fix is currently local, will be upstreamed to `stable-audio-tools` later.
### [v0.1.1] 10-01-25 
- Release


##

Note that whilst this code is MIT licensed, the model weights are covered by the [Stability AI Community License](https://huggingface.co/stabilityai/stable-codec-speech-16k/blob/main/LICENSE.md)

## Variants
The model is currently available in two variants:
- `stable-codec-speech-16k` is an improved finetune, with boosted latent semantics. __It should be used in 99% of use-cases.__
- `stable-codec-speech-16k-base` is the weights corresponding to the results in our [publication](https://arxiv.org/abs/2411.19842), provided for reproducibility.

### Additional Training

In addition to the training described in the paper, the weights for `stable-codec-speech-16k` have undergone 500k steps of finetuning with force-aligned data from LibriLight and the English portion Multilingual LibriSpeech. This was performed by using a CTC head to regress the force-aligned phoneme tags from pre-bottleneck latents. We found that this additional training significantly boosted the applicability of the codec tokens to downstream tasks like TTS, at a small cost to objective reconstruction metrics.

## Install

The model itself is defined in [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) package.

### Python Version Compatibility

**Important:** This package currently requires **Python 3.9, 3.10, or 3.11**. Python 3.12 and later are not supported due to incompatibilities in the `stable-audio-tools` dependency chain (specifically `PyWavelets==1.4.1` and `pandas==2.0.2`).

If you attempt to install on Python 3.12+, you will encounter build errors. Please use Python 3.11 or earlier.

To install `stable-codec`:

```bash
pip install stable-codec
pip install -U flash-attn --no-build-isolation
```

**IMPORTANT NOTE:** This model currently has a hard requirement for FlashAttention due to its use of sliding window attention. Inference without FlashAttention will likely be greatly degraded. This also means that the model currently does not support CPU inference. We will relax the dependency on FlashAttention in the future.

## Encoding and decoding

To encode audio or decode tokens, the `StableCodec` class provides a convenient wrapper for the model. It can be used with a local checkpoint and config as follows:

```python
import torch
import torchaudio
from stable_codec import StableCodec

model = StableCodec(
    model_config_path="<path-to-model-config>",
    ckpt_path="<path-to-checkpoint>", # optional, can be `None`,
    device = torch.device("cuda")
)

audiopath = "audio.wav"

latents, tokens = model.encode(audiopath)
decoded_audio = model.decode(tokens)

torchaudio.save("decoded.wav", decoded_audio, model.sample_rate)
```

To download the model weights automatically from HuggingFace, simply provide the model name:

```python
model = StableCodec(
    pretrained_model = 'stabilityai/stable-codec-speech-16k'
)
```
### Posthoc bottleneck configuration

Most usecases will benefit from replacing the training-time FSQ bottleneck with a post-hoc FSQ bottleneck, as described in the paper. This allows token dictionary size to be reduced to a reasonable level for modern language models. This is achieved by calling the `set_posthoc_bottleneck` function, and setting a flag to the encode/decode calls:

```python
model.set_posthoc_bottleneck("2x15625_700bps")
latents, tokens = model.encode(audiopath, posthoc_bottleneck = True)
decoded_audio = model.decode(tokens, posthoc_bottleneck = True)
```
`set_posthoc_bottleneck` can take a string as argument, which allows selection a number of recommended preset settings for the bottleneck:

| Bottleneck Preset | Number of Tokens per step | Dictionary Size | Bits Per Second (bps) |
|-------------------|------------------|-----------------|-----------------------|
| `1x46656_400bps`   | 1             | 46656             | 400                   |
| `2x15625_700bps`   | 2             | 15625             | 700                   |
| `4x729_1000bps`    | 4             | 729               | 1000                  |

Alternatively, the bottleneck stages can be specified directly. The format for specifying this can be seen in the definition of the `StableCodec` class in `model.py`.

### Normalization

The model is trained with utterances normalized to -20 +-5 LUFS. The `encode` function normalizes to -20 LUFS by default, but it can be disabled by setting `normalize = False` when calling the function. 

## Finetune

To finetune a model given its config and checkpoint, execute `train.py` file:

```bash
python train.py \
    --project "stable-codec" \
    --name "finetune" \
    --config-file "defaults.ini" \
    --save-dir "<ckpt-save-dir>" \
    --model-config "<path-to-config.json>" \
    --dataset-config "<dataset-config.json>" \
    --val-dataset-config "<dataset-config.json>" \
    --pretrained-ckpt-path "<pretrained-model-ckpt.ckpt>" \
    --ckpt-path "$CKPT_PATH" \
    --num-nodes $SLURM_JOB_NUM_NODES \
    --num-workers 16 --batch-size 10 --precision "16-mixed" \
    --checkpoint-every 10000 \
    --logger "wandb"
```

For dataset configuration, refer to `stable-audio-tools` [dataset docs](https://github.com/Stability-AI/stable-audio-tools/blob/main/docs/datasets.md).


### Using CTC loss

To use [CTC loss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
during training you have to enable it in the training configuration file
and in the training dataset configuration.

1. Modifying training configuration:
    - Enable CTC projection head and set its hidden dimension:
      ```python
      config["model"]["use_proj_head"] = True
      config["model"]["proj_head_dim"] = 81
      ```
    - Enable CTC in the training part of the config:
      ```python
      config["training"]["use_ctc"] = True
      ```
    - And set its loss config:
      ```python
      config["training"]["loss_configs"]["ctc"] = {
        "blank_idx": 80,
        "decay": 1.0,
        "weights": {"ctc": 1.0}
      }
      ```
    - Optionally, you can enable computation of the Phone-Error-Rate (PER) during validation:
      ```python
      config["training"]["eval_loss_configs"]["per"] = {}
      ```

2. Configuring dataset (only WebDataset format is supported for CTC):
   - The dataset configuration should have one additional field set to it (see [dataset docs](https://github.com/Stability-AI/stable-audio-tools/blob/main/docs/datasets.md) for other options):
     ```python
     config["force_align_text"] = True
     ```
   - And the JSON metadata file for each sample should contain force aligned transcript under `force_aligned_text` entry in the format specified below (besides other metadata).
     Where `transcript` is a list of word-level alignments with `start` and `end` fields specifying range **in seconds** of each word.
     ```json
     "normalized_text":"and i feel"
     "force_aligned_text":{
      "transcript":[
         {
            "word":"and",
            "start":0.2202,
            "end":0.3403
         },
         {
            "word":"i",
            "start":0.4604,
            "end":0.4804
         },
         {
            "word":"feel",
            "start":0.5204,
            "end":0.7006
         }
       ]
     }
     ```
## Objective Metrics

| Model                     | SI-SDR | Mel Dis | STFT Dis | PESQ | STOI | 
|---------------------------|-------:|--------:|---------:|-----:|-----:|
| `stable-codec-speech-16k-base`         | 4.73   | 0.86    | 1.26     | 3.09 | 0.92 |
| `stable-codec-speech-16k` | 3.58   | 0.90    | 1.30     | 3.01 | 0.90 | 

