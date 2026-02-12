import json
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.data.utils import VolumeNorm
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.fsq import DitheredFSQ
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

from .residual_fsq import ResidualFSQBottleneck


class StableCodec(nn.Module):
    def __init__(self,
        model_config_path: Optional[str] = None, ckpt_path: Optional[str] = None, pretrained_model: Optional[str] = None, device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        if pretrained_model is not None:
            print(f"Loading pretrained model `{pretrained_model}`.\n")
            self.model, model_config = get_pretrained_model(pretrained_model)
        else:
            if model_config_path is None:
                raise ValueError("Either `model_config_path` or `pretrained_model` should be provided.")
            print(f"Loading config from `{model_config_path}`.\n")
            with open(model_config_path) as f:
                model_config = json.load(f)
            self.model = create_model_from_config(model_config)
            if ckpt_path is not None:
                print(f"Loading weights from `{ckpt_path}`.\n")
                state = load_ckpt_state_dict(ckpt_path)
                copy_state_dict(self.model, state)
            
        self.model = self.model.to(self.device).eval().requires_grad_(False)
        
        self.residual_fsq: Optional[ResidualFSQBottleneck] = None

        self.sample_rate = model_config["sample_rate"]
        self.volume_norm = VolumeNorm([-20, 0], self.sample_rate)

        self.preset_bottleneck_configs = {
            "1x46656_400bps": [
                ([6, 6, 6, 6, 6, 6], 1.0)
            ],
            "2x15625_700bps": [
                ([5, 5, 5, 5, 5, 5], 1.0),
                ([5, 5, 5, 5, 5, 5], 0.25),
            ],
            "4x729_1000bps": [
                ([3, 3, 3, 3, 3, 3], 1.0),
                ([3, 3, 3, 3, 3, 3], 0.5),
                ([3, 3, 3, 3, 3, 3], 0.25),
                ([3, 3, 3, 3, 3, 3], 0.125),
            ]
        }

    def set_posthoc_bottleneck(self, stages):
        if isinstance(stages,str):
            if stages in self.preset_bottleneck_configs:
                stages = self.preset_bottleneck_configs[stages]
            else:
                raise ValueError(f"Unsupported preset bottleneck configuration `{stages}`.")

        self.residual_fsq = ResidualFSQBottleneck(stages).to(self.device).eval().requires_grad_(False)

    def encode(self, audio: Union[str, torch.Tensor], posthoc_bottleneck: bool = False, normalize: bool = True,**kwargs):
        """
        Encode audio into latents and tokens.

        Args:

        audio : Union[str, torch.Tensor]
            Path to an audio file or a `Tensor` of the eaudio itself.
        posthoc_bottleneck : bool
            Whether to inject a posthoc FSQ instead of the FSQ used during training.
            If `True`, its configuration should've been passed in with the `self.set_posthoc_bottleneck` method.
        normalize : bool
            Whether to normalize the audio to -20 LUFS before encoding (recommended).
        Other `kwargs` are the same as in `AudioAutoencoder.encode_audio` method.

        Returns:

        Tuple of `(continuous_latents, tokens)`.

        continuous_latents : torch.Tensor
            Pre-bottleneck latents in the `(B, H, S)` shape.
        tokens : torch.Tensor
            Bottleneck tokens in the `(B, S, 1)` shape.

        Where `B` is the batch size, `H` is the hidden dimension and `S` is the sequence length.
        """
        if isinstance(audio, str):
            audio, sample_rate = torchaudio.load(audio)
            audio = self.model.preprocess_audio_for_encoder(audio.to(self.device), sample_rate)
            if normalize:
                audio = self.volume_norm(audio.squeeze(0)).unsqueeze(0)

        latents, info = self.model.encode_audio(audio,
            return_info=True, skip_bottleneck=posthoc_bottleneck, **kwargs)
        if posthoc_bottleneck:
            tokens = self.residual_fsq.encode(latents)
        else:
            tokens = info["quantizer_indices"]

        return info["pre_bottleneck_latents"], tokens

    def decode(self, tokens: torch.Tensor, posthoc_bottleneck: bool = False, **kwargs):
        """
        Decode audio from tokens.

        Args:

        tokens : torch.Tensor
            Integer tokens produced by `encode` stage in `(B, S, 1)` shape.
        posthoc_bottleneck : bool
            Whether to inject a posthoc FSQ instead of the FSQ used during training.
            If `True`, its configuration should've been passed in with `self.set_posthoc_bottleneck` method.

        Returns:

        Decoded audio in the `(B, C, L)` shape.
        Where `B` is the batch size, `C` is the number of channels and `L` is the number of frames.
        """
        if posthoc_bottleneck:
            latents = self.residual_fsq.decode(tokens)
        else:
            latents = self.model.bottleneck.decode_tokens(tokens)
            latents = rearrange(latents, "b c n -> b n c")

        audio = self.model.decode_audio(latents, **kwargs)
        return audio

def main():
    sc = StableCodec(
       pretrained_model="stabilityai/stable-codec-speech-16k",
       device = torch.device("cuda")
    )

    sc.set_posthoc_bottleneck("2x15625_700bps")

    wavfile = "test.wav"

    posthoc_bottleneck = False
    latents, tokens = sc.encode(wavfile, posthoc_bottleneck=posthoc_bottleneck)
    decoded = sc.decode(tokens, posthoc_bottleneck=posthoc_bottleneck)
    torchaudio.save("decode.wav", decoded.squeeze(0).cpu(), 16000)

    posthoc_bottleneck = True
    latents, tokens = sc.encode(wavfile, posthoc_bottleneck=posthoc_bottleneck)
    decoded = sc.decode(tokens, posthoc_bottleneck=posthoc_bottleneck)
    torchaudio.save("decode-res.wav", decoded.squeeze(0).cpu(), 16000)

if __name__ == "__main__":
    main()
