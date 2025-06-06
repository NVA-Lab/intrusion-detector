#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/


from typing import List, Optional, Tuple, Union
import os
import torch

from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..configuration_llava import LlavaConfig
from ..mm_utils import get_model_name_from_path, tokenizer_image_token

class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"

## FIXME we will follow the convention to add a new class for CausalLM in the future
class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    tokenizer_image_token = staticmethod(tokenizer_image_token)

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.dam_model = None
        self.pretrained_model_name_or_path = None
        self.init_vlm(config=config, *args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = torch.float16,
        init_dam: bool = False,
        # conv_mode and prompt_mode are only used by `init_dam` in `from_pretrained` if `init_dam` is set to True
        conv_mode: str = "v1",
        prompt_mode: str = "full+focal_crop",
        **kwargs,
    ):
        if torch_dtype:
            config.model_dtype = str(torch_dtype)
        if hasattr(cls, "load_pretrained"):
            obj = cls.load_pretrained(pretrained_model_name_or_path,
                                       *model_args, config=config, cache_dir=cache_dir, ignore_mismatched_sizes=ignore_mismatched_sizes, force_download=force_download, local_files_only=local_files_only, token=token,
                                       revision=revision, use_safetensors=use_safetensors, **kwargs
                                       )
        else:
            obj = super(LlavaLlamaModel).from_pretrained(pretrained_model_name_or_path,
                                                      *model_args, config=config, cache_dir=cache_dir, ignore_mismatched_sizes=ignore_mismatched_sizes, force_download=force_download, local_files_only=local_files_only, token=token,
                                                      revision=revision, use_safetensors=use_safetensors, **kwargs)
        obj.pretrained_model_name_or_path = pretrained_model_name_or_path
        
        # `init_dam` is used to initialize a `DescribeAnythingModel` object in a `LlavaLlamaModel` in DAM. If you initialize `DescribeAnythingModel` on your own outside, then you don't have to use this option.
        # This is very useful if you use `from_pretrained` with remote code execution and don't want to put implementation for `DescribeAnythingModel` class in your codebase.
        if init_dam:
            obj.init_dam(conv_mode, prompt_mode)
        
        return obj

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )
        # Note (kentang-mit@): we have a unit test for this function.
        if self.training:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            )
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            new_input_ids = input_ids

        outputs = self.llm.forward(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            seqlens_in_batch=sorted_seqlens_in_batch,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generation_kwargs,
    ):
        if images is not None:
            (
                _,
                _,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, None, images
            )
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.to(self.dtype)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )
        return outputs

    def init_dam(self, conv_mode, prompt_mode):
        from ...describe_anything_model import DescribeAnythingModel
        
        model_name = get_model_name_from_path(self.pretrained_model_name_or_path)
        self.dam_model = DescribeAnythingModel(model_path=dict(model=self, tokenizer=self.tokenizer, model_name=model_name), conv_mode=conv_mode, prompt_mode=prompt_mode)

        return self.dam_model

    @property
    def dam(self):
        if self.dam_model is None:
            self.init_dam()
        return self.dam_model

AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
