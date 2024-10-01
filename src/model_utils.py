import torch
import typing
import transformers
import utils
import os
import logging

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
QWEN2_MODEL = transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
QWEN2_LAYER = transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer
QWEN2_MOE_MODEL = transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeForCausalLM
QWEN2_MOE_LAYER = transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeDecoderLayer
MISTRAL_MODEL = transformers.models.mistral.modeling_mistral.MistralForCausalLM
MISTRAL_LAYER = transformers.models.mistral.modeling_mistral.MistralDecoderLayer
MIXTRAL_MODEL = transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM
MIXTRAL_LAYER = transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer

def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, QWEN2_MOE_MODEL):
        return QWEN2_MOE_MODEL
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_MODEL
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_MODEL
    elif isinstance(model, MIXTRAL_MODEL):
        return MIXTRAL_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL):
        return "apply_rotary_pos_emb"
    if isinstance(model, QWEN2_MOE_MODEL) or isinstance(model, MIXTRAL_MODEL):
        return "apply_rotary_pos_emb"
    elif isinstance(model, MISTRAL_MODEL):
        return "apply_rotary_pos_emb"
    elif isinstance(model, QWEN2_MODEL):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers
    if isinstance(model, QWEN2_MOE_MODEL) or isinstance(model, MIXTRAL_MODEL):
        return model.model.layers
    if isinstance(model, QWEN2_MODEL):
        return model.model.layers
    if isinstance(model, MISTRAL_MODEL):
        return model.model.layers
    raise NotImplementedError


def get_model(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                                use_auth_token=hf_token,
                                                                low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model



def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif isinstance(model, QWEN2_MOE_MODEL):
        model_type = QWEN2_MOE_MODEL
    elif isinstance(model, QWEN2_MODEL):
        model_type = QWEN2_MODEL
    elif isinstance(model, MISTRAL_MODEL):
        model_type = MISTRAL_MODEL
    elif isinstance(model, MIXTRAL_MODEL):
        model_type = MIXTRAL_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type

def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL:
        return [model.model.embed_tokens]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    elif model_type == QWEN2_MOE_MODEL or model_type == MIXTRAL_MODEL:
        return [model.model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    elif model_type == QWEN2_MOE_MODEL or model_type == MIXTRAL_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == QWEN2_MODEL or model_type == MISTRAL_MODEL or model_type == LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    elif model_type == QWEN2_MOE_MODEL or model_type == MIXTRAL_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    elif model_type == QWEN2_MOE_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                        transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeRMSNorm)
    elif model_type == QWEN2_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                        transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm)
    elif model_type == MISTRAL_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                        transformers.models.mistral.modeling_mistral.MistralRMSNorm)
    elif model_type == MIXTRAL_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                        transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL:
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    elif model_type == QWEN2_MOE_MODEL or model_type == MIXTRAL_MODEL:
        return model.config.intermediate_size
    else:
        raise ValueError(f'Unknown model type {model_type}')

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, 'layer_io', f'{args.layer_idx:03d}.pt')

def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type == LLAMA_MODEL or model_type == QWEN2_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    elif model_type == QWEN2_MOE_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': [],
            'shared_expert_gate': [],
            'shared_expert.gate_proj': [],
            'shared_expert.down_proj': [],
            'shared_expert.up_proj': [],
            'gate': [],
        }
        for k in range(60):
            captured_inputs[f'expert.{k}.gate_proj'] = []
            captured_inputs[f'expert.{k}.up_proj'] = []
            captured_inputs[f'expert.{k}.down_proj'] = []

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    elif model_type == OPT_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'out_proj': [],
            'fc1': [],
            'fc2': []
        }
        captured_outputs = {
            'v_proj': [],
        }
        for name in captured_inputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f'Unknown model type {model_type}')

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx:seq_idx + 1].to(utils.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }
