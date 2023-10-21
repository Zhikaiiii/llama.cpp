#!/usr/bin/env python3
# HF baichuan --> gguf conversion

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
import itertools
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor  # type: ignore[import]

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py' / 'gguf'))
import gguf


if TYPE_CHECKING:
    from typing import TypeAlias

NDArray: TypeAlias = 'np.ndarray[Any, Any]'

# reverse HF permute back to original pth layout


def reverse_hf_permute(weights: NDArray, n_head: int, n_kv_head: int | None = None) -> NDArray:
    if n_kv_head is not None and n_head != n_kv_head:
        n_head //= n_kv_head

    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))

def reverse_hf_permute_part(weights: NDArray, n_part: int, n_head: int, n_head_kv: int| None = None) -> NDArray:
        r = weights.shape[0] // 3
        return (reverse_hf_permute(weights[r * n_part : r * n_part + r, ...], n_head, n_head_kv))

def reverse_hf_part(weights: NDArray, n_part: int) -> NDArray:
        r = weights.shape[0] // 3
        return weights[r * n_part : r * n_part + r, ...]

def count_model_parts(dir_model: str) -> int:
    num_parts = 0

    for filename in os.listdir(dir_model):
        if filename.startswith("pytorch_model-"):
            num_parts += 1

    if num_parts > 0:
        print("gguf: found " + str(num_parts) + " model parts")

    return num_parts



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a HuggingFace LLaMA model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file, or model file itself (*.bin)",
    )
    parser.add_argument(
        "ftype", type=int, choices=[0, 1], default=1, nargs='?',
        help="output format - use 0 for float32, 1 for float16",
    )
    return parser.parse_args()

args = parse_args()

dir_model = args.model
ftype = args.ftype
if not dir_model.is_dir():
    print(f'Error: {args.model} is not a directory', file = sys.stderr)
    sys.exit(1)

# possible tensor data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16

# map from ftype to string
ftype_str = ["f32", "f16"]

if args.outfile is not None:
    fname_out = args.outfile
else:
    # output in the same directory as the model by default
    fname_out = dir_model / f'ggml-model-{ftype_str[ftype]}.gguf'

print("gguf: loading model "+dir_model.name)

with open(dir_model / "config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)
print("hello print: ",hparams["architectures"][0])
if hparams["architectures"][0] != "ChatGLMModel":
    print("Model architecture not supported: " + hparams["architectures"][0])

    sys.exit()

# get number of model parts
num_parts = count_model_parts(dir_model)
print(f"num_parts:{num_parts}\n")
ARCH=gguf.MODEL_ARCH.CHATGLM
gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

print("gguf: get model metadata")

block_count = hparams["num_layers"]
head_count = hparams["num_attention_heads"]
head_count_kv = hparams.get("multi_query_group_num", head_count)
hidden_size = hparams["hidden_size"]
ffn_hidden_size = hparams["ffn_hidden_size"]
rope_dimension_count = hidden_size // head_count // 2 # for chatglm2
layer_norm_eps = hparams["layernorm_epsilon"]
rope_ratio = hparams.get("rope_ratio", 1.0)

if "_name_or_path" in hparams:
    hf_repo = hparams["_name_or_path"]
else:
    hf_repo = ""

if "seq_length" in hparams:
    ctx_length = hparams["seq_length"]
else:
    print("gguf: can not find ctx length parameter.")

    sys.exit()


gguf_writer.add_name(dir_model.name)
gguf_writer.add_source_hf_repo(hf_repo)
gguf_writer.add_tensor_data_layout("Meta AI original pth")
gguf_writer.add_context_length(ctx_length)
gguf_writer.add_embedding_length(hidden_size)
gguf_writer.add_block_count(block_count)
gguf_writer.add_feed_forward_length(ffn_hidden_size)
gguf_writer.add_rope_dimension_count(rope_dimension_count) # rope 维度
gguf_writer.add_head_count(head_count)
gguf_writer.add_head_count_kv(head_count_kv)
gguf_writer.add_layer_norm_rms_eps(layer_norm_eps)
gguf_writer.add_rope_scale_linear(rope_ratio) # rope ratio


# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: list[bytes] = []
scores: list[float] = []
toktypes: list[int] = []

tokenizer_model_file = dir_model / 'tokenizer.model'
if not tokenizer_model_file.is_file():
    print(f'Error: Missing {tokenizer_model_file}', file = sys.stderr)
    sys.exit(1)

# vocab type sentencepiece
print("gguf: get sentencepiece tokenizer vocab, scores and token types")

tokenizer = SentencePieceProcessor(str(tokenizer_model_file))
vocab_size = hparams.get('vocab_size')
if vocab_size is None:
    vocab_size = tokenizer.vocab_size()

for i in range(vocab_size):
    text: bytes
    score: float

    piece = tokenizer.id_to_piece(i)
    text = piece.encode("utf-8")
    score = tokenizer.get_score(i)

    toktype = 1  # defualt to normal token type
    if tokenizer.is_unknown(i):
        toktype = 2
    if tokenizer.is_control(i):
        toktype = 3

    # toktype = 4 is user-defined = tokens from added_tokens.json

    if tokenizer.is_unused(i):
        toktype = 5
    if tokenizer.is_byte(i):
        toktype = 6

    tokens.append(text)
    scores.append(score)
    toktypes.append(toktype)

added_tokens_file = dir_model / 'added_tokens.json'
if added_tokens_file.is_file():
    with open(added_tokens_file, "r", encoding="utf-8") as f:
        addtokens_json = json.load(f)

        print("gguf: get added tokens")

        for key in addtokens_json:
            tokens.append( key.encode("utf-8") )
            scores.append(-1000.0)
            toktypes.append(4) # user-defined token type


special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
for s in special_tokens:
    tokens.append(s.encode("utf-8"))
    scores.append(-1000.0)
    toktypes.append(4)

vocab_size = len(tokens)
gguf_writer.add_tokenizer_model("llama")
gguf_writer.add_token_list(tokens)
gguf_writer.add_token_scores(scores)
gguf_writer.add_token_types(toktypes)

special_vocab = gguf.SpecialVocab(dir_model)
special_vocab.add_to_gguf(gguf_writer)

# TENSORS

tensor_map = gguf.get_tensor_name_map(ARCH,block_count)

# tensor info
print("gguf: get tensor metadata")

if num_parts == 0:
    part_names = iter(("pytorch_model.bin",))
else:
    part_names = (
        f"pytorch_model-{n:05}-of-{num_parts:05}.bin" for n in range(1, num_parts + 1)
    )


for part_name in part_names:
    if args.vocab_only:
        break
    print("gguf: loading model part '" + part_name + "'")
    model_part = torch.load(f"{dir_model}/{part_name}", map_location="cpu")


    for name in model_part.keys():
        data = model_part[name]
        # we don't need these
        if name.endswith(".rotary_pos_emb.inv_freq"):
            continue

        old_dtype = data.dtype

        # convert any unsupported data types to float32
        if data.dtype != torch.float16 and data.dtype != torch.float32:
            data = data.to(torch.float32)

        data = data.squeeze().numpy()
        if name == 'transformer.embedding.word_embeddings.weight':
            data = data[:vocab_size]
        if name == 'transformer.output_layer.weight':
            data = data[:vocab_size]

        # map tensor names
        new_name = tensor_map.get_name(name, try_suffixes = (".weight", ".bias"))
        if new_name is None:
            print("Can not map tensor '" + name + "'")
            sys.exit()

        n_dims = len(data.shape)
        data_dtype = data.dtype

        # if f32 desired, convert any float16 to float32
        if ftype == 0 and data_dtype == np.float16:
            data = data.astype(np.float32)

        # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
        if ftype == 1 and data_dtype == np.float16 and n_dims == 1:
            data = data.astype(np.float32)

        # if f16 desired, convert any float32 2-dim weight tensors to float16
        if ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
            data = data.astype(np.float16)

        print(name + " -> " +  new_name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))
        gguf_writer.add_tensor(new_name, data)


print("gguf: write header")
gguf_writer.write_header_to_file()
print("gguf: write metadata")
gguf_writer.write_kv_data_to_file()
if not args.vocab_only:
    print("gguf: write tensors")
    gguf_writer.write_tensors_to_file()

gguf_writer.close()

print(f"gguf: model successfully exported to '{fname_out}'")
print("")
