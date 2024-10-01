# A4W4KV16 RTN
WIKI_PATH=(PATH to wikitext dataset) python main.py --model (path to model)  --w_rtn --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4 --w_clip

# A4W4KV16 GPTQ
WIKI_PATH=(PATH to wikitext dataset) python main.py --model (path to model)  --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4 --w_clip

# A4W4KV16 RS
WIKI_PATH=(PATH to wikitext dataset) python main.py --model (path to model)  --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4 --w_clip --a_runtime_smooth

# A4W4KV16 QuaRot
WIKI_PATH=(PATH to wikitext dataset) python main.py --model (path to model)  --rotate --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4 --w_clip

# A4W4KV16 RRS (ours)
WIKI_PATH=(PATH to wikitext dataset) python main.py --model (path to model)  --rotate --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4 --w_clip --a_runtime_smooth --act_scale_g128

# ----------------------------------------------------------------------------------------------------------------------------


# A4W4KV4 RTN
WIKI_PATH=(PATH to wikitext dataset) python main.py --model (path to model)  --w_rtn --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --k_groupsize 128 --w_clip

# A4W4KV4 GPTQ
WIKI_PATH=(PATH to wikitext dataset) python main.py --model (path to model)  --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --k_groupsize 128 --w_clip

# A4W4KV16 RS
WIKI_PATH=(PATH to wikitext dataset) python main.py --model (path to model)  --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --k_groupsize 128 --w_clip --a_runtime_smooth

# A4W4KV4 QuaRot
WIKI_PATH=(PATH to wikitext dataset) python main.py --model (path to model)  --rotate --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --k_groupsize 128 --w_clip

# A4W4KV4 RRS (ours)
WIKI_PATH=(PATH to wikitext dataset) python main.py --model (path to model)  --rotate --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --k_groupsize 128 --w_clip --a_runtime_smooth --act_scale_g128