
# Rotated Runtime Smooth: Training-Free Activation Smoother for accurate INT4 inference


## Abstract
Large language models have demonstrated promising capabilities upon scaling up parameters. However, serving large language models incurs substantial computation and memory movement costs due to their large scale.

Quantization methods have been employed to reduce service costs and latency. Nevertheless, the presence of outliers in activations hinders the development of INT4 weight-activation quantization. Existing approaches utilize mixed-precision strategies or migrate outliers from activations to weights, suffering from either high latency or accuracy degradation. Based on the observation of activations from large language models, outliers can be classified into channel-wise outliers and spike outliers.

In order to eliminate channel-wise outliers, we propose Runtime Smooth **RS**, which is a play-and-plugin operator for activation quantization. Within the operator, activation is divided by the channel-wise maximums during runtime prior to quantization. Subsequently, the quantized weight and activation, along with the channel-wise maximums, are sent to the fused GEMM kernel for output with negligible overhead. To further address spike outliers, we propose Rotated Runtime Smooth **RRS**, where one spike outlier is spread on the entire token after rotation; hence consistent smoothing scale eliminates the effect of victims. The proposed method outperforms the state-of-the-art method in the Llama and Qwen families and achieves an improvement in WikiText-2 perplexity from 57.33 to 6.66 for INT4 inference.


## Preparation

Download Models from [Huggingface](https://huggingface.co/models) or [ModelScope](https://www.modelscope.cn/home)


Download Dataset [WikiText-2](https://www.modelscope.cn/datasets/modelscope/wikitext)

Requirements:
- Python 3.9

Install python dependencies
```bash
pip install -r requirements.txt
```

## Reproducing
For simulation results with fake quantization, check script directory.

The end-to-end INT4 inference pipeline is still under development.
