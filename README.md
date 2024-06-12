## Install 


```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia

conda install nvidia/label/cuda-12.1.0::cuda
conda install nvidia/label/cuda-12.1.0::cuda-cudart-dev
conda install nvidia/label/cuda-12.1.0::cuda-cudart

pip install -U transformers==4.41.2
pip install vllm==0.4.3 
pip install accelerate
pip install optimum

git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install -vvv --no-build-isolation -e .

cd ..


pip install flash-attn==2.5.9.post1 --no-build-isolation
git clone https://github.com/casper-hansen/AutoAWQ && cd AutoAWQ
pip install -e .

git clone https://github.com/casper-hansen/AutoAWQ_kernels && cd AutoAWQ_kernels
pip install -e .
cd ..

pip install bitsandbytes==0.42.1

```

## VLLM Deployment 

```
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4
```