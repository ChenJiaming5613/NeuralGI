# NeuralGI

## Train Environment

- [cuda 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

- python env

```shell
conda create -n neural_gi python=3.10.19
conda activate neural_gi

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install scipy OpenEXR matplotlib tqdm loguru
```