# Anywhere: A Multi-Agent Framework for Reliable and Diverse Foreground-Conditioned Image Inpainting

<a href='https://anywheremultiagent.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2404.18598'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 

## Quick start
### Environment setup
```bash
git clone https://github.com/Sealical/anywhere-multi-agent.git
cd anywhere-multi-angent
conda create -n anywhere python=3.10 -y
conda activate anywhere
pip install -r requirements.txt 
```

### Set-API-Key
Set up your API keys in `structJsonOutput.struct_text_generate.py` file

### Download & Set model path
- Download [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4/blob/main/model.pth) and set model path in `rmbgSegMask.py` file.
- Download [ControlNet_SDXL_canny](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0), [RealVisXL_V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0), [SDXL_inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) and set model path in `product2Image.py` file.

### Usage
See more details in `run_anywhere_pipeline.ipynb`


## Reference
```
@misc{xie2024anywhere,
      title={Anywhere: A Multi-Agent Framework for Reliable and Diverse Foreground-Conditioned Image Inpainting}, 
      author={Tianyidan Xie and Rui Ma and Qian Wang and Xiaoqian Ye and Feixuan Liu and Ying Tai and Zhenyu Zhang and Zili Yi},
      year={2024},
      eprint={2404.18598},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
 






