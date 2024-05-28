import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import torch
from diffusers import AutoPipelineForInpainting,StableDiffusionXLInpaintPipeline

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL,DPMSolverMultistepScheduler,StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid
from PIL import Image,ImageOps
import torch
import numpy as np
import cv2
import requests

import sys
from rmbgSegMask import RMBG

rmbg = RMBG()

img_path = "../badcase/badcase_4_seg.png"
prompts = ['elegant sophistication, minimalist aesthetic, high-end luxury ',
 ' vibrant energy, colorful vibrance, dynamic action ',
 ' serene tranquility, natural authenticity, eco-friendly simplicity']
num_per_prompt = 2

# model path
controlnet_path = "/data1/huggingface_ckpts/controlnet-canny-sdxl-1.0/"
sd_path = "/data1/huggingface_ckpts/RealVisXL_V4.0/"
refiner_path = "/data1/huggingface_ckpts/RealVisXL_V4.0/"
inpainting_model_path = "/data1/huggingface_ckpts/stable-diffusion-xl-1.0-inpainting-0.1/"

# lora_path = '../../stable-diffusion-webui-forge/models/Lora/huwai.safetensors'

general_prompt = ", Professional product photos, high quality, realistic, professional photography"
negative_prompt = 'ldrawing, painting, crayon, sketch, graphite, impressionist, noisy, blur, soft, deformed, ugly,people,human, (((watermark))), ((text)), english,lowres, (bad anatomy), bad hands, mutated hand, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, out of focus, glowing eyes, (((multiple views))), (((bad proportions))), (((multiple legs))), (((multiple arms))), 3D, bad_prompt, (worst quality:2.0), (low quality:2.0), inaccurate limb, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, tall, (painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), bad-prompt:0.5, (((((watermark))))), text, error, blurry, jpeg artifacts, cropped, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4),nsfw,ng_deepnegative_v1_75t,(badhandv4:1.2),(worst quality:2),(low quality:2),(normal quality:2),lowres,bad anatomy,bad hands,((monochrome)),((grayscale)) watermark,moles,nsfw'
refiner_prompt = "professional product photography, soft natural light, medium shot, graphics, masterpiece, 8k uhd, high quality, ultra realistic, professional photography, award winning, trending on pinterest, simplicity, high-end, modern art"

def mask_find_max_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[stats[:,4].argsort()][:-1]
    return stats[stats[:,4].argmax()]


def padLargeImage(image):
    img_arr = np.array(image)
    mask=image.split()[-1]
    mask_arr = np.array(mask)
    ret, mask_arr = cv2.threshold(mask_arr, 127, 255, cv2.THRESH_BINARY)
    b = mask_find_max_bboxs(mask_arr)
    img_w,img_h = image.size
    pxy = (b[0],b[1])
    pw = b[2]
    ph = b[3]
    print(f'product wxh:{pw}x{ph}, img wxh: {img_w}x{img_h}')
    
    w_th = 0.7
    h_th = 0.7
    if pw / img_w > w_th:
        side_pad = int((pw/w_th - img_w) // 2)
        img_arr = cv2.copyMakeBorder(img_arr,side_pad,side_pad,side_pad,side_pad,borderType=cv2.BORDER_CONSTANT)
        print(img_arr.shape)
    elif ph / img_h > h_th:
        side_pad = int((ph/h_th - img_h) // 2)
        img_arr = cv2.copyMakeBorder(img_arr,side_pad,side_pad,side_pad,side_pad,borderType=cv2.BORDER_CONSTANT)
        print(img_arr.shape)
    image = Image.fromarray(img_arr)
    return image


def cropProduct(image, bbox, crop_buffer=30):
    """
    crop image using bbox info
    """
    left_upper = (max(bbox[0]-crop_buffer,0), 0) # upper-left coord
    right_lower = (min(bbox[0]+bbox[2]+crop_buffer,image.size[0]), image.size[1]) # lower-right coord

    # # product bbox + buffer
    # left_upper = (max(bbox[0]-crop_buffer,0), max(bbox[1]-crop_buffer,0)) # upper-left coord
    # right_lower = (min(bbox[0]+bbox[2]+crop_buffer,image.size[0]), min(bbox[1]+bbox[3]+crop_buffer,image.size[0])) # lower-right coord

    res_img = image.crop(left_upper+right_lower)

    fix_info = [left_upper, right_lower]
    return res_img, fix_info
    

def getUnoverlapArea(ori_img, temp_img):
    ori_mask = cv2.threshold(np.array(ori_img.resize((1024,1024)).split()[-1]), 127,255,cv2.THRESH_BINARY)[1]
    # get product bbox and crop prodcut area
    bbox = mask_find_max_bboxs(ori_mask)
    crop_mask, crop_info = cropProduct(Image.fromarray(ori_mask), bbox)
    crop_mask = np.array(crop_mask)
    
    seg_img = rmbg.Seg(input_img=temp_img)[1]
    mask = seg_img.split()[-1] # .resize((1024,1024))
    mask_arr = cv2.threshold(np.array(mask),127,255,cv2.THRESH_BINARY)[1]
    crop_mask_arr,_ = cropProduct(Image.fromarray(mask_arr), bbox)
    crop_mask_arr = np.array(crop_mask_arr)
    
    rev_crop_mask = 255 - crop_mask_arr
    rev_crop_ori = 255 - crop_mask
    crop_res = cv2.threshold(crop_mask_arr.astype(np.float32) * rev_crop_ori.astype(np.float32), 
                     127, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    crop_nonoverlap_area = (crop_res==255).sum()
    crop_product_area = (crop_mask==255).sum()
    crop_nonoverlap_rate = crop_nonoverlap_area / crop_product_area
    print(f'Unoverlap_rate: {crop_nonoverlap_rate}')
    # recover uncrop mask
    top, bottom, left, right = 0,0,crop_info[0][0],1024-crop_info[1][0]
    ori_res=cv2.copyMakeBorder(crop_res,top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)
    
    return ori_res, crop_nonoverlap_rate

def getUnoverlapBackground(temp_img, ori_img, mask_arr):
    template_img = temp_img.copy()
    template_img.paste(Image.fromarray(mask_arr), mask=Image.fromarray(mask_arr))
    template_img.alpha_composite(ori_img)
    image = template_img.convert('RGB')
    mask = Image.fromarray(mask_arr).convert('RGB')
    return image, mask

def inpaintingByMask(init_image, mask_image, prompt=[""],random_seed=0, num_per_prompt=2):
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        inpainting_model_path, torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    # pipeline.enable_xformers_memory_efficient_attention()
    
    generator = torch.Generator("cuda").manual_seed(random_seed)
    #prompt = "professional product photos, on the sea beach, graphics, 8k uhd, high quality, realistic, professional photography, award winning, pinterest trending"
    #prompt = "in the cosmos, blue light"
    prompt = [x+general_prompt for x in prompt]
    neg_prompts = [negative_prompt for _ in prompt]
    images = pipeline(
        prompt=prompt, 
        image=init_image, 
        mask_image=mask_image, 
        generator=generator,
        num_inference_steps=50,
        num_images_per_prompt=num_per_prompt,
        negative_prompt=neg_prompts,
        # guidance_scale=5,
        # strength=0.8,
    ).images
    return images


def fixUnoverlapArea(temp_img, ori_img, max_depth=2, num_per_prompt=3, unoverlap_rate_th=0.03, if_stop=True):
    """
    Greedy search retains top1 results
    max_depth: max trying times
    num_per_prompt
    unoverlap_rate_th: minimal unoverlap rate
    """
    results = {}
    for i in range(max_depth):
        # get i-th result
        mask_arr,rate = getUnoverlapArea(ori_img, temp_img)
        if rate <=  unoverlap_rate_th:
            return results
        init_image,mask_image = getUnoverlapBackground(temp_img, ori_img, mask_arr)
        images = inpaintingByMask(init_image, mask_image,num_per_prompt=num_per_prompt)
        res = []
        for j in range(num_per_prompt):
            mask_arr,new_rate = getUnoverlapArea(ori_img, images[j])
            res.append([images[j], new_rate])
        # if new unoverlap area > original unoverlap area then stop
        if if_stop:
            if np.mean([x[1] for x in res]) >= rate:
                break
        # sorted by unoveralap area
        res.sort(key=lambda x:x[1])
        #temp_img = Image.alpha_composite(ori_img, res[0][0].convert('RGBA'))
        temp_img = res[0][0].convert('RGBA')
        results[i] = [temp_img, res[0][1]]
    return results
    
def preProcess(image):
    #image = padLargeImage(image)
    image = image.resize((1024,1024))
    return image

def postProcess(original_img, img_results, nonoverlap_th = 0.05):
    """
    input: 
        - original_img: PIL.Image, (pad) original product image (remove background)
        - img_results: List, [ori_img1, paste_img1, ori_img2, paste_img2,...]
    output:
        - post_results: [(post_img1,..., flag1), (post_img2,..., flag2), ...]
    """
    
    post_results = []
    results = []
    assert len(img_results)%2==0
    rmbg = RMBG()
    for ori_img, paste_img in [(img_results[i],img_results[i+1]) for i in range(0,len(img_results),2)]:
        
        # get product image mask (need remove background and RGBA mode)
        product_img = original_img
        ori_mask = cv2.threshold(np.array(product_img.resize((1024,1024)).split()[-1]), 127,255,cv2.THRESH_BINARY)[1]
        
        # get product bbox and crop prodcut area
        bbox = mask_find_max_bboxs(ori_mask)
        crop_mask = np.array(cropProduct(Image.fromarray(ori_mask), bbox))
        
        # crop canny results and remove background
        # seg_img = usePixian(input_img=ori_img)
        seg_img = rmbg.Seg(input_img=ori_img)[1]
        mask = seg_img.split()[-1] # .resize((1024,1024))
        mask_arr = cv2.threshold(np.array(mask),127,255,cv2.THRESH_BINARY)[1]
        crop_mask_arr = np.array(cropProduct(Image.fromarray(mask_arr), bbox))
        
        # # get non-overlap area
        # rev_mask = 255-mask_arr
        # rev_ori = 255-ori_mask
        # res = cv2.threshold(mask_arr.astype(np.float32) * rev_ori.astype(np.float32), 
        #                     127, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        # nonoverlap_area = (res==255).sum()
        # product_area = (ori_mask==255).sum()
        # nonoverlap_rate = round((nonoverlap_area / product_area)*100,3)
        
        # get new area non-overlap area
        rev_crop_mask = 255 - crop_mask_arr
        rev_crop_ori = 255 - crop_mask
        crop_res = cv2.threshold(crop_mask_arr.astype(np.float32) * rev_crop_ori.astype(np.float32), 
                            127, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        crop_nonoverlap_area = (crop_res==255).sum()
        crop_product_area = (crop_mask==255).sum()
        crop_nonoverlap_rate = crop_nonoverlap_area / crop_product_area

        post_results.append((Image.fromarray(mask_arr), Image.fromarray(ori_mask), crop_nonoverlap_rate))

        # filter
        if crop_nonoverlap_rate < nonoverlap_th:
            results.extend([ori_img, paste_img])
    return results
        

def getCanny(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

def getPipelineResults(img_path='', prompts=[''], input_img=None, num_per_prompt=2):
    if input_img:
        image_original = input_img
    else:
        image_original = Image.open(img_path)
    
    image_original = preProcess(image_original)

    # get canny map
    image = getCanny(image_original)

    # seg original
    rmbg = RMBG()
    image_original = rmbg.Seg(input_img=image_original)[1]

    # get mask img
    mask_image = image_original.split()[-1]
    mask_image = ImageOps.invert(mask_image) # convert black white


    # init canny2img
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path,
        torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        sd_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    # pipe.load_lora_weights(lora_path)
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    results = []
    img_results = []
    prompts = [x+general_prompt for x in prompts] # add general prompt for each input prompt
    
    for prompt in prompts:
        images = pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            image=image, 
            generator=torch.manual_seed(0),
            controlnet_conditioning_scale=0.5,
            num_inference_steps=50,
            num_images_per_prompt=num_per_prompt
            ).images
        img_results.extend(images)
    
    image_product = image_original.convert("RGBA")
    
    # inpainting unoverlap area
    inpaint_results = []
    for img in img_results:
        rbga_img = img.convert('RGBA')
        inp_res = fixUnoverlapArea(rbga_img, image_product, max_depth=2, num_per_prompt=2, unoverlap_rate_th=0.03, if_stop=False)
        if inp_res:
            inp_img,unoverlap_rate = [x for x in inp_res.items()][-1][1]
            inpaint_results.append([inp_img, unoverlap_rate])
        else:
            _,unoverlap_rate = getUnoverlapArea(image_product, rbga_img)
            inpaint_results.append([img, unoverlap_rate])

    inpaint_imgs = [x[0] for x in inpaint_results]
    #inpaint_results = []
    
    # init img2img
    pipeline_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        # refiner sd
        refiner_path, 
        torch_dtype=torch.float16
    ).to("cuda")
    #pipeline_img2img.enable_xformers_memory_efficient_attention()
    pipeline_img2img.enable_model_cpu_offload()
    
    results = []
    for idx,img_step1 in enumerate(inpaint_imgs):  
        img_step1 = img_step1.convert("RGBA")
        img_step2 = Image.alpha_composite(img_step1,image_product).convert("RGB")
    
        finished_images=pipeline_img2img(
                prompt=refiner_prompt,
                negative_prompt=negative_prompt,
                image=img_step2,
                num_inference_steps = 30,
                num_images_per_prompt=1
        ).images
        img_step3 = finished_images[0].convert("RGBA")
        img_step4 = Image.alpha_composite(img_step3, image_product).convert("RGBA")
        results.extend([img_step1,img_step4])
    
    # postprocess
    post_results = [] # postProcess(image_original, results)
    return image_original, results, inpaint_results