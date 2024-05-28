import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import skimage
import torch
from PIL import Image
import sys
sys.path.append('./RMBG-1.4')
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
import numpy as np
import cv2
import requests

import io
import base64

model_path = '/data1/huggingface_ckpts/RMBG-1.4/model.pth'
im_path = "./remove_bg_examples/P01.jpg"

class RMBG:
    def __init__(self, model_path=model_path):
        self.net = self.Load_RMBG(model_path)

    def Load_RMBG(self, model_path):
        net = BriaRMBG()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()
        return net

    def Seg(self, im_path=None, input_img=None, model_input_size=[1024,1024], mask_filter_value=10, expand_ratio=None):
        print(f'Remove background using RMBG-1.4 ...')
        # prepare input
        if im_path:
            orig_image = Image.fromarray(skimage.io.imread(im_path))
            orig_im = np.array(Image.fromarray(skimage.io.imread(im_path)).convert('RGB'))
        if input_img:
            orig_image = input_img
            orig_im = np.array(input_img.convert('RGB'))
        if expand_ratio:
            fs=int(max(model_input_size) * expand_ratio)
            orig_im = cv2.copyMakeBorder(orig_im,fs,fs,fs,fs,borderType=cv2.BORDER_REPLICATE)
            orig_im = np.array(Image.fromarray(orig_im).resize(model_input_size))
        
        orig_im_size = orig_im.shape[0:2]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = preprocess_image(orig_im, model_input_size).to(device)
        
        # inference 
        result=self.net(image)
        
        # post process
        result_image = postprocess_image(result[0][0], orig_im_size)
        
        # save result
        pil_im = Image.fromarray(result_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))

        if expand_ratio:
            orig_image = Image.fromarray(orig_im)
        no_bg_image.paste(orig_image, mask=pil_im)
        mask_img_proc = Image.fromarray(np.where(result_image<mask_filter_value, 255, 0).astype(np.uint8))
        mask_img = Image.fromarray(255-result_image)
        return orig_image, no_bg_image, mask_img_proc, mask_img

def image2byte(image):
    '''
    image convert to byte
    image: PIL image format
    image_bytes: btye format
    '''
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="png")
    image_bytes = img_bytes.getvalue()
    return image_bytes

def usePixian(img_path=None, save_path=None, input_img=None):
    print(f'Remove background using Pixian ...')
    if img_path:
        bimg = open(img_path, 'rb')
    if input_img:
        bimg = image2byte(input_img)
    header = {'Authorization':'xxxx'}
    response = requests.post(
        'https://api.pixian.ai/api/v2/remove-background',
        files={'image': bimg},
        data={
            # TODO: Add more upload options here
        },
        headers=header
    )
    if response.status_code == requests.codes.ok:
        if save_path:
            with open(save_path, 'wb') as out:
                out.write(response.content)
        else:
            return Image.open(io.BytesIO(response.content))
    else:
        print("Error:", response.status_code, response.text)
    
    

if __name__ == "__main__":
    rmbg = RMBG(model_path)
    ori,nobg,mask = rmbg.Seg(im_path)
    print(ori.shape, nobg.shape, mask.shape)