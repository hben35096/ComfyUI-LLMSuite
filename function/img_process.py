from PIL import Image
# import cv2
import numpy as np
import torch
import base64
from io import BytesIO


# from was-node-suite-comfyui
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# # from ComfyUI_LayerStyle
# def pil2cv2(pil_img:Image) -> np.array:
#     np_img_array = np.asarray(pil_img)
#     return cv2.cvtColor(np_img_array, cv2.COLOR_RGB2BGR)

# def cv22pil(cv2_img:np.ndarray) -> Image:
#     cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(cv2_img)

def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return mask

    
def tensor2base64(image_tensor, image_format="JPEG", quality=85):
    base64_encoded, mime_type = None, None
    try:
        buffer = BytesIO()

        img = tensor2pil(image_tensor)
        img.save(buffer, format=image_format, quality=quality)
        
        # 字节流转 b64
        base64_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime_type = "image/jpeg" if image_format == "JPEG" else "image/png"
    
    except Exception as e:
        print(f"Tensor to Base64 conversion failed: {e}")

    return base64_encoded, mime_type