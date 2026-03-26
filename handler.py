import runpod
import torch
import base64
from io import BytesIO
from PIL import Image
from diffusers import FluxPipeline
import insightface
import numpy as np
import os
import requests

# 1. Initialize models globally (Runs once when the container starts)
print("Loading Flux Model...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", 
    torch_dtype=torch.bfloat16
).to("cuda")

print("Loading InsightFace Model...")
face_app = insightface.app.FaceAnalysis(name='buffalo_l', root='./insightface_model')
face_app.prepare(ctx_id=0, det_size=(640, 640))

def base64_to_cv2(b64_string):
    """Convert base64 string to OpenCV Image for InsightFace"""
    img_data = base64.b64decode(b64_string)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    # Convert PIL to CV2
    return np.array(img)[:, :, ::-1].copy()

def pil_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_similarity(img1_cv, img2_cv):
    """Compare two faces and return a similarity score (-1.0 to 1.0)"""
    faces1 = face_app.get(img1_cv)
    faces2 = face_app.get(img2_cv)
    
    if not faces1 or not faces2:
        return 0.0 # Face not found
        
    vec1 = faces1[0].embedding
    vec2 = faces2[0].embedding
    
    # Cosine Similarity
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(sim)

def handler(job):
    """RunPod Serverless Handler"""
    job_input = job['input']
    
    prompt = job_input.get('prompt')
    lora_id = job_input.get('lora_id') # Could be a HuggingFace path or URL
    reference_image_base64 = job_input.get('reference_image_base64')
    threshold = job_input.get('similarity_threshold', 0.65)
    num_steps = job_input.get('num_inference_steps', 4) # 4 is standard for FLUX.1-schnell
    
    if not prompt or not reference_image_base64:
        return {"error": "Missing prompt or reference_image_base64"}

    # Process reference image
    ref_cv2 = base64_to_cv2(reference_image_base64)

    # Note: If lora_id is provided, you would dynamically load it here
    # pipe.load_lora_weights(lora_id)

    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}: Generating Image...")
        
        # 1. Generate Image with Flux
        generated_image = pipe(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=0.0, # Schnell doesn't use guidance scale
            height=job_input.get('height', 1024),
            width=job_input.get('width', 832)
        ).images[0]

        # 2. Verify Face with InsightFace
        gen_cv2 = np.array(generated_image)[:, :, ::-1].copy()
        score = calculate_similarity(ref_cv2, gen_cv2)
        print(f"Face Similarity Score: {score}")

        if score >= threshold:
            # 3. Success! Return the image
            return {
                "verified_image_base64": pil_to_base64(generated_image),
                "similarity_score": score,
                "status": "success",
                "attempts": attempt + 1
            }
            
    # Unload LoRA if you loaded one to clear memory
    # pipe.unload_lora_weights()

    return {"error": f"Failed to generate an image above the {threshold} similarity threshold after {max_attempts} attempts."}

# Start the RunPod serverless function
runpod.serverless.start({"handler": handler})
