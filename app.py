import os
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import gradio as gr
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
)
from src.utils.mesh_util import save_obj
from src.utils.infer_util import remove_background, resize_foreground

import tempfile
from huggingface_hub import hf_hub_download

# Configuration setup
seed_everything(0)

config_path = 'configs/3dfusion.yaml'
config = OmegaConf.load(config_path)
model_config = config.model_config
infer_config = config.infer_config

device = torch.device('cuda')

# Load diffusion model
print('Loading diffusion model...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# Load custom UNet
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)
pipeline = pipeline.to(device)

# Load reconstruction model
print('Loading 3D reconstruction model...')
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_base.ckpt", repo_type="model")
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)
model = model.to(device).eval()

print('Model loading complete!')

# Custom CSS for clean UI
custom_css = """
#main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    font-family: 'Helvetica', Arial, sans-serif;
}

.header-section {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: #f8f9fa;
    border-radius: 12px;
}

.header-title {
    font-size: 2rem;
    color: #2d3436;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.header-subtitle {
    color: #636e72;
    margin-bottom: 1rem;
}

.input-section {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
}

.control-group {
    margin-bottom: 1.5rem;
}

.output-section {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.model-viewer {
    border: 2px dashed #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    min-height: 400px;
}

.processing-steps {
    color: #636e72;
    font-size: 0.9rem;
    margin-top: 1rem;
}
"""

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("Please upload an image to process")

def preprocess(input_image, do_remove_background):
    if do_remove_background:
        rembg_session = rembg.new_session()
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
    return input_image

def generate_mvs(input_image, sample_steps, sample_seed):
    seed_everything(sample_seed)
    generator = torch.Generator(device=device)
    z123_image = pipeline(
        input_image, 
        num_inference_steps=sample_steps, 
        generator=generator,
    ).images[0]
    
    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = rearrange(Image.fromarray(show_image), '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    return z123_image, show_image

def make3d(images):
    torch.cuda.empty_cache()
    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).float()
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)
    
    with torch.no_grad():
        planes = model.forward_planes(images, input_cameras)
    
    mesh_fpath = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    mesh_out = model.extract_mesh(planes, use_texture_map=False, **infer_config)
    
    vertices, faces, vertex_colors = mesh_out
    vertices = vertices[:, [1, 2, 0]]
    vertices[:, -1] *= -1
    faces = faces[:, [2, 1, 0]]
    save_obj(vertices, faces, vertex_colors, mesh_fpath)
    
    torch.cuda.empty_cache()
    return mesh_fpath

with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        # Header Section
        with gr.Column(visible=False) as header:
            gr.Markdown("""
            <div class="header-section">
                <h1 class="header-title">3DFusion: Image to 3D Object</h1>
                <p class="header-subtitle">Transform single images into detailed 3D models using advanced AI reconstruction</p>
            </div>
            """)
        
        # Input Section
        with gr.Column(visible=False) as input_section:
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    width=400,
                    height=400,
                    type="pil",
                    elem_classes=["input-image"],
                    show_label=False
                )
                processed_image = gr.Image(
                    label="Processed Image",
                    image_mode="RGBA",
                    width=400,
                    height=400,
                    type="pil",
                    interactive=False,
                    show_label=False
                )
            
            with gr.Row():
                with gr.Column():
                    do_remove_background = gr.Checkbox(
                        label="Automatically remove background",
                        value=True,
                        info="Recommended for objects with complex backgrounds"
                    )
                    
                    with gr.Row():
                        sample_seed = gr.Number(
                            label="Random Seed",
                            value=42,
                            precision=0,
                            info="Change seed value for different variations"
                        )
                        sample_steps = gr.Slider(
                            label="Processing Steps",
                            minimum=30,
                            maximum=75,
                            value=75,
                            step=5,
                            info="Higher values may improve quality but take longer"
                        )
                    
                    generate_btn = gr.Button(
                        "Generate 3D Model",
                        variant="primary",
                        size="lg",
                        icon="🛠️"
                    )
        
        # Output Section
        with gr.Column(visible=False) as output_section:
            with gr.Tab("3D Model"):
                output_model = gr.Model3D(
                    label="Generated 3D Model",
                    elem_classes=["model-viewer"],
                    interactive=False
                )
                gr.Markdown("""
                <div class="processing-steps">
                    <p>Processing Steps:</p>
                    <ol>
                        <li>Image preprocessing and background removal</li>
                        <li>Multi-view synthesis (75 diffusion steps)</li>
                        <li>3D geometry reconstruction</li>
                        <li>Mesh optimization and texturing</li>
                    </ol>
                </div>
                """)
            
            with gr.Tab("Multi-view Preview"):
                mv_preview = gr.Image(
                    label="Generated Views",
                    type="pil",
                    interactive=False
                )
        
        # Examples Section
        with gr.Column():
            gr.Examples(
                examples=[
                    os.path.join("examples", img) 
                    for img in sorted(os.listdir("examples"))
                ],
                inputs=[input_image],
                label="Example Inputs",
                examples_per_page=6
            )
    
    # State management
    mv_images = gr.State()
    
    # Event handling
    generate_btn.click(
        fn=check_input_image,
        inputs=[input_image]
    ).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[processed_image]
    ).success(
        fn=generate_mvs,
        inputs=[processed_image, sample_steps, sample_seed],
        outputs=[mv_images, mv_preview]
    ).success(
        fn=make3d,
        inputs=[mv_images],
        outputs=[output_model]
    )

demo.queue(concurrency_count=2)
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)