import gc
import os
import copy
import numpy as np
from PIL import Image
import torch
import gradio as gr
from collections import OrderedDict
from scipy.ndimage import binary_dilation
import json
from modules import scripts, shared, script_callbacks
from modules.ui import gr_show
from modules.ui_components import FormRow, ToolButton
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessing
from modules.devices import device, torch_gc, cpu
from modules.paths import models_path
from scripts.process_params import SAMProcessUnit, max_cn_num

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

SAM3_AVAILABLE = True

refresh_symbol = '\U0001f504'
sam_model_cache = OrderedDict()
scripts_sam_model_dir = os.path.join(scripts.basedir(), "models/sam")
sd_sam_model_dir = os.path.join(models_path, "sam")
sam_model_dir = sd_sam_model_dir if os.path.exists(sd_sam_model_dir) else scripts_sam_model_dir

def list_sam3_weight_files(dir_):
    allowed = {'.pt'}
    files = []
    for f in os.listdir(dir_):
        full = os.path.join(dir_, f)
        if not os.path.isfile(full):
            continue
        _, ext = os.path.splitext(f)
        if ext.lower() in allowed:
            files.append(f)
    return sorted(files)

sam_model_list = list_sam3_weight_files(sam_model_dir)
sam_device = device

# ── helpers ──────────────────────────────────────────────

def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)

def update_mask(mask_gallery, chosen_mask, dilation_amt, input_image):
    if isinstance(mask_gallery, list) and len(mask_gallery) > chosen_mask + 3:
        image_data = mask_gallery[chosen_mask + 3]
        if isinstance(image_data, tuple) and len(image_data) == 2:
            mask_image = Image.open(image_data[0])
        elif isinstance(image_data, dict) and 'name' in image_data:
            mask_image = Image.open(image_data['name'])
        else:
            mask_image = image_data
    else:
        return [None, None, None]

    binary_img = np.array(mask_image.convert('1'))
    if dilation_amt:
        mask_image, binary_img = dilate_mask(binary_img, dilation_amt)
    blended_image = Image.fromarray(show_masks(np.array(input_image), binary_img.astype(np.bool_)[None, ...]))
    matted_image = np.array(input_image)
    matted_image[~binary_img] = np.array([0, 0, 0, 0])
    return [blended_image, mask_image, Image.fromarray(matted_image)]

def dilate_mask(mask, dilation_amt):
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)
    dilated_binary_img = binary_dilation(mask, dilation_kernel)
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
    return dilated_mask, dilated_binary_img

def create_mask_output(image_np, masks, boxes_filt=None):
    mask_images, masks_gallery, matted_images = [], [], []
    for mask in masks:
        mask_union = np.any(mask, axis=0)
        mask_pil = Image.fromarray(mask_union)
        masks_gallery.append(mask_pil)

        blended_np = show_masks(image_np, mask)
        blended_pil = Image.fromarray(blended_np)
        mask_images.append(blended_pil)

        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~mask_union] = np.array([0, 0, 0, 0])
        matted_pil = Image.fromarray(image_np_copy)
        matted_images.append(matted_pil)

    return mask_images + masks_gallery + matted_images

def parse_points(points):
    if isinstance(points, str):
        points = json.loads(points)
    return points

# ── SAM3 Predictor ──────────────────────────────────────

class SAM3Predictor:
    def __init__(self, model):
        self.model = model
        self.processor = Sam3Processor(model, confidence_threshold=0.3)

    def set_image(self, image_rgb):
        self.state = self.processor.set_image(Image.fromarray(image_rgb))

    def predict_text(self, text_prompt, point_coords=None, point_labels=None,
                     box=None, score_threshold=0.3, multimask_output=True):
        self.processor.set_confidence_threshold(score_threshold)
        output = self.processor.set_text_prompt(state=self.state, prompt=text_prompt)

        masks_tensor = output["masks"]      # (N, 1, H, W)
        scores_tensor = output["scores"]    # (N,)

        # Union of all instance masks (everything above threshold)
        union_mask = masks_tensor.any(dim=0, keepdim=True)   # (1, 1, H, W)

        # Sort individual masks by score × area
        areas = masks_tensor.sum(dim=(-1, -2)).squeeze(1)   # (N,)
        combined = scores_tensor * areas                    # (N,)
        sort_idx = torch.argsort(combined, descending=True)
        sorted_masks = masks_tensor[sort_idx]

        # Prepend the union mask, then keep up to 3 masks total
        all_masks = torch.cat([union_mask, sorted_masks], dim=0)   # (1 + N, 1, H, W)
        k = min(3, all_masks.shape[0])
        masks_tensor = all_masks[:k]

        if isinstance(masks_tensor, torch.Tensor):
            masks_tensor = masks_tensor.cpu().numpy()
        if masks_tensor.ndim == 2:
            masks_tensor = masks_tensor[None, ...]

        # Pad to exactly 3 masks (empty if needed)
        while masks_tensor.shape[0] < 3:
            empty = np.zeros_like(masks_tensor[:1])
            masks_tensor = np.concatenate([masks_tensor, empty], axis=0)

        return masks_tensor.astype(bool)

# ── model loading ────────────────────────────────────────

def load_sam3_model(ckpt_name):
    if not SAM3_AVAILABLE:
        raise RuntimeError("SAM3 is not available.")
    ckpt_path = os.path.join(sam_model_dir, ckpt_name)
    extension_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bpe_path = os.path.join(extension_dir, "models", "bpe_simple_vocab_16e6.txt.gz")

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=ckpt_path,
        device='cpu',
        eval_mode=False,
        load_from_HF=False
    )
    model.to(sam_device)
    model.eval()
    return model

def clear_sam_cache():
    sam_model_cache.clear()
    gc.collect()
    torch_gc()

def clear_cache():
    clear_sam_cache()

def garbage_collect(sam):
    if shared.cmd_opts.lowvram:
        sam.to(cpu)
    gc.collect()
    torch_gc()

def refresh_sam_models(*inputs):
    global sam_model_list
    sam_model_list = list_sam3_weight_files(sam_model_dir)
    dd = inputs[0]
    selected = dd if dd in sam_model_list else (sam_model_list[0] if sam_model_list else None)
    return gr.Dropdown.update(choices=sam_model_list, value=selected)

def init_sam_model(sam_model_name):
    if sam_model_name in sam_model_cache:
        sam = sam_model_cache[sam_model_name]
        if shared.cmd_opts.lowvram or (str(sam_device) not in str(sam.device)):
            sam.to(device=sam_device)
        return sam
    elif sam_model_name in sam_model_list:
        clear_sam_cache()
        sam_model_cache[sam_model_name] = load_sam3_model(sam_model_name)
        return sam_model_cache[sam_model_name]
    else:
        raise Exception(f"{sam_model_name} not found, please download model to models/sam.")

# ── main segmentation ────────────────────────────────────

def sam_predict(sam_model_name, input_image, positive_points, negative_points,
                text_prompt, score_threshold=0.3, fallback_image_base64=None):
    # Fallback: decode base64 string from inpaint tab if present and direct image is missing
    if input_image is None and fallback_image_base64 and len(fallback_image_base64) > 0:
        import base64, io
        # Clean up the string: remove any whitespace, then strip the data URL header
        fallback_image_base64 = fallback_image_base64.strip()
        if ',' in fallback_image_base64:
            # e.g. "data:image/png;base64,iVBOR..."
            fallback_image_base64 = fallback_image_base64.split(',', 1)[-1].strip()
        try:
            img_bytes = base64.b64decode(fallback_image_base64)
            input_image = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
        except Exception as e:
            return [], f"Failed to decode fallback image: {e}"

    positive_points = parse_points(positive_points)
    negative_points = parse_points(negative_points)

    if sam_model_name is None:
        return [], "SAM model not found."
    if input_image is None:
        return [], "No input image."
    if not text_prompt or text_prompt.strip() == "":
        return [], "Please enter a text prompt."

    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]

    point_coords = np.array(positive_points + negative_points, dtype=np.float32) \
        if (positive_points or negative_points) else None
    point_labels = np.array([1]*len(positive_points) + [0]*len(negative_points), dtype=np.int64) \
        if point_coords is not None else None

    sam = init_sam_model(sam_model_name)

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
            pred = SAM3Predictor(sam)
            pred.set_image(image_np_rgb)
            masks = pred.predict_text(
                text_prompt=text_prompt,
                point_coords=point_coords,
                point_labels=point_labels,
                score_threshold=score_threshold,
                multimask_output=True
            )

    garbage_collect(sam)

    if masks.ndim == 3:
        masks = masks[:, None, :, :]

    return create_mask_output(image_np, masks), "Segmentation done."

# ── priority helper ──────────────────────────────────────

def priorize_sam_scripts(is_img2img):
    cnet_idx, sam_idx = None, None
    if is_img2img:
        for idx, s in enumerate(scripts.scripts_img2img.alwayson_scripts):
            if s.title() == "Segment Anything":
                sam_idx = idx
            elif s.title() == "ControlNet":
                cnet_idx = idx
        if cnet_idx is not None and sam_idx is not None and cnet_idx < sam_idx:
            scripts.scripts_img2img.alwayson_scripts[cnet_idx], scripts.scripts_img2img.alwayson_scripts[
                sam_idx] = scripts.scripts_img2img.alwayson_scripts[sam_idx], scripts.scripts_img2img.alwayson_scripts[cnet_idx]
    else:
        for idx, s in enumerate(scripts.scripts_txt2img.alwayson_scripts):
            if s.title() == "Segment Anything":
                sam_idx = idx
            elif s.title() == "ControlNet":
                cnet_idx = idx
        if cnet_idx is not None and sam_idx is not None and cnet_idx < sam_idx:
            scripts.scripts_txt2img.alwayson_scripts[cnet_idx], scripts.scripts_txt2img.alwayson_scripts[
                sam_idx] = scripts.scripts_txt2img.alwayson_scripts[sam_idx], scripts.scripts_txt2img.alwayson_scripts[cnet_idx]

# ── UI components ────────────────────────────────────────

def ui_inpaint(is_img2img, max_cn):
    with FormRow():
        inpaint_upload_enable = gr.Checkbox(
            value=False,
            label="Use this mask for inpainting",
        )
        cnet_inpaint_invert = gr.Checkbox(
            value=False,
            label='ControlNet inpaint not masked',
            visible=((max_cn > 0) and not is_img2img),
        )
        cnet_inpaint_idx = gr.Radio(
            value="0" if max_cn > 0 else None,
            choices=[str(i) for i in range(max_cn)],
            label='ControlNet Inpaint Index',
            type="index",
            visible=((max_cn > 0) and not is_img2img),
        )
    return inpaint_upload_enable, cnet_inpaint_invert, cnet_inpaint_idx

def ui_dilation(sam_output_mask_gallery, sam_output_chosen_mask, sam_input_image):
    sam_dilation_checkbox = gr.Checkbox(value=False, label="Expand Mask")
    with gr.Column(visible=False) as dilation_column:
        sam_dilation_amt = gr.Slider(minimum=0, maximum=100, value=0, label="Specify the amount that you wish to expand the mask by (recommend 30)")
        sam_dilation_output_gallery = gr.Gallery(label="Expanded Mask", columns=3)
        sam_dilation_submit = gr.Button(value="Update Mask")
        sam_dilation_submit.click(
            fn=update_mask,
            inputs=[sam_output_mask_gallery, sam_output_chosen_mask, sam_dilation_amt, sam_input_image],
            outputs=[sam_dilation_output_gallery])
    sam_dilation_checkbox.change(fn=gr_show, inputs=[sam_dilation_checkbox], outputs=[dilation_column], show_progress=False)
    return sam_dilation_checkbox, sam_dilation_output_gallery

# ── Script class ─────────────────────────────────────────

class Script(scripts.Script):
    def title(self):
        return "Segment Anything"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        if max_cn_num() > 0:
            priorize_sam_scripts(is_img2img)

        tab_prefix = ("img2img" if is_img2img else "txt2img") + "_sam_"

        with gr.Accordion('Segment Anything 3', open=False):
            with gr.Row():
                with gr.Column(scale=10):
                    with gr.Row():
                        sam_model_name = gr.Dropdown(
                            label="SAM Model",
                            choices=sam_model_list,
                            value=sam_model_list[0] if sam_model_list else None)
                        sam_refresh_models = ToolButton(value=refresh_symbol, variant="tool")
                        sam_refresh_models.click(refresh_sam_models, sam_model_name, sam_model_name)
                with gr.Column(scale=1):
                    sam_use_cpu = gr.Checkbox(value=False, label="Use CPU for SAM")
                    def change_sam_device(use_cpu=False):
                        global sam_device
                        sam_device = "cpu" if use_cpu else device
                    sam_use_cpu.change(fn=change_sam_device, inputs=[sam_use_cpu], show_progress=False)

            sam_fallback_image_data = gr.Textbox(
                visible=False,
                elem_id=f"{tab_prefix}fallback_image_data"
            )

            # --- Content (no TabItem) ---
            gr.HTML("<p>Left click to add a positive point (black), right click for negative (red). Left click a point to remove it.</p>")
            sam_input_image = gr.Image(label="Image for Segment Anything", elem_id=f"{tab_prefix}input_image",
                                       source="upload", type="pil", image_mode="RGBA")
            sam_remove_dots = gr.Button(value="Remove all point prompts")
            sam_dummy_component = gr.Textbox(visible=False)
            sam_remove_dots.click(
                fn=lambda _: None,
                _js="samRemoveDots",
                inputs=[sam_dummy_component],
                outputs=None)

            gr.HTML("<p>Enter a text prompt to guide segmentation (e.g., 'shoes', 'person', 'left arm').</p>")
            sam_text_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. shoes", elem_id=f"{tab_prefix}text_prompt")
            sam_score_threshold = gr.Slider(label="Score Threshold", value=0.3, minimum=0.0, maximum=1.0, step=0.01)

            sam_output_mask_gallery = gr.Gallery(label='Segment Anything Output', columns=3, interactive=False)
            sam_submit = gr.Button(value="Preview Segmentation", elem_id=f"{tab_prefix}run_button")
            sam_result = gr.Text(value="", label="Segment Anything status")

            sam_submit.click(
                fn=sam_predict,
                _js='submit_sam',
                inputs=[sam_model_name, sam_input_image,
                        sam_dummy_component, sam_dummy_component,
                        sam_text_prompt, sam_score_threshold,
                        sam_fallback_image_data],   # always passed, filled by JS when needed
                outputs=[sam_output_mask_gallery, sam_result])

            with FormRow():
                sam_output_chosen_mask = gr.Radio(label="Choose your favorite mask: ",
                                                 value="0", choices=["0", "1", "2"], type="index")
                gr.Checkbox(value=False, label="Preview automatically when add/remove points",
                            elem_id=f"{tab_prefix}realtime_preview_checkbox")

            (sam_inpaint_upload_enable, sam_cnet_inpaint_invert,
             sam_cnet_inpaint_idx) = ui_inpaint(is_img2img, max_cn_num())

            sam_dilation_checkbox, sam_dilation_output_gallery = ui_dilation(
                sam_output_mask_gallery, sam_output_chosen_mask, sam_input_image)

            ui_process = (
                sam_inpaint_upload_enable, sam_cnet_inpaint_invert, sam_cnet_inpaint_idx,
                sam_input_image, sam_output_mask_gallery, sam_output_chosen_mask,
                sam_dilation_checkbox, sam_dilation_output_gallery)

        return ui_process

    def process(self, p: StableDiffusionProcessing, *args):
        is_img2img = isinstance(p, StableDiffusionProcessingImg2Img)
        process_unit = SAMProcessUnit(args, is_img2img)
        process_unit.set_process_attributes(p)


# ── callbacks ─────────────────────────────────────────────

def on_after_component(component, **_kwargs):
    pass

def on_ui_settings():
    pass

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(on_after_component)