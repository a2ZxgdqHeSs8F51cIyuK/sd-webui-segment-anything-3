from typing import Tuple, List, Dict, Any, Union
from PIL import Image, ImageOps
import numpy as np
from modules import shared


def max_cn_num():
    if shared.opts.data is None:
        return 1
    return int(shared.opts.data.get('control_net_max_models_num', 1))


def _gallery_image_path(item: Union[Tuple, List, Dict, str, None]) -> str:
    """Extract file path from a Gradio gallery item (tuple/dict/string)."""
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get('name', None)
    if isinstance(item, (tuple, list)):
        return item[0] if len(item) > 0 else None
    if isinstance(item, str):
        return item
    return None


class SAMInpaintUnit:
    def __init__(self, args: Tuple, is_img2img=False):
        self.is_img2img = is_img2img

        self.inpaint_upload_enable: bool = False
        self.cnet_inpaint_invert: bool = False
        self.cnet_inpaint_idx: int = 0
        self.input_image = None
        self.output_mask_gallery: List = None
        self.output_chosen_mask: int = 0
        self.dilation_checkbox: bool = False
        self.dilation_output_gallery: List = None
        self.init_sam_single_image_process(args)

    def init_sam_single_image_process(self, args):
        self.inpaint_upload_enable      = args[0]
        self.cnet_inpaint_invert        = args[1]
        self.cnet_inpaint_idx           = args[2]
        self.input_image                = args[3]
        self.output_mask_gallery        = args[4]
        self.output_chosen_mask         = args[5]
        self.dilation_checkbox          = args[6]
        self.dilation_output_gallery    = args[7]

    def get_input_and_mask(self, mask_blur):
        image, mask = None, None
        if self.inpaint_upload_enable and self.input_image is not None and self.output_mask_gallery is not None:
            if self.dilation_checkbox and self.dilation_output_gallery is not None:
                path = _gallery_image_path(self.dilation_output_gallery[1])
                if path:
                    mask = Image.open(path).convert('L')
            elif self.output_mask_gallery is not None:
                idx = self.output_chosen_mask + 3
                if idx < len(self.output_mask_gallery):
                    path = _gallery_image_path(self.output_mask_gallery[idx])
                    if path:
                        mask = Image.open(path).convert('L')
            if mask is not None and self.cnet_inpaint_invert:
                mask = ImageOps.invert(mask)
            image = self.input_image
        return image, mask


class SAMProcessUnit:
    def __init__(self, args: Tuple, is_img2img=False):
        self.is_img2img = is_img2img

        # --- main SAM inpaint unit (always present) ---
        self.sam_inpaint_unit = SAMInpaintUnit(args[:8], is_img2img)

        # --- optional legacy features – gracefully handle missing args ---
        remaining = args[8:]

        # ControlNet segmentation copy (4 args)
        if len(remaining) >= 4:
            self.init_cnet_seg_process(remaining[:4])
            remaining = remaining[4:]
        else:
            self.cnet_seg_output_gallery = None
            self.cnet_seg_enable_copy = False
            self.cnet_seg_idx = 0
            self.cnet_seg_gallery_input = 0

        # Crop & inpaint unit (another 8 args)
        if len(remaining) >= 8:
            self.crop_inpaint_unit = SAMInpaintUnit(remaining[:8], is_img2img)
            remaining = remaining[8:]
        else:
            self.crop_inpaint_unit = None

        # ControlNet upload (4 args)
        if len(remaining) >= 4:
            self.init_cnet_upload_process(remaining[:4])
        else:
            self.cnet_upload_enable = False
            self.cnet_upload_num = 0
            self.cnet_upload_img_inpaint = None
            self.cnet_upload_mask_inpaint = None

    def init_cnet_seg_process(self, args):
        self.cnet_seg_output_gallery    = args[0]
        self.cnet_seg_enable_copy       = args[1]
        self.cnet_seg_idx               = args[2]
        self.cnet_seg_gallery_input     = args[3]

    def init_cnet_upload_process(self, args):
        self.cnet_upload_enable         = args[0]
        self.cnet_upload_num            = args[1]
        self.cnet_upload_img_inpaint    = args[2]
        self.cnet_upload_mask_inpaint   = args[3]

    def set_process_attributes(self, p):
        inpaint_mask_blur = getattr(p, "mask_blur", 0)

        # -------------------- Fallback image for SAM --------------------
        if self.sam_inpaint_unit.input_image is None:
            if self.is_img2img and hasattr(p, 'init_images') and p.init_images:
                self.sam_inpaint_unit.input_image = p.init_images[0]
            elif not self.is_img2img:
                # txt2img: try the image already set in the target ControlNet slot
                cn_idx = self.sam_inpaint_unit.cnet_inpaint_idx
                cn_input = getattr(p, 'control_net_input_image', None)
                if cn_input is not None:
                    if isinstance(cn_input, list) and cn_idx < len(cn_input):
                        entry = cn_input[cn_idx]
                    else:
                        entry = cn_input
                    if isinstance(entry, dict) and 'image' in entry:
                        self.sam_inpaint_unit.input_image = entry['image']
        # -----------------------------------------------------------------

        inpaint_image, inpaint_mask = self.sam_inpaint_unit.get_input_and_mask(inpaint_mask_blur)
        inpaint_cn_num = self.sam_inpaint_unit.cnet_inpaint_idx

        # Fallback: if the main unit didn't produce a mask, try the crop unit
        if inpaint_image is None and self.crop_inpaint_unit is not None:
            inpaint_image, inpaint_mask = self.crop_inpaint_unit.get_input_and_mask(inpaint_mask_blur)
            inpaint_cn_num = self.crop_inpaint_unit.cnet_inpaint_idx

        if inpaint_image is not None and inpaint_mask is not None:
            if self.is_img2img:
                p.init_images = [inpaint_image]
                p.image_mask = inpaint_mask
            else:
                self.set_p_value(p, 'control_net_input_image', inpaint_cn_num,
                                 {"image": inpaint_image, "mask": inpaint_mask.convert("L")})

        # Optional: ControlNet segmentation copy
        if self.cnet_seg_enable_copy and self.cnet_seg_output_gallery is not None:
            cnet_seg_gallery_index = 1
            if len(self.cnet_seg_output_gallery) == 3 and self.cnet_seg_gallery_input is not None:
                cnet_seg_gallery_index += self.cnet_seg_gallery_input
            if cnet_seg_gallery_index < len(self.cnet_seg_output_gallery):
                path = _gallery_image_path(self.cnet_seg_output_gallery[cnet_seg_gallery_index])
                if path:
                    img = Image.open(path)
                    self.set_p_value(p, 'control_net_input_image', self.cnet_seg_idx, img)

        # Optional: manual ControlNet upload
        if self.cnet_upload_enable and self.cnet_upload_img_inpaint is not None and self.cnet_upload_mask_inpaint is not None:
            self.set_p_value(p, 'control_net_input_image', self.cnet_upload_num,
                             {"image": self.cnet_upload_img_inpaint, "mask": self.cnet_upload_mask_inpaint.convert("L")})

    def set_p_value(self, p, attr: str, idx: int, v):
        value = getattr(p, attr, None)
        if isinstance(value, list):
            value[idx] = v
        else:
            value = [value] * max_cn_num()
            value[idx] = v
        setattr(p, attr, value)