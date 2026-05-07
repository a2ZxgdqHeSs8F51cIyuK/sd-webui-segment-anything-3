# sd-webui-segment-anything-3
An update of the sd-webui-segment-anything extension (https://github.com/continue-revolution/sd-webui-segment-anything) using SAM3 and updated to work with ForgeNEO
SAM 2 Version for ReForge: https://github.com/a2ZxgdqHeSs8F51cIyuK/sd-webui-segment-anything-2/

Necessary downloads:
- Put bpe_simple_vocab_16e6.txt.gz (from https://github.com/openai/CLIP/tree/main/clip) inside models
- Put sam3.pt (from https://huggingface.co/facebook/sam3) inside models/sam

Things that don't work yet:
- Point prompts
- Controlnet is not tested at all
- Automatically grabbing the inpaint image if no separate image is provided in the extension's menu doesn't work at the moment
