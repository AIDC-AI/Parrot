import os
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SigLipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    assert os.path.exists(vision_tower), f"Local path {vision_tower} for vision tower doesn't exist"

    if '/clip-vit' in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif '/siglip-so400m' in vision_tower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
