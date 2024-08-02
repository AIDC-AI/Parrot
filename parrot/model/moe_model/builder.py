from .moe import MultilingualMoE


def build_moe(config, delay_load=False, **kwargs):
    return MultilingualMoE(config)
