from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE, DynamicMeanVFE_grad
from .dynamic_pillar_vfe import DynamicPillarVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynMeanVFE_grad': DynamicMeanVFE_grad,
    'DynPillarVFE': DynamicPillarVFE,
}
