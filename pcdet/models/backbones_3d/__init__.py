from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x, VoxelWideResBackBone8x, VoxelWideResBackBone_L8x
from .spconv_backbone import VoxelBackBone8x_DIR, VoxelBackBone8x_DSR, VoxelBackBone8x_All2D, VoxelBackBone8x_DSR_SubGen
from .spconv_backbone import VoxelBackBone8x_Decoder
from .spconv_backbone_unibn import VoxelBackBone8x_UniBN, VoxelResBackBone8x_UniBN
from .spconv_unet import UNetV2
from .IASSD_backbone import IASSD_Backbone

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelWideResBackBone8x': VoxelWideResBackBone8x,
    'VoxelWideResBackBone_L8x': VoxelWideResBackBone_L8x,
    'VoxelBackBone8x_DIR': VoxelBackBone8x_DIR,
    'VoxelBackBone8x_DSR': VoxelBackBone8x_DSR,
    'VoxelBackBone8x_All2D': VoxelBackBone8x_All2D,
    'VoxelBackBone8x_DSR_SubGen': VoxelBackBone8x_DSR_SubGen,
    'VoxelBackBone8x_Decoder': VoxelBackBone8x_Decoder,
    # Dataset-specific Norm Layer
    'VoxelBackBone8x_UniBN':VoxelBackBone8x_UniBN,
    'VoxelResBackBone8x_UniBN':VoxelResBackBone8x_UniBN,
    'IASSD_Backbone': IASSD_Backbone,
}
