from image_stitching import *


#src: directory of your images
src = "/groups/CaiLab/personal/Lex/raw/250113_mb_BSpeg_xtra_potentialTriton/pyfish_tools/output/spatial_mapped_masks/"

#initialize class
IStitch = ImageStitcher(px_size=0.108)

## Stitch images with segmentation marker or dapi
#IStitch.stitch_images_from_csv(img_dir=src, imgchn=0, stain="spatialmapped", num_channels=1)

## Stitch masks with cell-type definitions
IStitch.stitch_rgb_images_from_csv(img_dir=src, stain="spatialmapped")

