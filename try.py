from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow, write_flow
import mmcv

# Specify the path to model config and checkpoint file
config_file = './flownet2css_8x1_sfine_flyingthings3d_subset_384x768.py'
checkpoint_file = 'flownet2css_8x1_sfine_flyingthings3d_subset_384x768.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test image pair, and save the results
img1='./demo/image000.png'
img2='./transform_image/15_image000.png'
result = inference_model(model, img1, img2)
# save the optical flow file
write_flow(result, flow_file='flow.flo')
# save the visualized flow map
flow_map = visualize_flow(result, save_file='flow_map.png')