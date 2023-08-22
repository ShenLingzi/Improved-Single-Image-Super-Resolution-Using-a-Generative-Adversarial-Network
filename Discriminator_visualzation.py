import netron
import torch
from model import Discriminator

model = Discriminator(4).eval()
dummy_input = torch.randn(1, 3, 256, 256)
modelPath = 'E:/Ds/Project/DIV2K-v1/epochs/netD_epoch_4_100.pth'
model.load_state_dict(torch.load(modelPath))

torch.onnx.export(model, dummy_input, "E:/Ds/Project/DIV2K-v1/epochs/pth2onnx.onnx",verbose=True)
netron.start("E:/Ds/Project/DIV2K-v1/epochs/Discriminator.onnx")