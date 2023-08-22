import netron
import torch
from model import Generator

model = Generator(4).eval()
dummy_input = torch.randn(1, 3, 64, 64)
modelPath = 'E:/Ds/Project/models/Extension/1/netG_epoch_4_1000.pth'
model.load_state_dict(torch.load(modelPath))

torch.onnx.export(model, dummy_input, "E:/Ds/Project/models/Extension/1/Generator.onnx",verbose=True)
netron.start("E:/Ds/Project/models/Extension/1/Generator.onnx")
