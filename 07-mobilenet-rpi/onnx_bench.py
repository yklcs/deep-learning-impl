import time
import torch
from torchvision import models, transforms

torch.backends.quantized.engine = 'qnnpack'

# create imageset 
preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


imgs = [preprocess(torch.randn((3, 224, 224))).unsqueeze(0) for i in range(100)]

#net = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.DEFAULT)
#net = torch.jit.script(net)

net = models.quantization.mobilenet_v2(weights = models.quantization.MobileNet_V2_QuantizedWeights.DEFAULT, quantize = True)
net = torch.jit.script(net)


# export ONNX model 
import onnx
torch.onnx.export(net,                      # model being run
                  imgs[0],                  # model input (or a tuple for multiple inputs)
                  "MV2.onnx",               # where to save the model (can be a file or file-like object)                  
                  do_constant_folding=True, # whether to execute constant folding for optimization
                  input_names = ['input'],  # the model's input names
                  output_names = ['output'],# the model's output names
                )               

onnx_model = onnx.load("MV2.onnx")
onnx.checker.check_model(onnx_model)


# load ONNX model 
import onnxruntime

# ---------- Change Begin ----------
# Profiling 
sess_options = onnxruntime.SessionOptions()
sess_options.enable_profiling = True

ort_session = onnxruntime.InferenceSession("MV2.onnx", sess_options=sess_options, providers=["CPUExecutionProvider"])
# ----------- Change End -----------

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

imgs = [to_numpy(imgs[i]) for i in range(100)]


# run benchmark 
started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():        
    for i in range(300):
        # run model
        ort_inputs = {ort_session.get_inputs()[0].name: imgs[i % 100]}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # log model performance
        frame_count += 1
        now = time.time()

        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0

# ---------- Change Begin ----------
prof_path = ort_session.end_profiling()
print("ORT profile saved to:", prof_path)
# ----------- Change End -----------