import time
import torch
from torchvision import models, transforms

torch.backends.quantized.engine = 'qnnpack'

# create imageset 
preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
imgs = [preprocess(torch.randn((3, 224, 224))).unsqueeze(0) for i in range(100)]


# create network 
#net = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.DEFAULT)
#net = torch.jit.script(net)

net = models.quantization.mobilenet_v2(weights = models.quantization.MobileNet_V2_QuantizedWeights.DEFAULT, quantize = True)
net = torch.jit.script(net)

# run benchmark 
started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():        
    for i in range(300):
        # run model
        output = net(imgs[i % 100])

        # log model performance
        frame_count += 1
        now = time.time()

        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0

