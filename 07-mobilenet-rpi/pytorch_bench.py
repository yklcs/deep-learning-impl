import time
import torch
from torchvision import models, transforms
# ---------- Change Begin ----------
import torch.autograd.profiler as profiler
# ----------- Change End -----------
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
# ---------- Change Begin ----------
with profiler.profile(with_stack=True, profile_memory=True) as prof:
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

print(prof.key_averages(group_by_stack_n=10).table(sort_by='self_cpu_time_total', row_limit=5))
# ----------- Change End -----------