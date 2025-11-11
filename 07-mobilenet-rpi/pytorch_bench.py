import json
import time

import torch
import torch.autograd.profiler as profiler
from torchvision import models, transforms
from tqdm import tqdm

torch.backends.quantized.engine = "qnnpack"

N_IMGS = 100

preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
imgs = [preprocess(torch.randn((3, 224, 224))).unsqueeze(0) for i in range(N_IMGS)]


def make_mobilenet_v2(quantized: bool, jit: bool):
    if quantized:
        net = models.quantization.mobilenet_v2(
            weights=models.quantization.MobileNet_V2_QuantizedWeights.DEFAULT,
            quantize=True,
        )
    else:
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if jit:
        net = torch.jit.script(net)

    return net


@torch.inference_mode()
def benchmark(net, n_runs: int, profile: bool, prefix: str):
    timings = []
    with profiler.profile(
        enabled=profile, with_stack=True, profile_memory=True
    ) as prof:
        for i in tqdm(range(n_runs)):
            start = time.perf_counter()
            _output = net(imgs[i % len(imgs)])
            elapsed = time.perf_counter() - start

            timings.append(elapsed)

    if prof is not None:
        print(
            prof.key_averages(group_by_stack_n=10).table(
                sort_by="self_cpu_time_total", row_limit=5
            )
        )
        prof.export_chrome_trace(f"pytorch-profile-{prefix}.json")

    print(f"Mean FPS {len(timings) / sum(timings)}")

    if profile is False:
        with open(f"pytorch-timings-{prefix}.json", "w") as f:
            json.dump(timings, f)
        print("Timings saved to", f"pytorch-timings-{prefix}.json")


if __name__ == "__main__":
    prefix = "noquantized-nojit"
    net = make_mobilenet_v2(quantized=False, jit=False)
    benchmark(net, 300, True, prefix)

    # prefix = "quantized-jit"
    # net = make_mobilenet_v2(quantized=True, jit=True)
    # benchmark(net, 1000, False, prefix)

    # prefix = "quantized-nojit"
    # net = make_mobilenet_v2(quantized=True, jit=False)
    # benchmark(net, 1000, False, prefix)

    # prefix = "noquantized-jit"
    # net = make_mobilenet_v2(quantized=False, jit=True)
    # benchmark(net, 1000, False, prefix)

    # prefix = "noquantized-nojit"
    # net = make_mobilenet_v2(quantized=False, jit=False)
    # benchmark(net, 1000, False, prefix)
