import json
import time

import onnx
import onnxruntime
import torch
from torchvision import models, transforms
from tqdm import tqdm

torch.backends.quantized.engine = "qnnpack"

N_IMGS = 100
MODEL_PATH = "mobilenet_v2.onnx"

# Dataset
preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
imgs = [preprocess(torch.randn((3, 224, 224))).unsqueeze(0) for i in range(N_IMGS)]


def make_mobilenet_v2_onnx(quantized: bool, jit: bool):
    if quantized:
        net = models.quantization.mobilenet_v2(
            weights=models.quantization.MobileNet_V2_QuantizedWeights.DEFAULT,
            quantize=True,
        )
    else:
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if jit:
        net = torch.jit.script(net)

    torch.onnx.export(
        net,  # model being run
        imgs[0],  # model input (or a tuple for multiple inputs)
        MODEL_PATH,  # where to save the model (can be a file or file-like object)
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )

    onnx_model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(onnx_model)


def benchmark(n_runs: int, profile: bool, prefix: str):
    imgs_np = [img.detach().cpu().numpy() for img in imgs]

    # ONNX session
    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_profiling = profile
    sess_options.profile_file_prefix = f"onnx-profile-{prefix}"

    ort_session = onnxruntime.InferenceSession(
        MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    timings = []
    with torch.inference_mode():
        for i in tqdm(range(n_runs)):
            start = time.perf_counter()
            ort_inputs = {ort_session.get_inputs()[0].name: imgs_np[i % len(imgs_np)]}
            _ort_outs = ort_session.run(None, ort_inputs)
            elapsed = time.perf_counter() - start

            timings.append(elapsed)

    print(f"Mean FPS {len(timings) / sum(timings)}")

    prof_path = ort_session.end_profiling()
    print("ORT profile saved to", prof_path)

    if profile is False:
        with open(f"onnx-timings-{prefix}.json", "w") as f:
            json.dump(timings, f)
        print("Timings saved to", f"onnx-timings-{prefix}.json")


if __name__ == "__main__":
    prefix = "noquantized-nojit"
    make_mobilenet_v2_onnx(quantized=False, jit=False)
    benchmark(300, True, prefix)

    # prefix = "quantized-jit"
    # make_mobilenet_v2_onnx(quantized=True, jit=True)
    # benchmark(1000, False, prefix)

    # prefix = "quantized-nojit"
    # make_mobilenet_v2_onnx(quantized=True, jit=False)
    # benchmark(1000, False, prefix)

    # prefix = "noquantized-jit"
    # make_mobilenet_v2_onnx(quantized=False, jit=True)
    # benchmark(1000, False, prefix)

    # prefix = "noquantized-nojit"
    # make_mobilenet_v2_onnx(quantized=False, jit=False)
    # benchmark(1000, False, prefix)
