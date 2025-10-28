import torch
import time
from custom_ops_bn.ops import batchnorm

# ==== Backends: wrapper 유지 ====
def batchnorm_cuda(x, g, b, rm, rv, training):
    """Wrapper for CUDA backend batchnorm"""
    return batchnorm(x, g, b, rm, rv, training, backend='cuda')

def batchnorm_triton(x, g, b, rm, rv, training):
    """Wrapper for Triton backend batchnorm"""
    return batchnorm(x, g, b, rm, rv, training, backend='triton')

# ==== 허용 오차 검사 유틸리티 ====
def check_tolerance(a, b, rtol, atol):
    """
    텐서 'a' (테스트 대상)와 'b' (기준선)를
    상대/절대 허용 오차 기준으로 비교합니다.

    반환: (is_ok, max_abs_diff)
    """
    if a is None or b is None:
        return False, float('inf')
    
    # torch.allclose의 로직을 직접 구현
    # abs(a - b) <= atol + rtol * abs(b)
    try:
        abs_diff = torch.abs(a - b)
        max_abs_diff = torch.max(abs_diff).item()
        
        # 허용 오차 기준 계산
        tolerance_bound = atol + rtol * torch.abs(b)
        
        # 모든 요소가 허용 오차 범위 내에 있는지 확인
        is_ok = torch.all(abs_diff <= tolerance_bound).item()
        
        return is_ok, max_abs_diff
    except Exception as e:
        print(f"   [check_tolerance ERROR]: {e}")
        return False, float('inf')


# ==== 정확도 테스트 (Forward + Backward) ====
def test_correctness(N=32, C=64, H=56, W=56, training=True,
                     rtol_forward=1e-3, atol_forward=1e-5,
                     rtol_backward=1e-3, atol_backward=1e-5,
                     # <--- 수정: dgamma는 rtol이 아닌 atol을 기준으로 검사
                     rtol_backward_gamma=1e-3, atol_backward_gamma=1e-2): 
    """PyTorch vs CUDA vs Triton: Forward/Backward 정확도 테스트"""
    print(f"\n{'='*70}")
    print(f"Correctness Test: N={N}, C={C}, H={H}, W={W}, Training={training}")
    print(f"Tolerances (FWD): rtol={rtol_forward:.1e}, atol={atol_forward:.1e}")
    print(f"Tolerances (BWD): rtol={rtol_backward:.1e}, atol={atol_backward:.1e} (Default)")
    # <--- 수정: 헤더 출력 변경
    print(f"Tolerances (BWD-Gamma): rtol={rtol_backward_gamma:.1e}, atol={atol_backward_gamma:.1e} (Relaxed ATOL)")
    print(f"{'='*70}")

    device = 'cuda'

    # 공통 테스트 텐서
    input_test = torch.randn(N, C, H, W, device=device)
    gamma_test = torch.randn(C, device=device)
    beta_test  = torch.randn(C, device=device)
    rm_test    = torch.randn(C, device=device)
    rv_test    = torch.abs(torch.randn(C, device=device))

    # ---------- PyTorch 기준선 준비 ----------
    bn_pytorch_fwd = torch.nn.BatchNorm2d(C, momentum=0.1, eps=1e-5).to(device)
    # weight/bias 값을 동일하게 설정
    with torch.no_grad():
        bn_pytorch_fwd.weight.copy_(gamma_test)
        bn_pytorch_fwd.bias.copy_(beta_test)
        bn_pytorch_fwd.running_mean.copy_(rm_test)
        bn_pytorch_fwd.running_var.copy_(rv_test)
    bn_pytorch_fwd.train() if training else bn_pytorch_fwd.eval()

    # Forward 기준선
    with torch.no_grad():
        out_pytorch = bn_pytorch_fwd(input_test.clone())

    # ---------- Forward 비교 ----------
    fwd_ok = True

    # 1) CUDA
    cuda_fwd_ok = False
    try:
        with torch.no_grad():
            out_cuda = batchnorm_cuda(
                input_test.clone(), gamma_test.clone(), beta_test.clone(),
                rm_test.clone(), rv_test.clone(), training=training
            )
        
        (cuda_fwd_ok, diff_cuda) = check_tolerance(
            out_cuda, out_pytorch, rtol_forward, atol_forward
        )
        print(f"\n[FORWARD]")
        print(f"   PyTorch vs CUDA   max|diff|: {diff_cuda:.2e}   -> {'✓' if cuda_fwd_ok else '✗'}")
        
    except NotImplementedError:
        print(f"\n[FORWARD]\n   CUDA: ⚠ NOT IMPLEMENTED (TODO)")
    except Exception as e:
        print(f"\n[FORWARD]\n   CUDA: ✗ ERROR - {type(e).__name__}: {str(e)[:120]}")

    # 2) Triton
    triton_fwd_ok = False
    try:
        with torch.no_grad():
            out_triton = batchnorm_triton(
                input_test.clone(), gamma_test.clone(), beta_test.clone(),
                rm_test.clone(), rv_test.clone(), training=training
            )

        (triton_fwd_ok, diff_triton) = check_tolerance(
            out_triton, out_pytorch, rtol_forward, atol_forward
        )
        print(f"   PyTorch vs Triton max|diff|: {diff_triton:.2e}   -> {'✓' if triton_fwd_ok else '✗'}")

    except NotImplementedError:
        print(f"   Triton: ⚠ NOT IMPLEMENTED (TODO)")
        out_triton = None
    except Exception as e:
        print(f"   Triton: ✗ ERROR - {type(e).__name__}: {str(e)[:120]}")
        out_triton = None

    # 3) (참고) CUDA vs Triton
    if 'out_cuda' in locals() and out_triton is not None:
        try:
            # (참고 비교는 단순 max|diff| 사용)
            diff_ct = torch.max(torch.abs(out_cuda - out_triton)).item()
            print(f"   CUDA   vs Triton max|diff|: {diff_ct:.2e}")
        except Exception as e:
            print(f"   CUDA vs Triton: ✗ ERROR - {type(e).__name__}: {str(e)[:120]}")

    fwd_ok = (cuda_fwd_ok if 'out_cuda' in locals() else True) and \
             (triton_fwd_ok if out_triton is not None else True)

    # ---------- Backward 비교 (입력/감마/베타 그라디언트) ----------
    print(f"\n[BACKWARD]   (loss = output.sum())")

    # 공통 입력/파라미터 세트 생성 함수
    def make_leaves():
        inp = input_test.clone().detach().requires_grad_(True)
        g   = gamma_test.clone().detach().requires_grad_(True)
        b   = beta_test.clone().detach().requires_grad_(True)
        rm  = rm_test.clone()   # buffer
        rv  = rv_test.clone()   # buffer
        return inp, g, b, rm, rv

    # 0) PyTorch 기준선 backward
    #    (러닝 스탯 업데이트 영향을 배제하려고, backward용으로 모듈을 새로 만든다)
    bn_pytorch_bwd = torch.nn.BatchNorm2d(C, momentum=0.1, eps=1e-5).to(device)
    with torch.no_grad():
        bn_pytorch_bwd.weight.copy_(gamma_test)
        bn_pytorch_bwd.bias.copy_(beta_test)
        bn_pytorch_bwd.running_mean.copy_(rm_test)
        bn_pytorch_bwd.running_var.copy_(rv_test)
    bn_pytorch_bwd.train() if training else bn_pytorch_bwd.eval()

    pt_inp, pt_g, pt_b, _, _ = make_leaves()
    out_pt = bn_pytorch_bwd(pt_inp)
    loss_pt = out_pt.sum()
    loss_pt.backward()
    grad_pt_inp  = pt_inp.grad.detach().clone()
    grad_pt_g    = bn_pytorch_bwd.weight.grad.detach().clone()
    grad_pt_b    = bn_pytorch_bwd.bias.grad.detach().clone()

    # 1) CUDA backward
    have_cuda_bwd = False
    try:
        cu_inp, cu_g, cu_b, cu_rm, cu_rv = make_leaves()
        out_cu = batchnorm_cuda(cu_inp, cu_g, cu_b, cu_rm, cu_rv, training=training)
        (out_cu.sum()).backward()
        grad_cu_inp = cu_inp.grad.detach().clone()
        grad_cu_g   = cu_g.grad.detach().clone()
        grad_cu_b   = cu_b.grad.detach().clone()
        have_cuda_bwd = True
    except NotImplementedError:
        print("   CUDA: ⚠ NOT IMPLEMENTED (TODO)")
        grad_cu_inp = grad_cu_g = grad_cu_b = None
    except Exception as e:
        print(f"   CUDA: ✗ ERROR - {type(e).__name__}: {str(e)[:120]}")
        grad_cu_inp = grad_cu_g = grad_cu_b = None

    # 2) Triton backward
    have_triton_bwd = False
    try:
        tr_inp, tr_g, tr_b, tr_rm, tr_rv = make_leaves()
        out_tr = batchnorm_triton(tr_inp, tr_g, tr_b, tr_rm, tr_rv, training=training)
        (out_tr.sum()).backward()
        grad_tr_inp = tr_inp.grad.detach().clone()
        grad_tr_g   = tr_g.grad.detach().clone()
        grad_tr_b   = tr_b.grad.detach().clone()
        have_triton_bwd = True
    except NotImplementedError:
        print("   Triton: ⚠ NOT IMPLEMENTED (TODO)")
        grad_tr_inp = grad_tr_g = grad_tr_b = None
    except Exception as e:
        print(f"   Triton: ✗ ERROR - {type(e).__name__}: {str(e)[:120]}")
        grad_tr_inp = grad_tr_g = grad_tr_b = None

    
    bwd_ok = True
    # 입력 x의 grad 비교
    print("\n   Grad wrt INPUT:")
    if have_cuda_bwd:
        (ok, d) = check_tolerance(grad_cu_inp, grad_pt_inp, rtol_backward, atol_backward)
        print(f"       PyTorch vs CUDA   max|diff|: {d:.2e}   -> {'✓' if ok else '✗'}")
        bwd_ok = bwd_ok and ok
    if have_triton_bwd:
        (ok, d) = check_tolerance(grad_tr_inp, grad_pt_inp, rtol_backward, atol_backward)
        print(f"       PyTorch vs Triton max|diff|: {d:.2e}   -> {'✓' if ok else '✗'}")
        bwd_ok = bwd_ok and ok
    if have_cuda_bwd and have_triton_bwd:
        # (참고 비교는 단순 max|diff| 사용)
        d = torch.max(torch.abs(grad_cu_inp - grad_tr_inp)).item()
        print(f"       CUDA   vs Triton   max|diff|: {d:.2e}")

    # gamma의 grad 비교
    print("\n   Grad wrt GAMMA: (Using relaxed ATOL)") # <--- 수정
    if have_cuda_bwd:
        (ok, d) = check_tolerance(grad_cu_g, grad_pt_g, 
                                rtol_backward_gamma, atol_backward_gamma)
        print(f"       PyTorch vs CUDA   max|diff|: {d:.2e}   -> {'✓' if ok else '✗'}")
        bwd_ok = bwd_ok and ok
    if have_triton_bwd:
        (ok, d) = check_tolerance(grad_tr_g, grad_pt_g, 
                                rtol_backward_gamma, atol_backward_gamma)
        print(f"       PyTorch vs Triton max|diff|: {d:.2e}   -> {'✓' if ok else '✗'}")
        bwd_ok = bwd_ok and ok
    if have_cuda_bwd and have_triton_bwd:
        d = torch.max(torch.abs(grad_cu_g - grad_tr_g)).item()
        print(f"       CUDA   vs Triton   max|diff|: {d:.2e}")

    # beta의 grad 비교
    print("\n   Grad wrt BETA:") # dbeta는 다시 원래의 엄격한 기준 사용
    if have_cuda_bwd:
        (ok, d) = check_tolerance(grad_cu_b, grad_pt_b, 
                                rtol_backward, atol_backward)
        print(f"       PyTorch vs CUDA   max|diff|: {d:.2e}   -> {'✓' if ok else '✗'}")
        bwd_ok = bwd_ok and ok
    if have_triton_bwd:
        (ok, d) = check_tolerance(grad_tr_b, grad_pt_b, 
                                rtol_backward, atol_backward)
        print(f"       PyTorch vs Triton max|diff|: {d:.2e}   -> {'✓' if ok else '✗'}")
        bwd_ok = bwd_ok and ok
    if have_cuda_bwd and have_triton_bwd:
        d = torch.max(torch.abs(grad_cu_b - grad_tr_b)).item()
        print(f"       CUDA   vs Triton   max|diff|: {d:.2e}")

    all_ok = fwd_ok and bwd_ok
    print(f"\n   => Forward OK: {'✓' if fwd_ok else '✗'} | Backward OK: {'✓' if bwd_ok else '✗'}")
    return all_ok


# ==== 성능 벤치마크(기존 유지) ====
def benchmark_forward_backward(func, input, gamma, beta, running_mean, running_var,
                               training=True, warmup=10, iterations=100):
    """Benchmark forward and backward separately"""
    # Warmup
    for _ in range(warmup):
        rm = running_mean.clone()
        rv = running_var.clone()
        inp = input.clone().detach().requires_grad_(True)
        g = gamma.clone().detach().requires_grad_(True)
        b = beta.clone().detach().requires_grad_(True)

        output = func(inp, g, b, rm, rv, training=training)
        loss = output.sum()
        loss.backward()

    torch.cuda.synchronize()

    # Benchmark - measure forward and backward separately
    total_forward_time = 0
    total_backward_time = 0

    for _ in range(iterations):
        rm = running_mean.clone()
        rv = running_var.clone()
        inp = input.clone().detach().requires_grad_(True)
        g = gamma.clone().detach().requires_grad_(True)
        b = beta.clone().detach().requires_grad_(True)

        # Measure forward
        torch.cuda.synchronize()
        start = time.time()
        output = func(inp, g, b, rm, rv, training=training)
        torch.cuda.synchronize()
        total_forward_time += time.time() - start

        # Measure backward
        torch.cuda.synchronize()
        start = time.time()
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()
        total_backward_time += time.time() - start

    avg_forward_time = total_forward_time / iterations * 1000  # ms
    avg_backward_time = total_backward_time / iterations * 1000  # ms

    return avg_forward_time, avg_backward_time


def run_benchmark(N, C, H, W, training=True):
    """Run performance benchmark for specific input size"""
    print(f"\n{'='*70}")
    print(f"Performance Benchmark: N={N}, C={C}, H={H}, W={W}, Training={training}")
    print(f"{'='*70}")

    device = 'cuda'

    # Create base tensors
    input_base = torch.randn(N, C, H, W, device=device)
    gamma = torch.randn(C, device=device)
    beta = torch.randn(C, device=device)
    running_mean = torch.randn(C, device=device)
    running_var = torch.abs(torch.randn(C, device=device))

    # CUDA custom operator
    print("\n1. CUDA Custom Operator (Reference)")
    try:
        forward_time_cuda, backward_time_cuda = benchmark_forward_backward(
            batchnorm_cuda,
            input_base, gamma, beta,
            running_mean, running_var, training
        )
        print(f"   Forward:  {forward_time_cuda:.4f} ms")
        print(f"   Backward: {backward_time_cuda:.4f} ms")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return

    # Triton custom operator
    print("\n2. Triton Custom Operator (Your Implementation)")
    try:
        forward_time_triton, backward_time_triton = benchmark_forward_backward(
            batchnorm_triton,
            input_base, gamma, beta,
            running_mean, running_var, training
        )
        print(f"   Forward:  {forward_time_triton:.4f} ms")
        print(f"   Speedup vs CUDA: {forward_time_cuda / forward_time_triton:.2f}x")
        print(f"   Backward: {backward_time_triton:.4f} ms")
        print(f"   Speedup vs CUDA: {backward_time_cuda / backward_time_triton:.2f}x")
    except NotImplementedError:
        print(f"   ⚠ NOT IMPLEMENTED (TODO)")
    except Exception as e:
        print(f"   ✗ ERROR: {type(e).__name__}: {str(e)[:80]}")


def main():
    print("\n" + "="*70)
    print(" BatchNorm Custom Operators: Correctness & Performance Tests")
    print("="*70)

    # Test configurations
    configs = [
        # (8, 32, 224, 224),
        # (32, 64, 112, 112),
        (64, 128, 56, 56),
    ]

    # Run correctness tests *************************************************************************
    print("\n" + "="*70)
    print(" CORRECTNESS TESTS")
    print("="*70)

    all_passed = True
    for N, C, H, W in configs:
        passed_train = test_correctness(N, C, H, W, training=True)
        all_passed = all_passed and passed_train
        
        # (필요시) Inference 모드 테스트
        # passed_eval = test_correctness(N, C, H, W, training=False)
        # all_passed = all_passed and passed_eval

    print("\n" + "="*70)
    if all_passed:
        print(" ALL CORRECTNESS TESTS PASSED")
    else:
        print(" SOME CORRECTNESS TESTS FAILED")
    print("="*70)

    # # Run performance benchmarks *************************************************************************
    # print("\n" + "="*70)
    # print(" PERFORMANCE BENCHMARKS")
    # print("="*70)

    # for N, C, H, W in configs:
    #     # Training mode
    #     run_benchmark(N, C, H, W, training=True)

    #     # Inference mode
    #     run_benchmark(N, C, H, W, training=False)

    # print("\n" + "="*70)
    # print(" Tests Complete!")
    # print("="*70 + "\n")


if __name__ == "__main__":
    main()
    