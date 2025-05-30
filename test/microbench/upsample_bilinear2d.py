import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
cache_r = torch.randn((1024 * 1024 * 1024), device="xpu")
cache_w = torch.randn((1024 * 1024 * 1024), device="xpu")


def simple_test(in_shape, scale_factor, backward, dtype, mode):
    in_tensor = torch.randn(
        in_shape, dtype=dtype, device=device, requires_grad=backward
    )
    output = torch.nn.functional.interpolate(
        in_tensor, mode=mode, scale_factor=scale_factor
    )

    # warm_up
    for _ in range(10):
        output = torch.nn.functional.interpolate(
            in_tensor, mode=mode, scale_factor=scale_factor
        )

    # go
    print(
        "shape:",
        (in_shape),
        "; datatype:",
        dtype,
        "; scale_factor:",
        scale_factor,
        "; mode:",
        mode,
        "; backward:",
        backward,
    )
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
    ) as prof:
        for i in range(20):
            cache_r = cache_w + 1
            output = torch.nn.functional.interpolate(
                in_tensor,
                mode=mode,
                scale_factor=scale_factor,
            )
            if backward:
                output = torch.autograd.grad(
                    output, in_tensor, grad_outputs=torch.ones_like(output)
                )
    print(prof.key_averages().table(sort_by="xpu_time_total"))


shape_list = [
    [1, 3, 1200, 1200],
    [1, 128, 1200, 1200],
    [1, 3, 1200, 1200],
    [128, 128, 5, 5],
    [8, 32, 256, 256],
]
scale_factor = [[3, 3], [3, 3], [7, 7], [7, 7], 3]
for sp, sf in zip(shape_list, scale_factor):
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for mode in ["bilinear"]:
            simple_test(sp, sf, backward, dtype, mode)
