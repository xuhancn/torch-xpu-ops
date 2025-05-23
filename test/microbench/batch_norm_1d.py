import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"

shape_list = [((64, 8), (8)), ((4, 128, 15000), (128)), ((4, 256, 512), (256))]

for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    for shape in shape_list:
        backward = True
        # input
        input = torch.randn(shape[0], device=device, dtype=dtype)

        if backward:
            input.requires_grad_(True)

        # warm up
        m = torch.nn.BatchNorm1d(shape[1], device=device)
        output = m(input)

        print(
            "shape:",
            shape[0],
            "; datatype:",
            dtype,
            "; num_features:",
            shape[1],
            "; backward:",
            backward,
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(20):
                m = torch.nn.BatchNorm1d(shape[1], device=device)
                output = m(input)
                if backward:
                    gy = torch.empty_like(output)
                    output.backward(gy)
        print(prof.key_averages().table(sort_by="xpu_time_total"))
