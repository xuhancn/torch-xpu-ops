import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (32, 32, 1024, 1024, 512, 512),
    (32, 4, 128, 128, 64, 64),
    (1, 3, 1200, 1200, 600, 600),
    (512, 512, 28, 28, 14, 14),
]


def fmp2d(shape, dtype, channels_last, backward):
    N, C, H, W, oH, oW = shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]

    if channels_last:
        input = (
            torch.randn(N, C, H, W)
            .to(memory_format=torch.channels_last)
            .to(device="xpu", dtype=dtype)
        )
    else:
        input = torch.randn(N, C, H, W).to(device="xpu", dtype=dtype)

    if backward:
        input.requires_grad_(True)
        grad = torch.randn([N, C, oH, oW]).to(device="xpu", dtype=dtype)

    fmp = torch.nn.FractionalMaxPool2d(2, output_size=(oH, oW), return_indices=True)

    output = fmp(input)

    if backward:
        output[0].backward(grad)


if __name__ == "__main__":
    backward = True
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                # warm up
                fmp2d(shape, dtype, channels_last, backward)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3]),
                    "; datatype:",
                    dtype,
                    "; channels_last:",
                    channels_last,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                    record_shapes=True,
                ) as prof:
                    for i in range(20):
                        fmp2d(shape, dtype, channels_last, backward)
                print(prof.key_averages().table(sort_by="xpu_time_total"))
