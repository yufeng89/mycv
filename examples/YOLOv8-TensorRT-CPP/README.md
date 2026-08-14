# YOLOv8 TensorRT C++ inference

This example converts `best.onnx` to a TensorRT engine on the current GPU and
then runs native C++ YOLOv8 inference. It reports preprocessing, GPU
inference, GPU pipeline, postprocessing, and end-to-end single-frame latency.

## Requirements

- NVIDIA driver and CUDA Toolkit
- TensorRT development package (`NvInfer.h`, `NvOnnxParser.h`, `libnvinfer`, `libnvonnxparser`)
- OpenCV development package
- CMake 3.16 or newer and a C++17 compiler

TensorRT engines are specific to the TensorRT version, GPU architecture, precision, and often the CUDA version. Build the engine on the deployment machine rather than copying an engine generated elsewhere.

## Configure CUDA and TensorRT

The following paths match the x86 installation in this workspace. Run them in
the same shell that will configure, build, and run the program:

```bash
export TensorRT_Lib=/home/yufeng/project/study/TensorRT-8.5.1.7/lib
export TensorRT_Inc=/home/yufeng/project/study/TensorRT-8.5.1.7/include
export TensorRT_Bin=/home/yufeng/project/study/TensorRT-8.5.1.7/bin

export CUDA_Lib=/usr/local/cuda/lib64
export CUDA_Inc=/usr/local/cuda/include
export CUDA_Bin=/usr/local/cuda/bin
export CUDA_HOME=/usr/local/cuda

export CUDNN_Lib=/usr/local/cuda/lib64
export PATH="$CUDA_Bin:$TensorRT_Bin:$PATH"
export LD_LIBRARY_PATH="$TensorRT_Lib:$CUDA_Lib:${LD_LIBRARY_PATH:-}"
```

`CMakeLists.txt` reads `CUDA_HOME`, `TensorRT_Inc`, `TensorRT_Lib`, and
`CUDA_Lib`, so `TensorRT_ROOT` does not need to be set separately.

## Build

```bash
cd /home/yufeng/project/study/examples/YOLOv8-TensorRT-CPP
cmake -S . -B build \
      -DTensorRT_ROOT=/home/yufeng/project/study/TensorRT-8.5.1.7 \
      -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
cmake --build build --parallel
```

The explicit `-D` options are optional when the environment variables above
are present. They are shown to make the selected CUDA and TensorRT versions
unambiguous.

## Convert `best.onnx` and run inference

When `best.engine` does not exist, the executable parses `best.onnx`, builds a
TensorRT engine, saves it, and continues with inference. Therefore, the normal
command is:

```bash
build/yolov8_tensorrt --build-only --fp16
```

This converts the default `best.onnx` to `best.engine` without requiring an
input image. Add `--rebuild` when `best.engine` already exists and must be
regenerated.

After the engine is available, run inference with:

```bash
yolov8_tensorrt_command=build/yolov8_tensorrt
"$yolov8_tensorrt_command" \
  --image /path/to/test.jpg \
  --fp16 \
  --warmup 10 \
  --iterations 100
```

The default paths are `best.onnx`, `best.engine`, and `classes.txt`, relative
to the current working directory. Paths can be overridden explicitly:

```bash
build/yolov8_tensorrt \
  --onnx best.onnx \
  --engine best_fp16.engine \
  --image /path/to/test.jpg \
  --classes classes.txt \
  --output tensorrt_result.jpg \
  --fp16
```

Add `--rebuild` to force rebuilding an existing engine. Remove `--fp16` to
build an FP32 engine. The generated engine must be rebuilt if the GPU,
TensorRT version, or relevant CUDA environment changes.

The timing report distinguishes:

- `preprocess`: CPU letterbox and HWC-to-CHW conversion
- `inference`: TensorRT enqueue measured with CUDA events, excluding copies
- `GPU pipeline`: host-to-device copy, TensorRT inference, and device-to-host copy
- `postprocess`: decode and class-aware NMS on CPU
- `end-to-end`: complete processing for one frame

Warmup runs are excluded from statistics. The annotated image is written only after timing, so file encoding does not affect latency.
