#ifndef YOLOV8_TENSORRT_INFERENCE_H
#define YOLOV8_TENSORRT_INFERENCE_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct Detection {
    int classId{0};
    float confidence{0.0F};
    cv::Rect boundingBox{};
};

struct FrameTiming {
    double preprocessMilliseconds{0.0};
    double inferenceMilliseconds{0.0};
    double gpuPipelineMilliseconds{0.0};
    double postprocessMilliseconds{0.0};
    double endToEndMilliseconds{0.0};
};

struct InferenceResult {
    std::vector<Detection> detections;
    FrameTiming timing;
};

struct EngineBuildOptions {
    bool enableFp16{false};
    std::size_t workspaceBytes{1ULL << 30U};
};

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override;
};

template <typename TensorRtType>
struct TensorRtDeleter {
    void operator()(TensorRtType* object) const noexcept
    {
        delete object;
    }
};

class TensorRtYoloV8 {
public:
    TensorRtYoloV8(const std::string& enginePath, int gpuDeviceIndex);
    ~TensorRtYoloV8();

    TensorRtYoloV8(const TensorRtYoloV8&) = delete;
    TensorRtYoloV8& operator=(const TensorRtYoloV8&) = delete;

    static void buildEngine(
        const std::string& onnxPath,
        const std::string& enginePath,
        const EngineBuildOptions& options);

    InferenceResult infer(
        const cv::Mat& image,
        float confidenceThreshold,
        float iouThreshold);

    int inputWidth() const noexcept;
    int inputHeight() const noexcept;

private:
    struct LetterboxTransform {
        float scale{1.0F};
        float paddingX{0.0F};
        float paddingY{0.0F};
    };

    void loadEngine(const std::string& enginePath);
    void initializeBindings();
    void enqueueInference();
    LetterboxTransform preprocess(const cv::Mat& image);
    std::vector<Detection> postprocess(
        const cv::Size& originalImageSize,
        const LetterboxTransform& transform,
        float confidenceThreshold,
        float iouThreshold) const;

    static std::size_t calculateVolume(const nvinfer1::Dims& dimensions);

    TensorRtLogger logger_;
    int gpuDeviceIndex_{0};
    std::unique_ptr<nvinfer1::IRuntime, TensorRtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtDeleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRtDeleter<nvinfer1::IExecutionContext>> context_;

    std::string inputTensorName_;
    std::string outputTensorName_;
    nvinfer1::Dims inputDimensions_{};
    nvinfer1::Dims outputDimensions_{};
    int inputBindingIndex_{-1};
    int outputBindingIndex_{-1};

    std::vector<float> hostInput_;
    std::vector<float> hostOutput_;
    void* deviceInput_{nullptr};
    void* deviceOutput_{nullptr};
    std::vector<void*> deviceBindings_;

    cudaStream_t stream_{nullptr};
    cudaEvent_t pipelineStartEvent_{nullptr};
    cudaEvent_t inferenceStartEvent_{nullptr};
    cudaEvent_t inferenceEndEvent_{nullptr};
    cudaEvent_t pipelineEndEvent_{nullptr};
};

#endif
