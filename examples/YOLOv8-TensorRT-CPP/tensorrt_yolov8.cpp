#include "tensorrt_yolov8.h"

#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

void checkCuda(cudaError_t status, const std::string& operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(operation + " failed: " + cudaGetErrorString(status));
    }
}

void requireCondition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<char> readBinaryFile(const std::string& filePath)
{
    std::ifstream inputFile(filePath, std::ios::binary | std::ios::ate);
    if (!inputFile) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    const std::streamsize fileSize = inputFile.tellg();
    requireCondition(fileSize > 0, "File is empty: " + filePath);
    inputFile.seekg(0, std::ios::beg);

    std::vector<char> fileContents(static_cast<std::size_t>(fileSize));
    requireCondition(
        static_cast<bool>(inputFile.read(fileContents.data(), fileSize)),
        "Unable to read file: " + filePath);
    return fileContents;
}

void writeBinaryFile(const std::string& filePath, const void* data, std::size_t size)
{
    const std::filesystem::path outputPath(filePath);
    if (!outputPath.parent_path().empty()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    std::ofstream outputFile(filePath, std::ios::binary);
    if (!outputFile) {
        throw std::runtime_error("Unable to create file: " + filePath);
    }
    outputFile.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    requireCondition(static_cast<bool>(outputFile), "Unable to write file: " + filePath);
}

bool dimensionsAreStatic(const nvinfer1::Dims& dimensions)
{
    for (int dimensionIndex = 0; dimensionIndex < dimensions.nbDims; ++dimensionIndex) {
        if (dimensions.d[dimensionIndex] <= 0) {
            return false;
        }
    }
    return true;
}

double elapsedMilliseconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct DetectionCandidate {
    int classId{0};
    float confidence{0.0F};
    cv::Rect boundingBox{};
};

}  // namespace

void TensorRtLogger::log(Severity severity, const char* message) noexcept
{
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TensorRT] " << message << '\n';
    }
}

void TensorRtYoloV8::buildEngine(
    const std::string& onnxPath,
    const std::string& enginePath,
    const EngineBuildOptions& options)
{
    if (!std::filesystem::is_regular_file(onnxPath)) {
        throw std::runtime_error("ONNX model does not exist: " + onnxPath);
    }

    TensorRtLogger logger;
    std::unique_ptr<nvinfer1::IBuilder, TensorRtDeleter<nvinfer1::IBuilder>> builder(
        nvinfer1::createInferBuilder(logger));
    requireCondition(builder != nullptr, "Unable to create TensorRT builder");

    const std::uint32_t networkFlags =
        1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition, TensorRtDeleter<nvinfer1::INetworkDefinition>> network(
        builder->createNetworkV2(networkFlags));
    requireCondition(network != nullptr, "Unable to create TensorRT network");

    std::unique_ptr<nvonnxparser::IParser, TensorRtDeleter<nvonnxparser::IParser>> parser(
        nvonnxparser::createParser(*network, logger));
    requireCondition(parser != nullptr, "Unable to create TensorRT ONNX parser");

    std::cout << "Parsing ONNX model: " << onnxPath << '\n';
    const bool parsedSuccessfully = parser->parseFromFile(
        onnxPath.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsedSuccessfully) {
        std::string parserErrors;
        for (int errorIndex = 0; errorIndex < parser->getNbErrors(); ++errorIndex) {
            parserErrors += "\n  - ";
            parserErrors += parser->getError(errorIndex)->desc();
        }
        throw std::runtime_error("TensorRT could not parse the ONNX model:" + parserErrors);
    }

    std::unique_ptr<nvinfer1::IBuilderConfig, TensorRtDeleter<nvinfer1::IBuilderConfig>> builderConfig(
        builder->createBuilderConfig());
    requireCondition(builderConfig != nullptr, "Unable to create TensorRT builder configuration");
    builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, options.workspaceBytes);

    if (options.enableFp16) {
        requireCondition(builder->platformHasFastFp16(), "The selected GPU does not support fast FP16 execution");
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    const auto buildStart = std::chrono::steady_clock::now();
    std::cout << "Building " << (options.enableFp16 ? "FP16" : "FP32")
              << " TensorRT engine. This can take several minutes...\n";
    std::unique_ptr<nvinfer1::IHostMemory, TensorRtDeleter<nvinfer1::IHostMemory>> serializedEngine(
        builder->buildSerializedNetwork(*network, *builderConfig));
    requireCondition(serializedEngine != nullptr, "TensorRT engine build failed");
    const auto buildEnd = std::chrono::steady_clock::now();

    writeBinaryFile(enginePath, serializedEngine->data(), serializedEngine->size());
    std::cout << "TensorRT engine saved to: " << std::filesystem::absolute(enginePath) << '\n';
    std::cout << "Engine build time: " << elapsedMilliseconds(buildStart, buildEnd) / 1000.0 << " seconds\n";
}

TensorRtYoloV8::TensorRtYoloV8(const std::string& enginePath, int gpuDeviceIndex)
    : gpuDeviceIndex_(gpuDeviceIndex)
{
    checkCuda(cudaSetDevice(gpuDeviceIndex_), "cudaSetDevice");
    loadEngine(enginePath);
    initializeBindings();

    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    checkCuda(cudaEventCreate(&pipelineStartEvent_), "cudaEventCreate pipeline start");
    checkCuda(cudaEventCreate(&inferenceStartEvent_), "cudaEventCreate inference start");
    checkCuda(cudaEventCreate(&inferenceEndEvent_), "cudaEventCreate inference end");
    checkCuda(cudaEventCreate(&pipelineEndEvent_), "cudaEventCreate pipeline end");
}

TensorRtYoloV8::~TensorRtYoloV8()
{
    if (pipelineEndEvent_ != nullptr) {
        cudaEventDestroy(pipelineEndEvent_);
    }
    if (inferenceEndEvent_ != nullptr) {
        cudaEventDestroy(inferenceEndEvent_);
    }
    if (inferenceStartEvent_ != nullptr) {
        cudaEventDestroy(inferenceStartEvent_);
    }
    if (pipelineStartEvent_ != nullptr) {
        cudaEventDestroy(pipelineStartEvent_);
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
    if (deviceOutput_ != nullptr) {
        cudaFree(deviceOutput_);
    }
    if (deviceInput_ != nullptr) {
        cudaFree(deviceInput_);
    }
}

void TensorRtYoloV8::loadEngine(const std::string& enginePath)
{
    const std::vector<char> serializedEngine = readBinaryFile(enginePath);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    requireCondition(runtime_ != nullptr, "Unable to create TensorRT runtime");

    engine_.reset(runtime_->deserializeCudaEngine(serializedEngine.data(), serializedEngine.size()));
    requireCondition(
        engine_ != nullptr,
        "Unable to deserialize TensorRT engine. Rebuild it on this GPU with this TensorRT version.");
    context_.reset(engine_->createExecutionContext());
    requireCondition(context_ != nullptr, "Unable to create TensorRT execution context");
}

void TensorRtYoloV8::initializeBindings()
{
#if NV_TENSORRT_MAJOR >= 10
    const int tensorCount = engine_->getNbIOTensors();
    for (int tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex) {
        const char* tensorName = engine_->getIOTensorName(tensorIndex);
        requireCondition(tensorName != nullptr, "TensorRT engine contains an unnamed I/O tensor");
        requireCondition(
            engine_->getTensorDataType(tensorName) == nvinfer1::DataType::kFLOAT,
            "This example currently requires FP32 engine I/O tensors");

        if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            requireCondition(inputTensorName_.empty(), "Only one input tensor is supported");
            inputTensorName_ = tensorName;
            inputDimensions_ = engine_->getTensorShape(tensorName);
        } else {
            requireCondition(outputTensorName_.empty(), "Only one output tensor is supported");
            outputTensorName_ = tensorName;
            outputDimensions_ = engine_->getTensorShape(tensorName);
        }
    }
#else
    const int bindingCount = engine_->getNbBindings();
    deviceBindings_.resize(static_cast<std::size_t>(bindingCount), nullptr);
    for (int bindingIndex = 0; bindingIndex < bindingCount; ++bindingIndex) {
        requireCondition(
            engine_->getBindingDataType(bindingIndex) == nvinfer1::DataType::kFLOAT,
            "This example currently requires FP32 engine I/O tensors");
        if (engine_->bindingIsInput(bindingIndex)) {
            requireCondition(inputBindingIndex_ < 0, "Only one input tensor is supported");
            inputBindingIndex_ = bindingIndex;
            inputTensorName_ = engine_->getBindingName(bindingIndex);
            inputDimensions_ = engine_->getBindingDimensions(bindingIndex);
        } else {
            requireCondition(outputBindingIndex_ < 0, "Only one output tensor is supported");
            outputBindingIndex_ = bindingIndex;
            outputTensorName_ = engine_->getBindingName(bindingIndex);
            outputDimensions_ = engine_->getBindingDimensions(bindingIndex);
        }
    }
#endif

    requireCondition(!inputTensorName_.empty(), "TensorRT engine has no input tensor");
    requireCondition(!outputTensorName_.empty(), "TensorRT engine has no output tensor");
    requireCondition(dimensionsAreStatic(inputDimensions_), "Dynamic input shapes are not supported by this example");
    requireCondition(dimensionsAreStatic(outputDimensions_), "Dynamic output shapes are not supported by this example");
    requireCondition(
        inputDimensions_.nbDims == 4 && inputDimensions_.d[0] == 1 && inputDimensions_.d[1] == 3,
        "Expected a static YOLO input tensor with shape [1, 3, height, width]");
    requireCondition(outputDimensions_.nbDims == 3, "Expected a rank-3 YOLOv8 output tensor");

    hostInput_.resize(calculateVolume(inputDimensions_));
    hostOutput_.resize(calculateVolume(outputDimensions_));
    checkCuda(
        cudaMalloc(&deviceInput_, hostInput_.size() * sizeof(float)),
        "cudaMalloc input tensor");
    checkCuda(
        cudaMalloc(&deviceOutput_, hostOutput_.size() * sizeof(float)),
        "cudaMalloc output tensor");

#if NV_TENSORRT_MAJOR >= 10
    requireCondition(
        context_->setTensorAddress(inputTensorName_.c_str(), deviceInput_),
        "Unable to bind TensorRT input tensor");
    requireCondition(
        context_->setTensorAddress(outputTensorName_.c_str(), deviceOutput_),
        "Unable to bind TensorRT output tensor");
#else
    deviceBindings_[static_cast<std::size_t>(inputBindingIndex_)] = deviceInput_;
    deviceBindings_[static_cast<std::size_t>(outputBindingIndex_)] = deviceOutput_;
#endif

    std::cout << "TensorRT input: " << inputTensorName_ << " [1, 3, "
              << inputHeight() << ", " << inputWidth() << "]\n";
    std::cout << "TensorRT output: " << outputTensorName_ << " ["
              << outputDimensions_.d[0] << ", " << outputDimensions_.d[1] << ", "
              << outputDimensions_.d[2] << "]\n";
}

TensorRtYoloV8::LetterboxTransform TensorRtYoloV8::preprocess(const cv::Mat& image)
{
    requireCondition(!image.empty(), "Input image is empty");
    requireCondition(image.type() == CV_8UC3, "Input image must be an 8-bit three-channel BGR image");

    const float horizontalScale = static_cast<float>(inputWidth()) / static_cast<float>(image.cols);
    const float verticalScale = static_cast<float>(inputHeight()) / static_cast<float>(image.rows);
    const float resizeScale = std::min(horizontalScale, verticalScale);
    const int resizedWidth = std::min(inputWidth(), static_cast<int>(std::round(image.cols * resizeScale)));
    const int resizedHeight = std::min(inputHeight(), static_cast<int>(std::round(image.rows * resizeScale)));
    const int totalHorizontalPadding = inputWidth() - resizedWidth;
    const int totalVerticalPadding = inputHeight() - resizedHeight;
    const int leftPadding = totalHorizontalPadding / 2;
    const int rightPadding = totalHorizontalPadding - leftPadding;
    const int topPadding = totalVerticalPadding / 2;
    const int bottomPadding = totalVerticalPadding - topPadding;

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat letterboxedImage;
    cv::copyMakeBorder(
        resizedImage,
        letterboxedImage,
        topPadding,
        bottomPadding,
        leftPadding,
        rightPadding,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

    const int imagePlaneSize = inputWidth() * inputHeight();
    for (int rowIndex = 0; rowIndex < inputHeight(); ++rowIndex) {
        const cv::Vec3b* row = letterboxedImage.ptr<cv::Vec3b>(rowIndex);
        for (int columnIndex = 0; columnIndex < inputWidth(); ++columnIndex) {
            const cv::Vec3b& bgrPixel = row[columnIndex];
            const int pixelIndex = rowIndex * inputWidth() + columnIndex;
            hostInput_[static_cast<std::size_t>(pixelIndex)] = static_cast<float>(bgrPixel[2]) / 255.0F;
            hostInput_[static_cast<std::size_t>(imagePlaneSize + pixelIndex)] =
                static_cast<float>(bgrPixel[1]) / 255.0F;
            hostInput_[static_cast<std::size_t>(2 * imagePlaneSize + pixelIndex)] =
                static_cast<float>(bgrPixel[0]) / 255.0F;
        }
    }

    return {resizeScale, static_cast<float>(leftPadding), static_cast<float>(topPadding)};
}

void TensorRtYoloV8::enqueueInference()
{
#if NV_TENSORRT_MAJOR >= 10
    requireCondition(context_->enqueueV3(stream_), "TensorRT enqueueV3 failed");
#else
    requireCondition(
        context_->enqueueV2(deviceBindings_.data(), stream_, nullptr),
        "TensorRT enqueueV2 failed");
#endif
}

InferenceResult TensorRtYoloV8::infer(
    const cv::Mat& image,
    float confidenceThreshold,
    float iouThreshold)
{
    const auto endToEndStart = std::chrono::steady_clock::now();
    const auto preprocessStart = endToEndStart;
    const LetterboxTransform transform = preprocess(image);
    const auto preprocessEnd = std::chrono::steady_clock::now();

    checkCuda(cudaEventRecord(pipelineStartEvent_, stream_), "Record GPU pipeline start");
    checkCuda(
        cudaMemcpyAsync(
            deviceInput_,
            hostInput_.data(),
            hostInput_.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream_),
        "Copy input to GPU");
    checkCuda(cudaEventRecord(inferenceStartEvent_, stream_), "Record inference start");
    enqueueInference();
    checkCuda(cudaEventRecord(inferenceEndEvent_, stream_), "Record inference end");
    checkCuda(
        cudaMemcpyAsync(
            hostOutput_.data(),
            deviceOutput_,
            hostOutput_.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_),
        "Copy output to CPU");
    checkCuda(cudaEventRecord(pipelineEndEvent_, stream_), "Record GPU pipeline end");
    checkCuda(cudaEventSynchronize(pipelineEndEvent_), "Synchronize GPU pipeline");

    float inferenceMilliseconds = 0.0F;
    float pipelineMilliseconds = 0.0F;
    checkCuda(
        cudaEventElapsedTime(&inferenceMilliseconds, inferenceStartEvent_, inferenceEndEvent_),
        "Measure TensorRT inference time");
    checkCuda(
        cudaEventElapsedTime(&pipelineMilliseconds, pipelineStartEvent_, pipelineEndEvent_),
        "Measure GPU pipeline time");

    const auto postprocessStart = std::chrono::steady_clock::now();
    std::vector<Detection> detections = postprocess(
        image.size(), transform, confidenceThreshold, iouThreshold);
    const auto postprocessEnd = std::chrono::steady_clock::now();

    InferenceResult result;
    result.detections = std::move(detections);
    result.timing.preprocessMilliseconds = elapsedMilliseconds(preprocessStart, preprocessEnd);
    result.timing.inferenceMilliseconds = inferenceMilliseconds;
    result.timing.gpuPipelineMilliseconds = pipelineMilliseconds;
    result.timing.postprocessMilliseconds = elapsedMilliseconds(postprocessStart, postprocessEnd);
    result.timing.endToEndMilliseconds = elapsedMilliseconds(endToEndStart, postprocessEnd);
    return result;
}

std::vector<Detection> TensorRtYoloV8::postprocess(
    const cv::Size& originalImageSize,
    const LetterboxTransform& transform,
    float confidenceThreshold,
    float iouThreshold) const
{
    const int firstOutputAxis = outputDimensions_.d[1];
    const int secondOutputAxis = outputDimensions_.d[2];
    const bool attributesFirst = firstOutputAxis < secondOutputAxis;
    const int attributeCount = attributesFirst ? firstOutputAxis : secondOutputAxis;
    const int candidateCount = attributesFirst ? secondOutputAxis : firstOutputAxis;
    const int classCount = attributeCount - 4;
    requireCondition(classCount > 0, "YOLOv8 output must contain four box values and class scores");

    const auto outputValue = [this, attributesFirst, candidateCount, attributeCount](
                                 int candidateIndex,
                                 int attributeIndex) -> float {
        const std::size_t outputIndex = attributesFirst
            ? static_cast<std::size_t>(attributeIndex * candidateCount + candidateIndex)
            : static_cast<std::size_t>(candidateIndex * attributeCount + attributeIndex);
        return hostOutput_[outputIndex];
    };

    std::vector<DetectionCandidate> candidates;
    candidates.reserve(static_cast<std::size_t>(candidateCount));
    for (int candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex) {
        int bestClassId = 0;
        float bestClassScore = -std::numeric_limits<float>::infinity();
        for (int classIndex = 0; classIndex < classCount; ++classIndex) {
            const float classScore = outputValue(candidateIndex, classIndex + 4);
            if (classScore > bestClassScore) {
                bestClassScore = classScore;
                bestClassId = classIndex;
            }
        }
        if (bestClassScore < confidenceThreshold) {
            continue;
        }

        const float centerX = outputValue(candidateIndex, 0);
        const float centerY = outputValue(candidateIndex, 1);
        const float boxWidth = outputValue(candidateIndex, 2);
        const float boxHeight = outputValue(candidateIndex, 3);
        const float originalLeft = (centerX - boxWidth * 0.5F - transform.paddingX) / transform.scale;
        const float originalTop = (centerY - boxHeight * 0.5F - transform.paddingY) / transform.scale;
        const float originalRight = (centerX + boxWidth * 0.5F - transform.paddingX) / transform.scale;
        const float originalBottom = (centerY + boxHeight * 0.5F - transform.paddingY) / transform.scale;

        const int left = std::clamp(static_cast<int>(std::floor(originalLeft)), 0, originalImageSize.width - 1);
        const int top = std::clamp(static_cast<int>(std::floor(originalTop)), 0, originalImageSize.height - 1);
        const int right = std::clamp(static_cast<int>(std::ceil(originalRight)), 0, originalImageSize.width);
        const int bottom = std::clamp(static_cast<int>(std::ceil(originalBottom)), 0, originalImageSize.height);
        if (right <= left || bottom <= top) {
            continue;
        }
        candidates.push_back({bestClassId, bestClassScore, cv::Rect(left, top, right - left, bottom - top)});
    }

    std::vector<Detection> detections;
    for (int classIndex = 0; classIndex < classCount; ++classIndex) {
        std::vector<cv::Rect> classBoxes;
        std::vector<float> classScores;
        std::vector<std::size_t> candidateIndices;
        for (std::size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
            if (candidates[candidateIndex].classId == classIndex) {
                classBoxes.push_back(candidates[candidateIndex].boundingBox);
                classScores.push_back(candidates[candidateIndex].confidence);
                candidateIndices.push_back(candidateIndex);
            }
        }

        std::vector<int> retainedIndices;
        cv::dnn::NMSBoxes(classBoxes, classScores, confidenceThreshold, iouThreshold, retainedIndices);
        for (int retainedIndex : retainedIndices) {
            const DetectionCandidate& candidate = candidates[candidateIndices[static_cast<std::size_t>(retainedIndex)]];
            detections.push_back({candidate.classId, candidate.confidence, candidate.boundingBox});
        }
    }

    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& first, const Detection& second) {
            return first.confidence > second.confidence;
        });
    return detections;
}

int TensorRtYoloV8::inputWidth() const noexcept
{
    return inputDimensions_.d[3];
}

int TensorRtYoloV8::inputHeight() const noexcept
{
    return inputDimensions_.d[2];
}

std::size_t TensorRtYoloV8::calculateVolume(const nvinfer1::Dims& dimensions)
{
    std::size_t volume = 1;
    for (int dimensionIndex = 0; dimensionIndex < dimensions.nbDims; ++dimensionIndex) {
        requireCondition(dimensions.d[dimensionIndex] > 0, "Cannot calculate the volume of a dynamic tensor");
        volume *= static_cast<std::size_t>(dimensions.d[dimensionIndex]);
    }
    return volume;
}
