#include "tensorrt_yolov8.h"

#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CommandLineOptions {
    std::string onnxPath{"best.onnx"};
    std::string enginePath{"best.engine"};
    std::string imagePath;
    std::string classesPath{"classes.txt"};
    std::string outputPath{"tensorrt_result.jpg"};
    int gpuDeviceIndex{0};
    int warmupIterations{10};
    int measuredIterations{100};
    int workspaceMegabytes{1024};
    float confidenceThreshold{0.25F};
    float iouThreshold{0.7F};
    bool enableFp16{false};
    bool rebuildEngine{false};
    bool buildOnly{false};
};

struct TimingStatistics {
    double mean{0.0};
    double minimum{0.0};
    double maximum{0.0};
    double median{0.0};
    double percentile95{0.0};
};

void printUsage(const char* executableName)
{
    std::cout
        << "Usage: " << executableName << " --image IMAGE [options]\n"
        << "\nRequired when building an engine:\n"
        << "  --onnx PATH             Source YOLOv8 ONNX model (default: best.onnx)\n"
        << "\nInference options:\n"
        << "  --engine PATH           TensorRT engine path (default: best.engine)\n"
        << "  --image PATH            Input image path\n"
        << "  --classes PATH          Class names file, one name per line (default: classes.txt)\n"
        << "  --output PATH           Annotated image path\n"
        << "  --gpu INDEX             CUDA device index (default: 0)\n"
        << "  --conf VALUE            Confidence threshold (default: 0.25)\n"
        << "  --iou VALUE             NMS IoU threshold (default: 0.7)\n"
        << "  --warmup COUNT          Warmup frame count (default: 10)\n"
        << "  --iterations COUNT      Measured frame count (default: 100)\n"
        << "\nEngine build options:\n"
        << "  --fp16                  Build an FP16 engine\n"
        << "  --workspace MB          TensorRT workspace size (default: 1024)\n"
        << "  --rebuild               Rebuild the engine even when it exists\n"
        << "  --build-only            Build the engine without running inference\n"
        << "  --help                  Show this message\n";
}

std::string requireArgumentValue(int argc, char** argv, int& argumentIndex)
{
    if (argumentIndex + 1 >= argc) {
        throw std::invalid_argument(std::string("Missing value for ") + argv[argumentIndex]);
    }
    ++argumentIndex;
    return argv[argumentIndex];
}

CommandLineOptions parseCommandLine(int argc, char** argv)
{
    CommandLineOptions options;
    for (int argumentIndex = 1; argumentIndex < argc; ++argumentIndex) {
        const std::string argument = argv[argumentIndex];
        if (argument == "--onnx") {
            options.onnxPath = requireArgumentValue(argc, argv, argumentIndex);
        } else if (argument == "--engine") {
            options.enginePath = requireArgumentValue(argc, argv, argumentIndex);
        } else if (argument == "--image") {
            options.imagePath = requireArgumentValue(argc, argv, argumentIndex);
        } else if (argument == "--classes") {
            options.classesPath = requireArgumentValue(argc, argv, argumentIndex);
        } else if (argument == "--output") {
            options.outputPath = requireArgumentValue(argc, argv, argumentIndex);
        } else if (argument == "--gpu") {
            options.gpuDeviceIndex = std::stoi(requireArgumentValue(argc, argv, argumentIndex));
        } else if (argument == "--warmup") {
            options.warmupIterations = std::stoi(requireArgumentValue(argc, argv, argumentIndex));
        } else if (argument == "--iterations") {
            options.measuredIterations = std::stoi(requireArgumentValue(argc, argv, argumentIndex));
        } else if (argument == "--workspace") {
            options.workspaceMegabytes = std::stoi(requireArgumentValue(argc, argv, argumentIndex));
        } else if (argument == "--conf") {
            options.confidenceThreshold = std::stof(requireArgumentValue(argc, argv, argumentIndex));
        } else if (argument == "--iou") {
            options.iouThreshold = std::stof(requireArgumentValue(argc, argv, argumentIndex));
        } else if (argument == "--fp16") {
            options.enableFp16 = true;
        } else if (argument == "--rebuild") {
            options.rebuildEngine = true;
        } else if (argument == "--build-only") {
            options.buildOnly = true;
        } else if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }

    if (!options.buildOnly && options.imagePath.empty()) {
        throw std::invalid_argument("--image is required unless --build-only is used");
    }
    if (options.warmupIterations < 0 || options.measuredIterations <= 0) {
        throw std::invalid_argument("--warmup must be non-negative and --iterations must be positive");
    }
    if (options.workspaceMegabytes <= 0) {
        throw std::invalid_argument("--workspace must be positive");
    }
    if (options.confidenceThreshold < 0.0F || options.confidenceThreshold > 1.0F ||
        options.iouThreshold < 0.0F || options.iouThreshold > 1.0F) {
        throw std::invalid_argument("--conf and --iou must be within [0, 1]");
    }
    return options;
}

std::vector<std::string> loadClassNames(const std::string& classesPath)
{
    std::vector<std::string> classNames;
    if (classesPath.empty()) {
        return classNames;
    }

    std::ifstream classesFile(classesPath);
    if (!classesFile) {
        throw std::runtime_error("Unable to open classes file: " + classesPath);
    }

    std::string className;
    while (std::getline(classesFile, className)) {
        if (!className.empty()) {
            classNames.push_back(className);
        }
    }
    return classNames;
}

double calculatePercentile(const std::vector<double>& sortedValues, double percentile)
{
    if (sortedValues.empty()) {
        return 0.0;
    }
    const double position = percentile * static_cast<double>(sortedValues.size() - 1);
    const std::size_t lowerIndex = static_cast<std::size_t>(std::floor(position));
    const std::size_t upperIndex = static_cast<std::size_t>(std::ceil(position));
    const double interpolation = position - static_cast<double>(lowerIndex);
    return sortedValues[lowerIndex] * (1.0 - interpolation) + sortedValues[upperIndex] * interpolation;
}

TimingStatistics calculateStatistics(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    TimingStatistics statistics;
    statistics.minimum = values.front();
    statistics.maximum = values.back();
    statistics.mean = std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    statistics.median = calculatePercentile(values, 0.5);
    statistics.percentile95 = calculatePercentile(values, 0.95);
    return statistics;
}

void printStatistics(const std::string& label, const std::vector<double>& values)
{
    const TimingStatistics statistics = calculateStatistics(values);
    std::cout << std::left << std::setw(16) << label
              << " mean=" << std::right << std::setw(8) << statistics.mean
              << " ms  P50=" << std::setw(8) << statistics.median
              << " ms  P95=" << std::setw(8) << statistics.percentile95
              << " ms  min=" << std::setw(8) << statistics.minimum
              << " ms  max=" << std::setw(8) << statistics.maximum << " ms\n";
}

cv::Scalar colorForClass(int classId)
{
    const int blue = (37 * classId + 80) % 255;
    const int green = (17 * classId + 160) % 255;
    const int red = (29 * classId + 220) % 255;
    return {blue, green, red};
}

void drawDetections(
    cv::Mat& image,
    const std::vector<Detection>& detections,
    const std::vector<std::string>& classNames)
{
    for (const Detection& detection : detections) {
        const cv::Scalar color = colorForClass(detection.classId);
        cv::rectangle(image, detection.boundingBox, color, 2);

        const std::string className = detection.classId >= 0 &&
                detection.classId < static_cast<int>(classNames.size())
            ? classNames[static_cast<std::size_t>(detection.classId)]
            : "class_" + std::to_string(detection.classId);
        const std::string label = className + " " + cv::format("%.2f", detection.confidence);
        int baseline = 0;
        const cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        const int labelTop = std::max(detection.boundingBox.y, labelSize.height + 6);
        cv::rectangle(
            image,
            cv::Point(detection.boundingBox.x, labelTop - labelSize.height - 6),
            cv::Point(detection.boundingBox.x + labelSize.width + 6, labelTop),
            color,
            cv::FILLED);
        cv::putText(
            image,
            label,
            cv::Point(detection.boundingBox.x + 3, labelTop - 4),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 0, 0),
            1,
            cv::LINE_AA);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CommandLineOptions options = parseCommandLine(argc, argv);
        const cudaError_t deviceSelectionStatus = cudaSetDevice(options.gpuDeviceIndex);
        if (deviceSelectionStatus != cudaSuccess) {
            throw std::runtime_error(
                "Unable to select CUDA device " + std::to_string(options.gpuDeviceIndex) +
                ": " + cudaGetErrorString(deviceSelectionStatus));
        }
        const std::filesystem::path enginePath(options.enginePath);
        const bool shouldBuildEngine = options.rebuildEngine || !std::filesystem::is_regular_file(enginePath);

        if (shouldBuildEngine) {
            if (options.onnxPath.empty()) {
                throw std::invalid_argument("--onnx is required when the TensorRT engine must be built");
            }
            EngineBuildOptions buildOptions;
            buildOptions.enableFp16 = options.enableFp16;
            buildOptions.workspaceBytes = static_cast<std::size_t>(options.workspaceMegabytes) * 1024ULL * 1024ULL;
            TensorRtYoloV8::buildEngine(options.onnxPath, options.enginePath, buildOptions);
        }

        if (options.buildOnly) {
            if (!shouldBuildEngine) {
                std::cout << "TensorRT engine already exists: "
                          << std::filesystem::absolute(enginePath) << '\n';
                std::cout << "Use --rebuild to regenerate it.\n";
            }
            return 0;
        }

        const cv::Mat sourceImage = cv::imread(options.imagePath, cv::IMREAD_COLOR);
        if (sourceImage.empty()) {
            throw std::runtime_error("Unable to read input image: " + options.imagePath);
        }

        const std::vector<std::string> classNames = loadClassNames(options.classesPath);
        TensorRtYoloV8 detector(options.enginePath, options.gpuDeviceIndex);
        std::cout << "Input tensor: 1x3x" << detector.inputHeight() << 'x' << detector.inputWidth() << '\n';
        std::cout << "Warmup iterations: " << options.warmupIterations << '\n';
        for (int iteration = 0; iteration < options.warmupIterations; ++iteration) {
            detector.infer(sourceImage, options.confidenceThreshold, options.iouThreshold);
        }

        std::vector<double> preprocessTimes;
        std::vector<double> inferenceTimes;
        std::vector<double> gpuPipelineTimes;
        std::vector<double> postprocessTimes;
        std::vector<double> endToEndTimes;
        preprocessTimes.reserve(options.measuredIterations);
        inferenceTimes.reserve(options.measuredIterations);
        gpuPipelineTimes.reserve(options.measuredIterations);
        postprocessTimes.reserve(options.measuredIterations);
        endToEndTimes.reserve(options.measuredIterations);

        InferenceResult latestResult;
        for (int iteration = 0; iteration < options.measuredIterations; ++iteration) {
            latestResult = detector.infer(sourceImage, options.confidenceThreshold, options.iouThreshold);
            preprocessTimes.push_back(latestResult.timing.preprocessMilliseconds);
            inferenceTimes.push_back(latestResult.timing.inferenceMilliseconds);
            gpuPipelineTimes.push_back(latestResult.timing.gpuPipelineMilliseconds);
            postprocessTimes.push_back(latestResult.timing.postprocessMilliseconds);
            endToEndTimes.push_back(latestResult.timing.endToEndMilliseconds);
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\nSingle-frame latency over " << options.measuredIterations << " iterations:\n";
        printStatistics("Preprocess", preprocessTimes);
        printStatistics("Inference", inferenceTimes);
        printStatistics("GPU pipeline", gpuPipelineTimes);
        printStatistics("Postprocess", postprocessTimes);
        printStatistics("End-to-end", endToEndTimes);
        const double meanEndToEndMilliseconds = calculateStatistics(endToEndTimes).mean;
        std::cout << "Throughput: " << (1000.0 / meanEndToEndMilliseconds) << " FPS\n";
        std::cout << "Detections: " << latestResult.detections.size() << '\n';

        cv::Mat annotatedImage = sourceImage.clone();
        drawDetections(annotatedImage, latestResult.detections, classNames);
        const std::filesystem::path outputPath(options.outputPath);
        if (!outputPath.parent_path().empty()) {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        if (!cv::imwrite(options.outputPath, annotatedImage)) {
            throw std::runtime_error("Unable to write output image: " + options.outputPath);
        }
        std::cout << "Annotated result: " << std::filesystem::absolute(outputPath) << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }
}
