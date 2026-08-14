./build/yolov8_tensorrt \
  --engine best.engine \
  --image /home/yufeng/project/study/examples/YOLOv8-TensorRT-CPP/20260325_002018.jpg \
  --classes classes.txt \
  --output tensorrt_result.jpg \
  --warmup 10 \
  --iterations 100


