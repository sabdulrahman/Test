//install required libraries
//pip3 install tensorflow
//pip3 install opencv-python
//pip3 install keras
//pip3 install imageai --upgrade
//download coco weights
//https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5

from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "./models/resnet50_coco.h5"
input_path = "./images/444.jpg"
output_path = "./output/newimage.jpg"

detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])
