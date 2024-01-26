from flask import Flask, request, jsonify
import requests
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

app = Flask(__name__)

model = YOLO('best (3).pt')

@app.route('/', methods=['GET'])
def index():
    img_url = request.args.get('img')

    response = requests.get(img_url, stream=True)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print(img_url)

    classes_ = {0: 'noti', 1: 'pop'}

    results = model.predict(source=img, conf = 0.7)

    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    print(boxes)
    print(classes)
    print(names)
    print(confidences)

    result_dict = {"boxes": boxes, "classes": classes, "names": names, "confidence": confidences}

    return jsonify(result_dict)


# for box, cls, conf in zip(boxes, classes, confidences):
#     x1, y1, x2, y2 = box
#     confidence = conf
#     detected_class = cls
#     name = names[int(cls)]



# def plot_img_bbox(img, target):
#     fig, a = plt.subplots(1,1)
#     fig.set_size_inches(10, 10)
#     a.imshow(img)
#     for i, box in enumerate(target):
#         #print(target['boxes'])
#         x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
# #         if arr[target['labels'][i]] == 'ad':
#         rect = patches.Rectangle((x, y),
#                                      width, height,
#                                      linewidth = 2,
#                                      edgecolor = 'r',
#                                      facecolor = 'none')
#         a.text(x, y-20, classes_[classes[i]], color='b', verticalalignment='top')

#         a.add_patch(rect)
#     plt.show()

# plot_img_bbox(img, boxes)


if __name__ == "__main__":
    app.run(debug=True)
