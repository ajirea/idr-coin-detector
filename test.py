import io
import cv2
import os
import requests
import numpy as np
from PIL import Image
import base64
from requests_toolbelt.multipart.encoder import MultipartEncoder


def prediction_image(m):
    resp = requests.post("https://detect.roboflow.com/coin-yxjpa/3?api_key=ef15bcCXZl0iHL09PWOf&format=image", data=m, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    imgRes = np.asarray(bytearray(resp.read()), dtype="uint8")
    imgRes = cv2.imdecode(imgRes, cv2.IMREAD_COLOR)
    cv2.imshow("image", imgRes)
    # imgFinal = cv2.imread()

    # cv2.imshow('image', )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def prediction_draw_rect_manual(m):
    response = requests.post("https://detect.roboflow.com/coin-yxjpa/3?api_key=ef15bcCXZl0iHL09PWOf",
                             data=m, headers={'Content-Type': m.content_type})
    json_res = response.json()

    for prediction in json_res.get('predictions'):
        x = prediction.get('x')
        y = prediction.get('y')
        w = int(prediction.get('width'))
        h = int(prediction.get('height'))
        cls = prediction.get('class')
        x1 = int(x - w / 2)
        x2 = int(x + w / 2)
        y1 = int(y - h / 2)
        y2 = int(y + h / 2)

        label = cls + ": " + str(int(prediction.get('confidence') * 100)) + "%"

        cv2.rectangle(img, (x1, y1),
                      (x2, y2), (0, 255, 0), 2)

        cv2.putText(img, label, (x1+10, y1+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for imgPath in os.listdir('test'):
    # Load Image with PIL
    img = cv2.imread("test/" + imgPath)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(image)

    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    pilImage.save(buffered, quality=100, format="JPEG")

    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode("ascii")

    # Build multipart form and post request
    m = MultipartEncoder(
        fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

    try:
        prediction_draw_rect_manual(m)
    except Exception as e:
        continue
