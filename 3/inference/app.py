from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor, Model
from mindvision.classification.models import mobilenet_v2
import numpy as np
import gradio as gr
from mindspore.nn import Softmax
import mindspore.dataset.vision.c_transforms as C


NUM_CLASS = 2
class_names = ['Croissants', 'Dog']
img_size = 224

def image_preprocessing(img):
    opr = C.Resize((img_size, img_size))
    img = opr(img)
    mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
    std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
    image = np.array(img)
    image = (image - mean) / std
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)
    image = Tensor(image)
    return image


def transform(img):
    img = image_preprocessing(img)

    net = mobilenet_v2(num_classes=2, resize=img_size)

    param_dict = load_checkpoint("./models/mobilenet_v2_0804.ckpt")
    load_param_into_net(net, param_dict)
    model = Model(net)

    predict_score = model.predict(img)
    predict_probability = Softmax()(predict_score)[0]
    predict_probability = predict_probability.asnumpy()

    return {class_names[i]: float(predict_probability[i]) for i in range(NUM_CLASS)}


examples = ['./examples/dog1.jpg', './examples/dog2.jpg', './examples/croissants1.jpg', './examples/croissants2.jpg']


gr.Interface(fn=transform,
            title='基于MobileNetV2的狗和牛角包二分类',
             inputs=[gr.inputs.Image()],
             outputs=gr.outputs.Label(num_top_classes=NUM_CLASS, label="预测类别"),
             examples=examples
             ).launch(share=True)