from utils import nms, letterbox
import onnxruntime
import numpy as np
import time
from camera import get_img, connect_camera



IMAGE_SIZE = 416
anchor_list = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
stride = [8, 16, 32]
CLASSES = 80
CONF_TH = 0.6
NMS_TH = 0.6
area = IMAGE_SIZE * IMAGE_SIZE


# model = onnxruntime.InferenceSession("helmet.onnx",providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
model = onnxruntime.InferenceSession("yolov5s.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
camera_list = connect_camera()
camera_num = len(camera_list)
obj_name = ['瓶子']*8
index_dection = [39]*8


while True:
    t = time.time()
    check_num = 0
    for idx, s in enumerate(camera_list):
        frame = get_img(s, 1152, 648)          # 减小送入网络图片的分辨率，原尺寸2304*1296
        img, ratio, (dw, dh) = letterbox(frame, new_shape=IMAGE_SIZE, auto=False)
        img = img.transpose(2, 0, 1).astype(np.float32)
        img = img / 255.0
        img = img.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        inputs = {model.get_inputs()[0].name: img}
        pred = model.run(None, inputs)[0]
        anchor = np.array(anchor_list).astype(np.float16).reshape(3, -1, 2)
        size = [int(area / stride[0] ** 2), int(area / stride[1] ** 2), int(area / stride[2] ** 2)]
        feature = [[int(j / stride[i]) for j in (IMAGE_SIZE, IMAGE_SIZE)] for i in range(3)]
        y = []
        y.append(pred[:, :size[0] * 3, :])
        y.append(pred[:, size[0] * 3:size[0] * 3 + size[1] * 3, :])
        y.append(pred[:, size[0] * 3 + size[1] * 3:, :])
        grid = []
        for k, f in enumerate(feature):
            grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])
        z = []
        for i in range(3):
            src = y[i]
            xy = src[..., 0:2] * 2. - 0.5
            wh = (src[..., 2:4] * 2) ** 2
            dst_xy = []
            dst_wh = []
            for j in range(3):
                dst_xy.append((xy[:, j * size[i]:(j + 1) * size[i], :] + grid[i]) * stride[i])
                dst_wh.append(wh[:, j * size[i]:(j + 1) * size[i], :] * anchor[i][j])
            src[..., 0:2] = np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1)
            src[..., 2:4] = np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1)
            z.append(src.reshape(1, -1, CLASSES + 5))  # 85

        pred = np.concatenate(z, 1)
        pred = nms(pred, CONF_TH, NMS_TH)

        boxes = pred[0]
        if boxes is not None:
            for box in boxes:
                if dw == 0:
                    box[1] = (box[1] - dh)
                    box[3] = (box[3] - dh)
                elif dh == 0:
                    box[0] = (box[0] - dw)
                    box[2] = (box[2] - dw)
                for i in range(4):
                    box[i] = box[i] / ratio

            color = (255, 0, 0)
            c = 0
            for box in boxes:
                if box[-1] == index_dection[idx]:
                    c += 1
                    # frame= frame[int(box[1])+1:int(box[3]),int(box[0])+1:int(box[2]),:]
                    # cv2.rectangle(frame, (int(box[0]), int(box[1])),  (int(box[2]), int(box[3])), color)
            print('camera-{} find : {} {}'.format(idx + 1, c, obj_name[idx]))
            check_num += 1
        else:
            print('camera-{} find nothing'.format(idx + 1))
            check_num += 1
    if check_num == camera_num:
        print('---------{} cameras one round use time: {} s'.format(camera_num, time.time() - t))
        # frame=cv2.resize(frame, (1152, 648))
        # cv2.imshow('test', frame)












