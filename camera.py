import os
import threading
import cv2
import time


class ipcamCapture:
    def __init__(self, URL, allow=False):
        self.Frame = []
        self.status = False
        self.isstop = False
        self.allow = allow
        self.url = URL
        self.capture = cv2.VideoCapture(URL)

    def start(self):
        # 把程序放进子线程，daemon=True 表示该线程会随着主线程关闭而关闭。
        print('{} ipcam started!'.format(self.url))
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        # 记得要设计停止无限循环的开关。
        self.isstop = True
        print('ipcam stopped!')

    def getframe(self):
        # 当有需要影像时，再回传最新的影像。
        return self.Frame

    def queryframe(self):
        while (not self.isstop):
            if self.allow:
                self.status, self.Frame = self.capture.read()
                if self.Frame is None:
                    print('camera {} loss frame'.format(self.url))
                    break
            else:
                time.sleep(0.1)
        self.capture.release()

def connect_camera():
    sources = 'streams.txt'
    if os.path.isfile(sources):
        with open(sources, 'r') as f:
            sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
    else:
        print('[ERROR]: 缺少摄像机流媒体资源！')

    ipcams = [ipcamCapture(sources[i], allow=True) for i in range(len(sources))]
    for i in range(len(ipcams)):
        ipcams[i].start()
    time.sleep(1)
    return ipcams


def get_img(ipcam, w, h):
    frame = ipcam.getframe()
    frame = cv2.resize(frame, (w, h))
    return frame