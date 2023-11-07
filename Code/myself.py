import cv2 as cv
import numpy as np


class KCFTracker:
    def __init__(self, win):
        self.w = None
        self.h = None
        self.scale_w = None
        self.scale_h = None
        self.alpha = None
        self.x = None
        self.block_size = np.array((8, 8))
        self.block_stride = np.array((4, 4))
        self.cell_size = np.array((4, 4))
        self.n_bins = 9
        self.padding = 2.5
        self.Sigma = 0.125
        self.sigma = 0.6
        self.lambdaf = 1e-9
        self.lr = 0.02
        self.win = win
        win_size = np.array((win[2], win[3])) * 256 / float(max(win[2], win[3])) // 4 * 4 + 4
        win_size = (int(win_size[0]), int(win_size[1]))
        self.win_size = win_size
        self.win_stride = win_size
        self.hog = cv.HOGDescriptor(win_size, self.block_size, self.block_stride, self.cell_size, self.n_bins)

    def Get_hog_feature(self, img):
        hist = np.array(self.hog.compute(img, self.win_stride))
        size = self.win_size // self.block_stride - 1
        return hist.reshape(size[0], size[1], 36).transpose(2, 1, 0)

    def Gaussian_peak(self, size):
        w, h = size
        Sigma = np.sqrt(w * h) / self.padding * self.Sigma
        mat = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                mat[i, j] = np.exp(-0.5 * ((i - h // 2) ** 2 + (j - w // 2) ** 2) / (Sigma ** 2))
        mat /= 2 * np.pi * Sigma ** 2
        return mat

    def Kernel_correlation(self, x1, x2):
        f = np.conj(np.fft.fft2(x1)) * np.fft.fft2(x2)
        f = np.sum(f, axis=0)
        f = np.fft.fftshift(np.fft.ifft2(f)) * 2
        f -= np.sum(x1 ** 2) + np.sum(x2 ** 2)
        f = np.abs(f)
        f /= self.sigma ** 2 * f.size
        f = np.exp(-f)
        return f

    def Hanning_window(self, size):
        w, h = size
        windows = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                windows[i, j] = 0.5 * (1 - np.cos(2 * np.pi * i / h)) * 0.5 * (1 - np.cos(2 * np.pi * j / w))

        return windows

    def Sub_windows(self, img):
        px, py, w, h = self.win
        mx = px + w // 2
        my = py + h // 2
        self.w = int(w * self.padding) // 2 * 2
        self.h = int(h * self.padding) // 2 * 2
        x = int(mx - w // 2)
        y = int(my - h // 2)

        if y < 0 or x < 0 or y + self.h > img.shape[0] or x + self.w > img.shape[1]:
            return cv.resize(img, self.win_size)

        return cv.resize(img[y:y + self.h, x:x + self.w, :], self.win_size)

    def Get_feature(self, img):
        sub_img = self.Sub_windows(img)
        feature = self.Get_hog_feature(sub_img)
        c, h, w = feature.shape
        self.scale_w = w / self.w
        self.scale_h = h / self.h
        return feature * self.Hanning_window((w, h))

    def Train(self):
        y = self.Gaussian_peak((self.x.shape[2], self.x.shape[1]))
        k = self.Kernel_correlation(self.x, self.x)
        return np.fft.fft2(y) / (np.fft.fft2(k) + self.lambdaf)

    def Detect(self, z):
        k = self.Kernel_correlation(self.x, z)
        return np.fft.ifft2(self.alpha * np.fft.fft2(k))

    def Init(self, img):
        self.x = self.Get_feature(img)
        self.alpha = self.Train()

    def Update(self, img):
        z = self.Get_feature(img)
        y = self.Detect(z)
        mx_id = (0, 0)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i, j] > y[mx_id]:
                    mx_id = (i, j)
        d = np.array(((mx_id[1] - y.shape[1] // 2) / self.scale_w, (mx_id[0] - y.shape[0] // 2) / self.scale_h, 0, 0))
        self.win += d
        self.x = (1 - self.lr) * self.x + self.lr * z
        self.alpha = (1 - self.lr) * self.alpha + self.lr * self.Train()


if __name__ == "__main__":
    video = cv.VideoCapture("../Dataset/1.mp4")
    con, img = video.read()
    output = cv.VideoWriter('../Result/myself.mp4', cv.VideoWriter.fourcc('m', 'p', '4', 'v'), 30,
                            (img.shape[1], img.shape[0]))
    win = cv.selectROI(img, False)
    tracker = KCFTracker(win)
    tracker.Init(img)
    while video.isOpened():
        con, img = video.read()
        if not con:
            break
        tracker.Update(img)
        win = tracker.win
        cv.rectangle(img,
                     (int(win[0]), int(win[1])),
                     (int(win[0] + win[2]),
                      int(win[1] + win[3])),
                     (0, 255, 0), 2)
        cv.imshow("img", img)
        output.write(img)
        cv.waitKey(1)
    video.release()
    cv.destroyAllWindows()
