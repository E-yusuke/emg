import numpy as np
import pandas as pd
import os
import re


class preprocessing(object):
    def __init__(self, path_in, window_size=250):
        self.path_in = path_in
        self.emg = np.zeros((300, 12000))
        self.emg_rms = np.zeros((300, 12000))
        self.emg_iemg = np.zeros((300, 4))

    def emgdata(self):
        emg = self.data()
        # 最大値を抽出＝＞最大値正規化＆整流
        emg_max = np.max(np.abs(emg))
        emg_std = np.abs(emg)/emg_max
        # RMS
        emg_rms = self.emg_rms
        for i in range(300):
            for j in range(0, 12000, 3000):
                emg_rms[i, (0 + j):(3000 + j)] = self.window_rms(emg_std[i, (0 + j):(3000 + j)])
        return emg_rms

    def iemgdata(self):
        emg_rms = self.emgdata()
        # iemg+RMS
        emg_iemg = self.emg_iemg
        for i in range(300):
            for index, j in enumerate([0, 2999, 5999, 8999]):
                emg_iemg[i, index] = np.sum(emg_rms[i, (0 + j):(3000 + j)])/3000
        iemg_max = np.sum(emg_iemg, axis=1)
        emg_iemg = emg_iemg/np.tile(iemg_max, (4, 1)).T
        return emg_iemg

    def emg_3ch_rms(self):
        emg_rms = self.emgdata()
        return self.convert_data(emg_rms)

    def emg_3ch_iemg(self):
        emg_iemg = self.iemgdata()
        return self.convert_data(emg_iemg)

    def data(self):
        path_in = self.path_in
        files = os.listdir(path_in)  # ファイル名取得
        files = sorted(files, key=lambda x: int((re.search(r"[0-9]+", x)).group(0)))  # 測定順に並べ替え
        connected_horizontally = pd.DataFrame()
        connected_vertical = pd.DataFrame()
        for filename in files:
            df = pd.read_csv(os.path.join(path_in, filename), header=None, nrows=3000).T
            for channel in range(24):
                if 'df' in locals():
                    connected_horizontally = pd.concat([connected_horizontally, df.iloc[channel, :]], axis=0)
                else:
                    connected_horizontally = df.iloc[channel, :]
                if connected_horizontally.size == 12000:
                    if 'df' in locals():
                        connected_vertical = pd.concat([connected_vertical, connected_horizontally], axis=1)
                    else:
                        connected_vertical = connected_horizontally
                    connected_horizontally = pd.DataFrame()
        emg_ = connected_vertical.T

        # 動作ごとにまとめる
        emg = self.emg
        count = 0
        for j in range(6):
            for i in range(50):
                emg[count, :] = emg_.iloc[j+i*6, :].values
                count += 1
        return emg

    def convert_data(self, emg):
        length = emg.shape[1]
        print('data length:', emg.shape)
        print('data per 1ch:', int(length/4))
        emg_3ch = []
        emg_index = [list(range(0, int(length/4))),
                     list(range(int(length/4), int(length/4)*2)),
                     list(range(int(length/4)*2, int(length/4)*3)),
                     list(range(int(length/4)*3, int(length/4)*4))]
        print([len(v) for v in emg_index])
        for index in range(4):
            print('index:', index % 4, (index+1) % 4, (index+2) % 4)
            emg3 = np.hstack((emg[:, emg_index[index % 4][:]], emg[:, emg_index[(index+1) % 4][:]],
                             emg[:, emg_index[(index+2) % 4][:]]))
            emg_3ch.append(emg3)

        return np.array(emg_3ch)

    def window_rms(self, a, window_size=250):
        a2 = np.power(a, 2)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(a2, window, "same"))
