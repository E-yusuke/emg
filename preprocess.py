import pandas as pd
import numpy as np
import time
import os
import re


def convert_data(emg):
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


class preprocess(object):
    def __init__(self, path_in, window_size=250):
        self.path_in = path_in
        self.window_size = window_size
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.Powerless = None
        self.Hold = None
        self.Open = None
        self.Palm = None
        self.Dorsi = None
        self.Chop = None

    def readcsv(self, files, path_in):  # データを取得
        df = pd.DataFrame()
        for filename in files:
            csv_df = pd.read_csv(os.path.join(path_in, filename), header=None, nrows=3000)

            if 'df' in locals():
                df = pd.concat([df, csv_df])  # 列方向に結合

            else:
                df = csv_df

        return df

    def window_rms(self, a, window_size):
        a2 = np.power(a, 2)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(a2, window, "same"))

    def rms_editing(self, length, data, WINDOW_SIZE):
        for total in range(length):
            for ch in range(24):
                RMS = self.window_rms(data.iloc[0+3000*total:3000+3000*total, ch], WINDOW_SIZE)
                data.iloc[0+3000*total:3000+3000*total, ch] = RMS
        return data

    def emgdata(self):  # EMGの導出
        X_train, Y_train = self.X_train, self.Y_train
        Powerless, Hold, Open, Palm, Dorsi, Chop = self.Powerless, self.Hold, self.Open, self.Palm, self.Dorsi, self.Chop

        df = self.join()
        length = int(df.shape[0]/3000)  # 総データ数
        EMG = np.zeros((length, 72000))  # emptyのほうが早い3，4倍

        for index in range(length):
            data = np.reshape(np.array(df.iloc[0+3000*index:3000+3000*index, :].T), (1, -1))
            EMG[index, :] = data/np.max(data)
        Powerless = EMG[:, 0:12000]
        Hold = EMG[:, 12000:24000]
        Open = EMG[:, 24000:36000]
        Palm = EMG[:, 36000:48000]
        Dorsi = EMG[:, 48000:60000]
        Chop = EMG[:, 60000:72000]
        '''学習用データ、検証用データ'''
        # train data
        X_train = np.block([[Powerless], [Hold], [Open], [Palm], [Dorsi], [Chop]])
        Y_ = np.ones(50, dtype="int64")
        Y_train = np.block([0*Y_, Y_, 2*Y_, 3*Y_, 4*Y_, 5*Y_])

        return X_train, Y_train

    def iemgdata(self):
        df = self.join()
        length = int(df.shape[0]/3000)  # 総データ数
        ch = int(df.shape[1]/6)
        IEMG = np.zeros((length, df.shape[1]))  # emptyのほうが早い3，4倍
        Powerless = np.zeros((length, ch))
        Hold = np.zeros((length, ch))
        Open = np.zeros((length, ch))
        Palm = np.zeros((length, ch))
        Dorsi = np.zeros((length, ch))
        Chop = np.zeros((length, ch))

        for index in range(length):
            IEMG[index, :] = np.sum(df.iloc[0+3000*index:3000+3000*index, :])/3000

        for ch in range(4):
            Powerless[:, ch] = IEMG[:, ch]/np.sum(IEMG[:, 0:4], axis=1)
            Hold[:, ch] = IEMG[:, ch+4]/np.sum(IEMG[:, 4:8], axis=1)
            Open[:, ch] = IEMG[:, ch+8]/np.sum(IEMG[:, 8:12], axis=1)
            Palm[:, ch] = IEMG[:, ch+12]/np.sum(IEMG[:, 12:16], axis=1)
            Dorsi[:, ch] = IEMG[:, ch+16]/np.sum(IEMG[:, 16:20], axis=1)
            Chop[:, ch] = IEMG[:, ch+20]/np.sum(IEMG[:, 20:24], axis=1)

        '''学習用データ、検証用データ'''
        # train data
        X_train = np.block([[Powerless], [Hold], [Open], [Palm], [Dorsi], [Chop]])
        Y_ = np.ones(50, dtype="int64")
        Y_train = np.block([0*Y_, Y_, 2*Y_, 3*Y_, 4*Y_, 5*Y_])

        return X_train, Y_train

    def join(self):
        path_in = self.path_in
        window_size = self.window_size
        files = os.listdir(path_in)  # ファイル名取得
        files = sorted(files, key=lambda x: int((re.search(r"[0-9]+", x)).group(0)))  # 測定順に並べ替え

        t1 = time.time()
        # 関数の呼び出し
        data = self.readcsv(files, path_in)  # 計測データ(DataFrame)
        rmsdata = self.rms_editing(len(files), data, window_size)  # RMS処理後データ(DataFrame)
        t2 = time.time()
        elapsed_time = t2-t1
        print("\t 前処理時間\t"+str(elapsed_time)+"[s]")  # 経過時間
        return rmsdata
