from __future__ import print_function
from preprocess import preprocess
import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import sys
sys.path.append('D:\\NAGATA\\hand_recognition')

name = ["nagata", "noda", "shiba", "kirikihira", "asae2"]  # "fukumitu","izumine"
name_action = ["nagata_action", "noda_action", "shiba_action", "kirikihira_action", "asae_action"]
name_list = name + name_action
name_list = ["choppy", "hasegawa", "kaito", "manzen",
             "nagaoka", "nagata", "noda", "otsuka", "tanabe", "tsutsumi"]
data_labels = {0: "Relax", 1: "Hold", 2: "Open",
               3: "Palmar Flexion", 4: "Dorsal Flexion", 5: "Ulnar Flexion"}


def main():
    # number for CV
    fold_num = 5
    batch_size = 10
    num_classes = 6

    experiment = 'IEMG'

    co_epochs = [1]

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    Data = None
    Labels = None

    for name_ in name_list:
        # デスクトップのパス
        path_in = '../measurement/{}/'.format(str(name_))
        path_out = '../Result2/experiment1_FNN_{}/{}/'.format(experiment, str(name_))
        # path_in = '../hasegawa_csvdata/{}/'.format(str(name_))
        # path_out = '../Result_hasegawa/experiment1_FNN_{}/{}/'.format(experiment,str(name_))
        try:
            os.makedirs(path_out, exist_ok=True)
        except FileExistsError:
            pass
        data = preprocess(path_in)
        if experiment == 'IEMG':
            Data, Labels = data.iemgdata()
            input_dim = Data.shape[1]

        elif experiment == 'RMS':
            Data, Labels = data.emgdata()
            input_dim = Data.shape[1]

        # scikit_learn's cross-validation flow
        kfold = StratifiedKFold(n_splits=fold_num, shuffle=False, random_state=seed)

        for epochs in co_epochs:
            scores, conf_mat, accuracy_fold, loss_fold = feedforwardNN(kfold=kfold, X=Data, Y=Labels,
                                                                       epochs=epochs,
                                                                       input_dim=input_dim,
                                                                       batch_size=batch_size,
                                                                       num_classes=num_classes,
                                                                       path_out=path_out)

            # 保存
            path = path_out + 'ep{}_{}_conf.csv'.format(epochs, str(name_))
            pd.DataFrame(conf_mat).to_csv(path, mode='w')

            path = path_out + 'ep{}_{}_loss.csv'.format(epochs, str(name_))
            with open(path, mode='w') as f:
                write = csv.writer(f)
                write.writerows(loss_fold)

            path = path_out + 'ep{}_{}_accuracy.csv'.format(epochs, str(name_))
            with open(path, mode='w') as f:
                write = csv.writer(f)
                write.writerows(accuracy_fold)

            path = path_out + 'ep{}_{}_score.csv'.format(epochs, str(name_))
            with open(path, mode='w') as f:
                write = csv.writer(f)
                write.writerows(scores)
                f.write("\nloss,accuracy,precision,recall\n")


def feedforwardNN(kfold, X, Y, epochs, input_dim, batch_size, num_classes, path_out):
    cvscores = []
    accuracy_fold = []
    loss_fold = []
    conf_mat = np.zeros((6, 6))
    count = 0
    for train, test in kfold.split(X, Y):
        count += 1
        # create model
        model = Sequential()
        model.add(Dense(10, activation='sigmoid', input_dim=input_dim))
        model.add(Dense(6, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', Precision, Recall])

        # Fit the model
        history = model.fit(X[train], keras.utils.to_categorical(Y[train], num_classes),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0)

        # Evaluate
        scores = model.evaluate(X[test], keras.utils.to_categorical(
            Y[test], num_classes), verbose=0)
        print(list(zip(model.metrics_names, scores)))
        cvscores.append(scores)

        # Create loss,accuracy
        accuracy_fold.append(history.history['accuracy'])
        loss_fold.append(history.history['loss'])

        # saving model
        path = path_out + 'ep{}_model_{}.h5'.format(epochs, count)
        model.save(path)

        path = path_out + 'ep{}_model_{}_labels.csv'.format(epochs, count)
        path_ = path_out + 'ep{}_{}_report.csv'.format(epochs, count)

        # train
        predict_label_train = pd.DataFrame()
        predict_classes1 = pd.DataFrame(model.predict(X[train], batch_size=batch_size))
        predict_classes = model.predict_classes(X[train], batch_size=batch_size)
        predict_classes2 = pd.Series(predict_classes, name='predict label')
        true_classes_df = pd.Series(Y[train], name='true label')
        predict_label_train = pd.concat(
            [predict_classes1, predict_classes2, true_classes_df], axis=1)

        predict_label_train.to_csv(path, mode='w')
        print(predict_label_train)

        true_classes = Y[train]
        class_repo = classification_report(
            y_true=true_classes, y_pred=predict_classes, target_names=data_labels, output_dict=True)
        print(class_repo)
        with open(path_, mode='w') as f:
            f.write("train\n")
        pd.DataFrame(class_repo).T.to_csv(path_, mode='a')

        # test
        predict_label_test = pd.DataFrame()
        predict_classes1 = pd.DataFrame(model.predict(X[test], batch_size=batch_size))
        predict_classes = model.predict_classes(X[test], batch_size=batch_size)
        predict_classes2 = pd.Series(predict_classes, name='predict label')
        true_classes_df = pd.Series(Y[test], name='true label')
        predict_label_test = pd.concat(
            [predict_classes1, predict_classes2, true_classes_df], axis=1)
        predict_label_test.to_csv(path, mode='a')
        print(predict_label_test)

        true_classes = Y[test]
        conf_mat = conf_mat + np.array(confusion_matrix(true_classes, predict_classes))

        class_repo = classification_report(
            y_true=true_classes, y_pred=predict_classes, target_names=data_labels, output_dict=True)
        print(class_repo)
        with open(path_, mode='a') as f:
            f.write("\ntest\n")
        pd.DataFrame(class_repo).T.to_csv(path_, mode='a')

    cvscores = np.array(cvscores)
    accuracy_fold = np.array(accuracy_fold)
    loss_fold = np.array(loss_fold)
    print(conf_mat)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores[:, 1]*100), np.std(cvscores[:, 1]*100)))

    return cvscores, conf_mat, accuracy_fold, loss_fold


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
        emg3 = np.hstack((emg[:, emg_index[index % 4][:]],
                          emg[:, emg_index[(index+1) % 4][:]],
                          emg[:, emg_index[(index+2) % 4][:]]))
        emg_3ch.append(emg3)

    return np.array(emg_3ch)


def plot_result(history, path_out):
    '''
    plot result
    全ての学習が終了した後に、historyを参照して、accuracyとlossをそれぞれplotする
    '''

    # accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='acc', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.title('accuracy')
    path = path_out + '{}_graph_accuracy.png'
    plt.savefig(path)
    plt.show()

    # loss
    plt.figure()
    plt.plot(history.history['loss'], label='loss', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.title('loss')
    path = path_out + '{}_graph_loss.png'
    plt.savefig(path)
    plt.show()

    return

# precision


def Precision(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32'))

    precision = true_positives / (pred_positives + K.epsilon())
    return precision

# recall


def Recall(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.20), 'float32'))

    recall = true_positives / (poss_positives + K.epsilon())
    return recall

# f-measure


def F(y_true, y_pred):
    p_val = Precision(y_true, y_pred)
    r_val = Recall(y_true, y_pred)
    f_val = 2*p_val*r_val / (p_val + r_val)

    return f_val


if __name__ == '__main__':
    main()
