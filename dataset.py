import numpy as np
import pickle
import random

class DataSet(object):
    def __init__(self, path, unit_num = 1000, rate = 0.8, num_seen_class=8, seed=2019):
        self.path = path
        self.unit_num = unit_num
        self.num_seen_class = num_seen_class
        self.seed = seed

        self.load_dataSet(path, self.unit_num)
        self.split_train_test(rate,num_seen_class, seed)

    def get_meta(self):
        return self.X, self.lbl, self.snrs, self.mods

    def get_train_test(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test, self.train_class

    def get_train_test_maps(self):
        return self.train_map,self.test_map,self.unknown_test_map

    def load_dataSet(self, path, unit_num):
        Xd = pickle.load(open(path, 'rb'), encoding='iso-8859-1')

        # snrs(20) = -20, ... , 18  mods(11) = ['8PSK', 'AM-DSB', ...]
        self.snrs, self.mods = map(
            lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

        self.X = []
        self.lbl = []

        print(self.mods)
        print(self.snrs)

        self.class_count=[0]
        count=0

        for mod in self.mods:
            for snr in self.snrs:
                if not snr in (18,16):
                    continue

                single_data=Xd[(mod, snr)][:]
                self.X.append(single_data)
                count+=single_data.shape[0]

                print('Shape:{}'.format(single_data.shape[0]))
                self.lbl.extend([(mod, snr) for _ in range(single_data.shape[0])])

            self.class_count.append(count)

        self.X = np.vstack(self.X)
        print(self.X.shape)

    def to_onehot(self, yy):
        yy1 = np.zeros([len(yy), len(self.mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    def split_unknown(self,rate=0.8):
        untrain_class=list(set(range(0,len(self.mods)))-set(self.train_class))
        print(untrain_class)

        self.unknown_test_indices=[]
        self.unknown_test_map={}

        for idx, certain_class in enumerate(untrain_class):
            class_indices=list(range(self.class_count[certain_class],self.class_count[certain_class+1]))
            unknown_test_class_idx=list(np.random.choice(class_indices, size=int(len(class_indices)*(1-rate)+1), replace=False))
            self.unknown_test_map[idx + len(self.train_class)]=self.X[unknown_test_class_idx]
            self.unknown_test_indices+=unknown_test_class_idx

        print(len(self.unknown_test_indices))

    def split_train_test(self, rate=0.8, num_seen_class = 8, seed=2019):
        np.random.seed(seed)

        train_class=list(set(range(11))-set([2,5]))
        self.train_class=train_class

        print(train_class)

        self.train_indices=[]
        self.test_indices=[]
        self.train_map={}
        self.test_map={}

        for idx, certain_class in enumerate(train_class):
            class_idx=list(range(self.class_count[certain_class],self.class_count[certain_class+1]))
            class_train_idx=list(np.random.choice(class_idx, size=int(len(class_idx)*rate), replace=False))
            class_test_idx=list(set(class_idx)-set(class_train_idx))
            self.train_map[idx]=self.X[class_train_idx]
            self.test_map[idx]=self.X[class_test_idx]
            self.train_indices += class_train_idx
            self.test_indices += class_test_idx

        random.shuffle(self.train_indices)
        random.shuffle(self.test_indices)
        self.X_train=self.X[self.train_indices]
        self.X_test = self.X[self.test_indices]

        self.Y_train = self.to_onehot(list(map(lambda x: train_class.index(self.mods.index(self.lbl[x][0])), self.train_indices)))
        self.Y_test = self.to_onehot(list(map(lambda x: train_class.index(self.mods.index(self.lbl[x][0])), self.test_indices)))

        return self.X_train, self.Y_train, self.X_test, self.Y_test, self.mods
