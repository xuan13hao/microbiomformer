import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '16'
sys.path.append(".")

import pandas as pd
import numpy as np
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
helper =  "call by\n    ~ <profile> <report_name> <sp_code> <feature1> <feature2> ..."
helper += "\t\t<profile>     - code for profile, must from {pfam, tax, k30}"
helper += "\t\t<report_name> - name to mark your selected set, can be any string without space"
helper += "\t\t<sp_code>     - code for training-testing split, must from {mix, dds3, ds3, ds12}"
helper += "\t\t                 mix,  [1,2,3|1,2,3], mix all samples from all three datasets"
helper += "\t\t                 dds3, [1,2,3|3], using all DS1 and DS2 in addition to part of DS3, same number of training sample as [1,2,3|1,2,3]"
helper += "\t\t                 ds3,  [3|3]"
helper += "\t\t                 ds12, [1,2|1,2]"
helper += "\t\t<feature1> <feature2> ...  "
helper += "                  - selected feature from given profile"

if len(sys.argv) > 4:
    pf = sys.argv[1]
    ss_name = sys.argv[2]
    ds = sys.argv[3]
    selected = sys.argv[4:]
else:
    print(helper)
    exit()

if ds not in ['mix', 'dds3', 'ds3', 'ds12']:
    print("you must choose <sp_code> from {mix, dds3, ds3, ds12}!")
    print(helper)
    exit()

if pf not in ['pfam', 'k30', 'tax']:
    print("you must choose <profile> from {pfam, tax, k30}!")
    exit()

# if pf == 'k30':
#     for i,v in enumerate(selected):
#         selected[i] = int(v)


top = 100

"""== Parameters ============================================================"""
ff_skipNormalize = False
map_class = {"PR":1, "CR":1, "PD":0, "DD":0, "SD":1}
base_dir   = f"../data"
work_space = f"{base_dir}/TestUsingGivenProfile"
os.system("mkdir -p " + work_space) # create work_space if not exist
"""== Utility ==============================================================="""

def getDS(labels):
    gt = np.zeros(len(labels))
    for (i, v) in enumerate(labels):
        if v[0] == 'J':
            gt[i] = 0
        elif v[0] == 'E':
            gt[i] = 1
        elif v[0] == 'S':
            gt[i] = 2
    return gt

def getGT(labels):
    gt = np.zeros(len(labels))
    for (i, v) in enumerate(labels):
        gt[i] = map_class[v[-2:]]
    return gt

def normalize_feature(df):
    ans = []
    df_std = df.std()
    df_mean = df.mean()
    for col in df.columns:
        ans.append((df[col] - df_mean[col]) / (df_std[col]+1e-17))
    return pd.concat(ans, axis=1)

def normalize_test(df, df_train):
    ans = []
    df_std = df_train.std()
    df_mean = df_train.mean()
    for col in df.columns:
        ans.append((df[col] - df_mean[col]) / (df_std[col]+1e-17))
    return pd.concat(ans, axis=1)

renameSample_test = {}
renameSample_pfs = {}
stable_list = []
with open(f"{base_dir}/metadata.csv") as fin:
    for line in fin:
        if not line[0] == '#':
            cont = line.strip().split(',')
            renameSample_test[cont[0]] = cont[0] + '_' +cont[5]
            if len(cont[12]) > 0:
                renameSample_pfs[cont[0]] = cont[0] + '_' +cont[12]
            if cont[5] == 'SD':
                stable_list.append(cont[0] + '_' +cont[5])

training_sample = [renameSample_test[x] for x in renameSample_pfs]

acc_length_pfam = {}
acc_name_pfam = {}
with open(f"{base_dir}/pfam.acclengname.csv", 'r') as fin:
    for line in fin:
        cont = line.strip().split(',')
        acc_length_pfam[cont[0]] = float(cont[1])/1000
        acc_name_pfam[cont[0]] = cont[2]

def loadPFAM(fname, ff_fast=False):
    df_pfam = pd.DataFrame(pd.read_csv(fname, sep='\t', index_col=0))
    if not ff_fast:
        for col in df_pfam.columns:
            df_pfam[col] = df_pfam[col]*1e6 / df_pfam[col].sum()
        for row in df_pfam.index:
            df_pfam.loc[row] = df_pfam.loc[row] / acc_length_pfam[row]
    df_pfam.sort_index(axis=1, inplace=True)
    return df_pfam

def loadTAX(fname):
    df_tax  = pd.DataFrame(pd.read_csv(fname, sep=',', index_col=0))
    df_tax.sort_index(axis=1, inplace=True)
    return df_tax

def loadKMC(fname):
    df_kmc  = pd.DataFrame(pd.read_csv(fname, sep='\t', index_col=0))
    df_kmc.sort_index(axis=1, inplace=True)
    return df_kmc

if pf == 'pfam':
    df_pfam = loadPFAM(f"{base_dir}/pfam.count.tsv", ff_skipNormalize)
    df_pfam.rename(columns=renameSample_test,inplace=True)
    map_data = df_pfam
elif pf == 'tax':
    df_tax = loadTAX(f"{base_dir}/otu.count.csv")
    df_tax.rename(columns=renameSample_test,inplace=True)
    map_data = df_tax
elif pf == 'k30':
    df_kmc = loadKMC(f"{base_dir}/k30.count.tsv")
    df_kmc.rename(columns=renameSample_test,inplace=True)
    map_data = df_kmc


"""== Test bestK ============================================================"""
def pickFeature(df, select):
    if len(set(df.columns) & set(select)) == 0:
        print("None of the selected feature was found in data")
        return df
    return df.T.reindex(select).fillna(0).T.copy()

def my_training(XX, yy, X, y):
    max_mcc = -2
    mark = -1
    for i in range(0,1000):
        try:
            clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=400, random_state=i, early_stopping=True)
            clf.fit(XX, yy)
            p_test = clf.predict(X)
            mcc = f1_score(y, p_test)
            if mcc > max_mcc:
                max_mcc = mcc
                mark = i
        except:
            print("bad random_state: {}".format(i))
    return mark

def my_training_testing_split(df_join, ds, rs_sp=None):
    Y = getGT(df_join.index)
    ds_label = getDS(df_join.index)
    if ds == 'dds3':
        df_train = df_join[ds_label!=2]
        y_train = Y[ds_label!=2]
        df_join = df_join[ds_label==2]
        Y = Y[ds_label==2]
        N = len(ds_label)
    elif ds in ['ds1', 'ds2', 'ds3']:
        df_join = df_join[ds_label==(int(ds[-1])-1)]
        Y = Y[ds_label==(int(ds[-1])-1)]
        df_train = pd.DataFrame()
        y_train = np.array([])
        N = len(Y)
    elif ds == 'ds12':
        df_join = df_join[ds_label!=2]
        Y = Y[ds_label!=2]
        df_train = pd.DataFrame()
        y_train = np.array([])
        N = len(Y)
    else:
        df_train = pd.DataFrame()
        y_train = np.array([])
        N = len(Y)
    if not rs_sp or rs_sp < 0:
        rs_sp = np.random.randint(9999999)
    else:
        rs_sp = int(rs_sp)
    kf = StratifiedShuffleSplit(n_splits=1, test_size=int(np.ceil(0.3*N)), random_state=rs_sp)
    for T_index, t_index in kf.split(df_join, Y):
        df_tmp , y_tmp   = df_join.iloc[T_index], Y[T_index]
        df_test,  y_test = df_join.iloc[t_index], Y[t_index]
    df_train = pd.concat([df_train, df_tmp])
    y_train = np.concatenate((y_train, y_tmp))
    return (rs_sp, df_train, y_train, df_test, y_test)


def crossValidateSingle():
    df_join = pickFeature(map_data.T.copy(), selected)
    fout = open(f"{work_space}/{pf}_{ds}_{ss_name}_prob.csv", 'w')
    fout.close()
    if os.path.exists(f"{work_space}/{ds}_sp.csv"):
        df_rs_states = pd.read_csv(f"{work_space}/{ds}_sp.csv", dtype=int)
    else:
        df_rs_states = pd.DataFrame({"rs_sp":[np.random.randint(9999999) for x in range(top)], "rs_model":np.zeros(top)}, dtype=int)

    for index,row in df_rs_states.iterrows():
        rs_sp = row['rs_sp']
        (rs_sp, df_train, y_train, df_test, y_test) = my_training_testing_split(df_join, ds, rs_sp)
        print(rs_sp, df_train.shape, df_test.shape, set(y_train), set(y_test))
        X_train = normalize_feature(df_train)
        X_test  = normalize_test(df_test, df_train)
        rs_model = my_training(X_train, y_train, X_test, y_test)
        print(rs_model)
        clf = MLPClassifier(hidden_layer_sizes=(100,100), random_state=rs_model, max_iter=400, early_stopping=True)
        clf.fit(X_train, y_train)
        # random states
        p_test = clf.predict(X_test)
        df_rs_states.loc[index, 'rs_sp']    = rs_sp
        df_rs_states.loc[index, 'rs_model'] = rs_model

        # Probability
        pred = clf.predict_proba(X_test)
        l1 = ""
        l2 = ""
        for it in sorted(zip(pred[:,1], y_test)):
            l1 += str(it[0]) + ','
            l2 += str(int(it[1])) + ','

        fout = open(f"{work_space}/{pf}_{ds}_{ss_name}_prob.csv", 'a')
        fout.write(l1[:-1]+'\n')
        fout.write(l2[:-1]+'\n')
        fout.close()

    df_rs_states.to_csv(f"{work_space}/{ds}_sp.csv", index=None)

crossValidateSingle()
