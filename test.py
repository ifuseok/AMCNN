import pandas as pd
from Token import Token
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from tensorflow import keras
import tensorflow as tf
import numpy as np
from Model import AMCNN
import argparse
import os

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Test Model Option")
parser.add_argument('--max_length',"-max_len", type=int, help="Max length of sequence",default=100)
parser.add_argument("--att_reg","-reg",type=float,help="L2 regularizer term of Attention Layer",default=0.0001)
parser.add_argument("--channel", type=int, help="Number of Attention Layer Channels",default=2)
parser.add_argument('--weight_save_path',type=str,help="Train weights save path",default="Weights")
parser.add_argument('--val_model_epoch',"-val_model",type=int,help="Which Epoch Model to use? -1 means using last weihgts",default=-1)
parser.add_argument('--test_data',type=str,help="",required=True)
parser.add_argument('--document',type=str,help="Variable name of document column",required=True)
parser.add_argument('--label',type=str,help="Variable name of label column",required=True)

args = parser.parse_args()


def main():
    # Check Gpu Enable
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # parsing Arg
    test_data_path = args.test_data
    max_len = args.max_length
    att_reg = args.att_reg
    weight_save_path = args.weight_save_path
    document = args.document
    label = args.label
    channel = args.channel
    val_model_epoch = args.val_model_epoch

    # Read Data
    if ".csv" in test_data_path:
        read_data = pd.read_csv
    elif ".xlsx" in test_data_path:
        read_data = pd.read_excel
    else:
        read_data = pd.read_table
    test_data = read_data(test_data_path)

    # Make Tokenizer Token
    tk = Token("Tokenizer", max_len)
    test_data["Token"] = test_data[document].apply(lambda x: tk.make_token_ori(x))

    # Using Keras Tokenizer
    print("Load Keras tokenizer for validate in %s"%(weight_save_path))
    with open(os.path.join(weight_save_path,"keras_tokenizer.pkl"), "rb") as f:
        k_tokenizer = pickle.load(f)
    words_count = len(k_tokenizer.word_counts)

    #  K_tokenizer Sequence
    sequences = k_tokenizer.texts_to_sequences(test_data['Token'])
    x_test = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
    y_test = test_data[label].values

    # Build simple binary model
    tf.keras.backend.clear_session()
    amcnn = AMCNN(maxlen=max_len,
                  embed_dim=500,
                  words_count=words_count,
                  filter_size=50,
                  channel=channel,
                  mask_prob=0.5,
                  att_reg=att_reg)
    model = amcnn.build(pre_emb=False)
    if val_model_epoch == -1:
        model_lst = [i for i in os.listdir(weight_save_path) if ".h5" in i]
        model_weight_path = model_lst[-1]
    else:
        model_weight_path = "model-%4d.h5"%(val_model_epoch)
        model_weight_path = model_weight_path.replace(" ","0")
    model.load_weights(os.path.join(weight_save_path,model_weight_path))
    print("Evaluate %s Test data"%(os.path.join(weight_save_path,model_weight_path)))
    pred_test = model.predict(x_test,verbose=1)
    pred_test2 = np.int32(pred_test >= 0.5).reshape(-1)
    print("==============Evaluate Result============")
    print("f1_score :", f1_score(y_test, pred_test2))
    print("acc_score :", accuracy_score(y_test, pred_test2))
    print("recall_score :", recall_score(y_test, pred_test2))
    print("precision_score :", precision_score(y_test, pred_test2))

if __name__ == "__main__":
    main()
