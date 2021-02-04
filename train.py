import pandas as pd
from Token import Token
from gensim.models import word2vec
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from Metric import *
from tensorflow import keras
import numpy as np
from Model import AMCNN
import argparse
from tensorflow.keras import backend as K
import os

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Train Model Option")
parser.add_argument('--epochs',"-e",type=int,help="Train epoch size",default=30)
parser.add_argument('--train_steps',"-t_steps",type=int,help="Train epoch size",default=200)
parser.add_argument('--batch_size',"-b", type=int, help="Train batch size",default=64)
parser.add_argument('--max_length',"-max_len", type=int, help="Max length of sequence",default=100)
parser.add_argument("--lr_rate","-lr",type=float,help="Train learning rate",default=0.001)
parser.add_argument("--lr_decay","-decay",type=float,help="Train learning rate decay factor new_lr = lr*decay",default=0.9)
parser.add_argument("--patience",type=int,help="Number of epochs with no improvement after which learning rate will be reduced",default=5)
#parser.add_argument("--save_period","-period", type=int, help="Train model save weight term",default=1)
parser.add_argument("--att_reg","-reg",type=float,help="L2 regularizer term of Attention Layer",default=0.0001)

parser.add_argument("--channel", type=int, help="Number of Attention Layer Channels",default=2)
parser.add_argument('--weight_save_path',type=str,help="Train weights save path",default="Weights")
parser.add_argument('--train_data',type=str,help="",required=True)
parser.add_argument('--document',type=str,help="Variable name of document column",required=True)
parser.add_argument('--label',type=str,help="Variable name of label column",required=True)

args = parser.parse_args()


def main():
    # Check Gpu Enable
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # parsing Arg
    train_data_path = args.train_data
    max_len = args.max_length
    epoch = args.epochs
    batch_size = args.batch_size
    att_reg = args.att_reg
    lr_decay = args.lr_decay
    lr_rate = args.lr_rate
    warmup_lr_rate = lr_rate*0.1
    patience = args.patience
    #period = args.save_period
    weight_save_path = args.weight_save_path
    document = args.document
    label = args.label
    channel = args.channel
    steps_per_epoch = args.train_steps

    # model weight pah
    if os.path.isdir(weight_save_path) == False:
        os.mkdir(weight_save_path)

    # Read Data
    if ".csv" in train_data_path:
        read_data = pd.read_csv
    elif ".xlsx" in train_data_path:
        read_data = pd.read_excel
    else:
        read_data = pd.read_table
    train_data = read_data(train_data_path)

    # Make Tokenizer Token
    tk = Token("Tokenizer", max_len)
    train_data["Token"] = train_data[document].apply(lambda x: tk.make_token_ori(x))

    # Using Keras Tokenizer
    k_tokenizer = keras.preprocessing.text.Tokenizer(filters='')
    k_tokenizer.fit_on_texts(train_data["Token"].values.tolist())
    words_count = len(k_tokenizer.word_counts)
    print("Save Keras tokenizer for validate in %s"%(weight_save_path))
    with open(os.path.join(weight_save_path,"keras_tokenizer.pkl"), "wb") as f:
        pickle.dump(k_tokenizer, f)
    # Load Pre-trained embedding Word2Vec
    w2v_model = word2vec.Word2Vec.load("w2v_pretrain_emb/w2v_20M_500.model")
    init_weight = np.random.uniform(size=(words_count + 1, 500), low=-1, high=1)

    words_lst = []
    for i in range(1, len(k_tokenizer.index_word) + 1):
        words = k_tokenizer.index_word[i]
        try:
            words_vector = w2v_model.wv[words]
        except:
            words_lst.append([i, words])
        init_weight[i] = words_vector

    #  K_tokenizer Sequence
    sequences = k_tokenizer.texts_to_sequences(train_data['Token'])
    x_train = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
    y_train = train_data[label].values

    # Define validation set 분할
    x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    # Build simple binary model
    tf.keras.backend.clear_session()
    amcnn = AMCNN(maxlen=max_len,
                  embed_dim=500,
                  words_count=words_count,
                  filter_size=50,
                  channel=channel,
                  mask_prob=0.5,
                  att_reg=att_reg)
    model = amcnn.build(emb_trainable=False, emb_weight=init_weight)

    model.compile(optimizer=tf.keras.optimizers.Adam(warmup_lr_rate), loss="binary_crossentropy",
                         metrics=["accuracy", k_precision, k_recall, k_f1score])
    checkpoint_path = os.path.join(weight_save_path,"model-{epoch:04d}.h5")

    # Define callbacks condition
    callbacks = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                verbose=1,save_best_only=True, save_weights_only=True) #  period=period,


    # Define reduce Learning rate schedule
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=lr_decay,
                                   cooldown=0,
                                   patience=patience,
                                   min_lr=lr_rate*0.01,
                                   verbose=1)


    # Train Warm up stage
    print("===========Warm up %d Epoch Stage==========="%(int(epoch*0.1)))
    # warm up embedding weight
    model.fit(x_train2, y_train2, epochs=int(epoch*0.1), callbacks=[callbacks, lr_reducer], steps_per_epoch=steps_per_epoch,
                     batch_size=batch_size,verbose=2)
    print("============Main %d Epoch Stage============="%(epoch-int(epoch*0.1)))
    K.set_value(model.optimizer.learning_rate, lr_rate)
    model.fit(x_train2, y_train2, epochs=epoch-int(epoch * 0.1), callbacks=[callbacks, lr_reducer], steps_per_epoch=steps_per_epoch
              ,validation_steps=steps_per_epoch,
              batch_size=batch_size, validation_data=(x_val, y_val),verbose=2)
    print("Complete Training Model")
    print("Check Model Weight file in %s"%(weight_save_path))

if __name__ == "__main__":
    main()
