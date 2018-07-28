from Tensorflow_autoencoder import Autoencoder
import tensorflow as tf
import os
from datetime import datetime 

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data_dir='../Data/'

df_data=pd.read_csv("../Data/merged_final_data", sep=',', encoding='latin1', error_bad_lines=False)
print('file_loaded..')

d=df_data.loc[:, ~df_data.columns.str.contains('^Unnamed: 0')]
df = d[['SYSBP','DBP','HR','TEMP','WEIGHT','HEIGHT']]


epocha=1000
batch_size=35
iterations=int(len(df)//35)

def scale_data(df_d):
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(df_d)

    return scaled_data

# print(scale_data(df))

def get_batch(i,batch_size=35):
    batch_slice=df[i*batch_size:(i+1)*batch_size]
#     print([list(i) for i in batch_slice])
    scaler = MinMaxScaler()
    scaled_data=scaler.fit_transform(batch_slice)
#     print("vvlaues",datae)
    return {'data':np.array(scaled_data)}



# #     return [list(i) for i in batch_slice]

def evaluate_model(model):
    sess = tf.get_default_session()
    train_batch_mse = sess.run(model.test, feed_dict={model.placeholder['input']: scale_data(df)})

    return {'batch_mse':train_batch_mse}



def train_model(model):
    now = datetime.now()
    save_model = os.path.join(data_dir, 'tensorflow_model.ckpt')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(epocha):
            for i in range(iterations):
                batch_xs = get_batch(i)['data']
                c = sess.run(model.output, feed_dict={model.placeholder['input']: batch_xs})
                print(c['cost'])

        # Display logs per epoch step
            if epoch % 5 == 0:
                print(evaluate_model(model))

        print("Optimization Finished!")
    
        save_path = saver.save(sess, save_model)
        print("Model saved in file: %s" % save_path)

if __name__ == "__main__":
    model = Autoencoder(6, 8, 3)

    train_model(model)









