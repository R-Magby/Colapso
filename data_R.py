 
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import tensorflow as tf
T=10000;L=10000
rango=200.0
epsilon=0.0001
dr=rango/(L)
#data=np.loadtxt("campo_escalar.dat").reshape((T,L))
r_0=20.0
sigma2=1.5*1.5
A=0.2
k=3.1415*8.0
inputs=tf.keras.Input(shape=(1,),dtype=tf.float32)
capa1=tf.keras.layers.Dense(units=64,activation=tf.tanh)(inputs)
capa2=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa1)
capa3=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa2)
capa4=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa3)#units = cantidad de naurona
outputs=tf.keras.layers.Dense(units=1,activation=None)(capa4)
model2=tf.keras.Model(inputs,outputs)

model2.load_weights('model.h5')
r=np.linspace(0,200,L).astype('float32')
data=np.zeros(L)
lista=model2(r)
for i in range(L):
    data[i]=lista[i,None]

print(data)
plt.plot(r,data,"-r")
plt.show()
data=data.astype(np.float32)
data=data-data[0]
print(data.dtype,)
fh = open("test2.npy", "bw")
data.tofile(fh)

