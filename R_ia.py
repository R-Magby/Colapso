import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import tensorflow as tf

@tf.function
def edo_func(modelo,r,S):
  alpha=0.001
  k=3.1415*8.0
  mol_R=modelo(r)
  dR=tf.gradients(mol_R,r)[0]
  ddR=tf.gradients(dR,r)[0]
  y_nau=ddR+(1-dR*dR)/(2.0*mol_R+alpha)-0.25*(k*(S*S))
  return y_nau

#funcion de coste
def perdida(modelo,r,r_0,S):
  y_nau=edo_func(modelo,r,S)
  y_inicial=cond_inicial(modelo,r_0)
  return tf.reduce_sum(tf.square(y_nau))+tf.reduce_sum(tf.square(y_inicial[0]))*2.0+tf.reduce_sum(tf.square(y_inicial[1]))*2.0

#gradietne de la perdida
def perdida_gradiente(modelo,r,r_0,S):
  with tf.GradientTape() as tape:
    loss=perdida(modelo,r,r_0,S)
  grand=tape.gradient(loss,modelo.trainable_weights)
  return [loss,grand]


@tf.function
def cond_inicial(modelo,r_0):

  R_0_mol=modelo(r_0)
  dR_0_mol=tf.gradients(R_0_mol,r_0)[0]

  R_0_real=0.0
  dR_0_real=1.0
  return [R_0_mol - R_0_real , dR_0_mol - dR_0_real]

def optimizar(modelo,r,r_0,S,N=1000):
  optimi=tf.keras.optimizers.Adam()
  err=[]
  for i in list(range(N)):
    grad=perdida_gradiente(modelo,r,r_0,S)
    loss=grad[0]
    value=grad[1]
    optimi.apply_gradients(zip(value,modelo.trainable_weights))
    if i%100==1:
      print("Entrenamiento :",i,", perdida :",float(loss))
    err.append(loss)
  return err


T=10000;L=10000
rango=200.0
epsilon=0.0001
dr=rango/(L)
#data=np.loadtxt("campo_escalar.dat").reshape((T,L))
r_0=20.0
sigma2=1.5*1.5
A=0.2
k=3.1415*8.0
r=np.linspace(0,100,L).astype('float32')
chi=-2.0*A*(r-r_0)/sigma2 * np.exp(-(r-r_0)*(r-r_0)/sigma2)
S=np.zeros(L).astype('float32')
for i in range(1,L-1):
    S[i]=(chi[i+1]-chi[i-1])/(2.0*dr)



inputs=tf.keras.Input(shape=(1,),dtype=tf.float32)
capa1=tf.keras.layers.Dense(units=64,activation=tf.tanh)(inputs)
capa2=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa1)
capa3=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa2)
capa4=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa3)#units = cantidad de naurona
outputs=tf.keras.layers.Dense(units=1,activation=None)(capa4)#dense : neurona que esta conectada con todas las anteriores



r=r[:,None] 
#idx=[np.random.choice(t.shape[0],20, replace=False)]
#t_col=t_col[idx,:][0]

model2=tf.keras.Model(inputs,outputs)
model2.save_weights('model.h5')
print(1/dr)
for i in range(int(rango)):
  model2.load_weights('model.h5')
  r=np.linspace(0.0,(i+1),int((i+1)/dr)).astype('float32')[:None]

  chi=-2.0*A*(r-r_0)/sigma2 * np.exp(-(r-r_0)*(r-r_0)/sigma2)
  S=np.zeros(int((i+1)/dr)).astype('float32')
  for m in range(0,int((i+1)/dr)):
    if m==int((i+1)/dr)-1:
      S[m]=(chi[m]-chi[m-1])/(dr)
    elif i==0:
      S[m]=(chi[m+1]-chi[m])/(dr)
    else:
      S[m]=(chi[m+1]-chi[m-1])/(2.0*dr)

  opt=optimizar(model2,r,np.array([0.0,],dtype=np.float32)[:,None],S[:None],1*10**3)
  model2.save_weights('model.h5')



plt.plot(r,model2(r),"--r")
plt.legend(("NN","seno(t)"))
plt.show()
plt.plot(list(range(len(opt))),opt)
plt.show()