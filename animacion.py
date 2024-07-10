import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt

T=20000;L=10000
#data=np.loadtxt("campo_escalar.dat").reshape((T,L))
with open("campo_escalar_colapso.dat") as f:
    data = np.fromfile(f, dtype=np.dtype(np.float32))

data=data.reshape((T,L))
x=list(range(50));y=list(range(50))
dx=200/L
limite=int(100/dx)
posicion=np.linspace(0.0,200,L)[:limite]
 
fig = plt.figure()
axl = plt.axes()
maximo=np.max(data)
axl.set_ylim(-maximo,maximo)

print(limite)
data=data[:,:limite]
line,=axl.plot(posicion,data[0,:],"-k" )

def animate(i):
    line.set_ydata(data[i*20,:])
    plt.legend([i*20])
    return line,
     
 
anim = FuncAnimation(fig, animate, frames=5000,interval = 20)
#anim.save('colapso_final_0.01.gif')
plt.show()