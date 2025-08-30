from mymicrograd import scalar,Neuron,Layer,MLP
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
random.seed(42)
np.random.seed(42)
X_moons,y_moons=make_moons(n_samples=100,noise=0.1,random_state=0)
y_moons=y_moons*2-1
model=MLP(2,[4,4,1])
def loss():
    scores=[model(x) for x in X_moons]
    losses=[(1+ -yi*si).relu() for si,yi in zip(scores,y_moons)]
    data_loss=sum(losses)/len(losses)
    accuracy=[(yi>0)==(scorei.data>0)for yi,scorei in zip(y_moons,scores)]
    acc=sum(accuracy)/len(accuracy) 
    return data_loss,acc
for k in range(1000):
    data_loss,acc=loss()

    for p in model.parameters():
        p.grad=0.0
    data_loss.backward()
    for p in model.parameters():
        p.data -= 0.1*p.grad
    print(f"step{k} loss {data_loss.data} accuracy {acc*100}%")
    
import numpy as np
xx,yy=np.meshgrid(
    np.linspace(X_moons[:,0].min()-0.5,X_moons[:,0].max()+0.5,200),
    np.linspace(X_moons[:,1].min()-0.5,X_moons[:,1].max()+0.5,200)
)
grid=np.c_[xx.ravel(),yy.ravel()]
scores=[model([xi,yi]) for xi,yi in grid]
Z=np.array([s.data for s in scores]).reshape(xx.shape)
plt.contourf(xx,yy,Z>0,alpha=0.4,cmap='coolwarm')
plt.contour(xx,yy,Z,levels=[0],colors='k',linewidths=2)
plt.scatter(X_moons[:,0],X_moons[:,1],c=y_moons,cmap='coolwarm',edgecolor='k')
plt.title(f"Decision boundary (Accuracy:{acc*100}%)")
plt.xlabel("x1")
plt.ylabel("y1")
plt.show()
