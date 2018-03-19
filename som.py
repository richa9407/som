import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#Importing datasets
dataset = pd.read_csv('/Users/savita/desktop/australian.csv')
x= dataset.iloc[: , :-1].values
y=dataset.iloc[: , -1].values

#Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
x = sc.fit_transform(x)

#Training the som 
from minisom import MiniSom 
som = MiniSom(x=10 , y=10 , input_len =15 , sigma=1.0 , lr=0.5 , decay_function = None )
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

from pylab import bone, pcolor , colorbar, plot , show 
bone()
pcolor(som.distance_map().T)
colorbar()
markers = [ 'o' , 's']
colors = [ 'r' ,'g']
for i, x in enumerate(x):
	w = som.winner(x)
	plot(w[o] +0.5 , w[1] +0.5 , markers[y[i]], colors[y[i]] , markeredgecolor = colors[y[i]] , markersize= 10 , markeredgewidth = 2)
show()

mappings =  som.win_map(x)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]) , axis=0)
frauds = sc.inverse_transform(frauds)


