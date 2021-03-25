import pandas as pd 
import numpy as np 
import scipy as scp
import math 
import func as fct 
import matplotlib.pyplot as plt 
import csv 
from sklearn.linear_model import LogisticRegression
import random

"""
	Ouverture du CSV
"""
#path = 'C:/Users/jchaudron/Downloads/nettoye.csv'
#path = '/home/chaudron/Documents/Lyon/Infomod/Programmation/K-NN/TD1/got.csv'

#df = pd.read_csv(path,sep=";")

path='normalise.csv'
with open(path,newline='') as file:
	rid = csv.reader(file,delimiter=',')
	mat = list(rid)
got = {mat[0][x] : [mat[y][x] for y in range(len(mat)) if y != 0] for x in range(len(mat[0])) if not x in [0,1]}


for x in got.keys():
	got[x] = np.array(got[x]).astype(np.float)


"""
	Création de la variable "dim" qui permettra, lors de la mesure des distance, de ne sélectionner que les 
	dimensions sur lesquelles ont veut mesurer la distance
"""


cor = [[scp.stats.pearsonr(got[x],got['isAlive'])[0],x] for x in got.keys() if x != 'isAlive']
cor.sort(key = lambda x: abs(x[0])) #On ordonne les dimensions par leur correlation avec isAlive

garde = random.sample(range(len(got['isAlive'])),50)
ngarde = [x for x in range(len(got['isAlive'])) if not x in garde]
leave = {a : [got[a][b] for b in garde] for a in got.keys()}
train = {a : [got[a][b] for b in ngarde] for a in got.keys()}

lr = LogisticRegression(solver='liblinear').fit(np.transpose([train[x] for x in train.keys() if x != 'isAlive']),train['isAlive'])
coef = lr.coef_
print(coef)
coef = {list(train.keys())[a] : abs(coef[0][a]) for a in range(len(train.keys())-1)}
cor = [x[1] for x in cor]


clf = fct.K_NN(train,coef,cor[:7])
pmf = fct.Poisson(got,cor[:7],coef)

pmf.Densité(5)
clf.seuil(leave,pmf)



"""
	On récupère l'index du nom qui nous intéresse
	On pourrait faire une fonction ou bien un DataFrame qui aurait ça à la place
"""
"""
#df_dist = pd.DataFrame(index = df['name'],columns=['Arya Stark'])

df_dist = fct.normalisation(df_dist)
print("ok")
"""
#df_dist = fct.distance(dimdim,fct.index(['Arya Stark'],df)[0],df.index,df,df_dist)
#print(df_dist)


"""
	Récupération des coefficients de la régression logistique pour pondérer les dimensions
"""



"""
regr = fct.RegLog([got[cor[x][1]] for x in range(len(cor))],got['isAlive'],[got[cor[x][1]] for x in range(len(cor))])

coef = regr.ajust.coef_
print(coef[0])
print(got)
"""




"""
	Leave one out
"""

"""
ind_mort = [x for x in df.index if df.loc[x,'isAlive'] == 0]
df.drop(labels = fct.index(['Jorah Mormont'],df),axis = 0,inplace=True)

take = np.random.choice(maure,5)
ones_out = pd.DataFrame(data = [df.loc[x,cor_ord[:5]] for x in choixe],index = [df.loc[x,'name'] for x in choixe],columns = cor_ord)
ones_in = pd.DataFrame(data = [df.loc[x,cor_ord[:5]] for x in df.index if not x in choixe],index = [df.loc[x,'name'] for x in df.index if not x in choixe],columns = cor_ord)


clas_nn = fct.k_nn(ones_in,df,truc=None)
clas_nn.normalisation()

"""
"""
	Création des estimateurs Bayésiens
"""
"""
"""