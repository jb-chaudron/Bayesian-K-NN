import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as st 
from sklearn.linear_model import LogisticRegression
import math
import random
import itertools as itr
import time 
import csv
from operator import mul
from sklearn.metrics import accuracy_score



def normalisation(data):
    """
        Les données en entrées sont déjà normalisée, mais je laisse la fonction quand même
    """
    new_dat = {a : [] for a in data.keys()}
    col = [x for x in data.keys() if x != 'isAlive']

    for i in col:
        if not isinstance(data[i][0],np.float):
            moy = {a : sum([new_dat['isAlive'][x] for x in range(len(new_dat['isAlive'])) if new_dat[i][x] == a]) for a in set(new_dat[i])}
            new_dat[i] = [moy[a] for a in new_dat[i]]   
        else:
            pass
            
        maxi = max(data[i])
        mini = min(data[i])
        new_dat[i] = [2*(x-(maxi+mini)/2)/(maxi+mini) if x != np.nan else 0 for x in data[i] ]

    new_dat['isAlive'] = data['isAlive']
    return new_dat


class K_NN:
    """Objet K-nn"""
    def __init__(self,data,coef,dim=None):
        """
            data.vrai : les vraie données qui contiennent tout même le isAlive
            data : les données sur lesquelles ont se base pour le K-NN
            data_dist : le dataframe qui va 
        """
        self.data = data
        self.dst = []
        self.dim = dim
        self.mort = sum([1 if x == 0 else 0 for x in data['isAlive']])/len(data['isAlive'])
        self.nbmrt = None
        self.coef = coef

    
    def distance(self,fix,var):
        """
            fix : la personne que l'on va comparer à tout le tableau
            var : les index des personnes que l'on veut classer
        """
        for a in var:
            delta = [pow(fix[x]-self.data[x][a],2) for x in self.dim]#On divise car on utilise les coefs de la regression logistique qui augmentent
                                                                                     #la valeur des dimensions importantes au lieu de la réduire
            self.dst += [math.sqrt(sum(delta))]
        self.dst = [x for x in enumerate(self.dst)]



            
    def seuil(self,arra,poiss,k=None,trig=None):
        """
            arra :  dictionnaire qui a pour clef les dimensions qui nous intéressent et les personnages que l'on veut mesurer
            k : le seuil avec lequel on décide de savoir si la personne est morte ou vivante
        """
        prediction = []
        if k == None:
            k = math.floor(math.sqrt(len(self.data)))
        else:
            pass

        if trig == None:
            trig = 0.5
        else:
            pass
        
        for a in range(len(arra[self.dim[0]])):
            """
                a prend la valeur de la position d'un personnage dont les caractéristique sont stocké à la position "a" de chaque dimension de arra
            """
            personnage = {b : arra[b][a] for b in arra.keys()}
            K_NN.distance(self,fix = personnage,var = range(len(self.data[self.dim[0]])))


            #On tri les personnages par leur distance puis on prend les K plus proches
            self.dst.sort(key=lambda x: x[1])
            print(self.dst)
            LoI = self.dst[:k]

            #On établit ne nombre de morts dans les K voisins et on récupère la vraisemblance de ces observations
            self.nbmrt = sum([1 for x in LoI if self.data['isAlive'][x[0]] == 0])
            print("zer", LoI[k-1])
            self.lik_m,self.lik_v = poiss.Vraisemblance(LoI[k-1][1],personnage,self.nbmrt,(k-self.nbmrt))

            #On calcul la probabilité d'être mort, si c'est supérieur au seuil on dit que la personne est morte sinon on dit qu'elle est vivante
            post = lik_m*self.mort/(lik_m*self.mort+lik_v*(1-self.mort))
            if post > trig:
                post = 0
            else:
                post = 1

            prediction += [post]

        precision = accuracy_score(arra['isAlive'],prediction)
        return precision





class Poisson():
    """
        Établit la répartition des individus selon les différentes dimension et estime la vraisemblance de l'obtention de K-voisins morts ou vivants

        data : Le dictionnaire sur lequel on travail avec le K-NN
        dim : Les dimensions que l'on utilise pour le K-NN
        coef : Les coefficients qui sont utilisé pour mesurer les distance entre les individus 
    """
    def __init__(self,data,dim,coef):

        self.data = data
        self.dens_m = {x : {} for x in self.data.keys() if x in dim}
        self.dens_v = {x : {} for x in self.data.keys() if x in dim}
        self.span = 1
        self.dim = dim
        self.coef = {x : coef[x] for x in dim}

    def Densité(self,span):
        """
            Fonction pour trouver la densité de répartition des vivants et des morts selon les différentes dimensions
            Les paramètres sont :
                span : la résolution de la fonction de masse
        """

        #On initialise le nombre total de mort de vivants
        self.span = span
        total_m = sum([1 for x in self.data['isAlive'] if x == 0])
        total_v = sum([x for x in self.data['isAlive']])


        #On initialise les dictionnaires avec des séquences d'espace de taille "span"
        for a in self.dim:
            self.dens_m[a] = {b : 0 for b in np.linspace(-1,1,self.span)}
            self.dens_v[a] = {b : 0 for b in np.linspace(-1,1,self.span)}

        #On attribut à chaque tranche la densité de personnes mortes ou vivantes
        for x,y in itr.product(self.dim,range(len(self.data))):
            if self.data['isAlive'][y] == 0:
                self.dens_m[x][Etendue(self.data[x][y],self.span,-1,1)] += 1/total_m
            else:
                self.dens_v[x][Etendue(self.data[x][y],self.span,-1,1)] += 1/total_v

    def Voxel(self,clef,val):
        """
            Retourne la probabilité de trouver quelqu'un de mort à cet endroit
            clef : vecteur contenant les coordonées cartésienne de la cible
        """
        #On multiplie la probabilité d'être à l'endroit x de la dimension d_1 avec l'endroit y de la dimension d_2 etc ...
        vox_dens_m = reduce(mul,[self.dens_m[x][y] for x,y in zip(clef,val)])
        vox_dens_v = reduce(mul,[self.dens_v[x][y] for x,y in zip(clef,val)])
        return vox_dens_m, vox_dens_v

    def Vraisemblance(self,r,O,m,v):
        """
            Retourne la vraisemblance des données
        """
        dim = [x for x in self.data.keys() if x != 'isAlive']

        #On récupère l'étendue sur laquelle s'étend l'hypersphère
        #On fait len(self.data.keys())-1 pour ne pas avoir la valeur 'isAlive' de la cible
        print("O",O)
        lim_sphere = {a : [Etendue(O[a]-r,self.span,-1,1),Etendue(O[a]+r,self.span,-1,1),self.span] for a in dim}


        lambm, lambv, comp = 0,0,0
        for x in np.transpose(self.Comb(lim_sphere,r)):
            print(x[0])
            a, b= Poisson.Voxel(self.dens_m.keys(),x)
            lambm += a
            lambv += b
        
        lambm = lambm * math.exp(-lambm) / math.factorial(m)
        lambv = lambv * math.exp(-lambv) / math.factorial(v)
        return lambm,lambv

    
    def Comb(self,lim,r):
        """
            Retourne l'ensemble des combinaisons de l'espace qui doivent être calculée
            On dit moins de 10 000 pour ne pas avoir un temps de calcul aberrant 
        """
        print(lim)
        wo = [lim[x] for x in lim.keys()]
        print(wo)
        a = math.ceil(r/self.span)
        b = len(lim.keys())
        while pow(a,b) > 10000:
            a -= 1

        wo = [[np.linspace(x[0],x[1])] for x in wo]
        return itr.product(*wo)

def Etendue(val,span,de,a):
    deb = de
    print("deb ",deb)
    print("span",span)
    print("a",a)
    print("val",val)
    for x in np.linspace(de,a,span):
        print(x)

        if 1 == x:
            print('ok')
        if val <= x and val >= deb:
            print("1",x)
            return float(x)
        elif val < de:
            print("2",span-1)
            return float(span-1)
        elif val > a:
            print("3")
            return float(1)
        else:
            print("4")
            deb += 2/span
    return deb



