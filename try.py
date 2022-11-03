import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from LeafNode import *
from sklearn.cluster import KMeans
import numpy as np

# Lire le fichier, ajouter les titres de colomns
df = pd.read_csv('data.csv', header=None ,delimiter='\t')
df.columns = ['data1', 'data2', 'class']

# EXO 2-1: calculer nombre de inliers et outliers
nb_outlier = df.loc[df['class'] == 1]
nb_inlier = df.loc[df['class'] == 0]
print("Question 2-1:---------------------------------------------")
print("Nombre de outliers: " + str(len(nb_outlier)) )
print("Nombre de inlier: " + str(len(nb_inlier)) )

# EXO 2-2: seaborn preview
sb.scatterplot(data = df, x = 'data1', y = 'data2', hue = 'class')
#plt.show()

# EXO 4.1

# Calculer ecart type
def ecart_t(x1,x2):
    ecart_x1 = np.std(x1)
    ecart_x2 = np.std(x2)
    if (ecart_x1 >= ecart_x2):
        return x1
    else:
        return x2

data_choisi = ecart_t(df['data1'], df['data2'])

print("\nQuestion 4.1:---------------------------------------------")
print("L'ecart type choisi est: " + str(np.std(data_choisi)))

new_array1 = data_choisi.to_numpy()
array_choisi = new_array1.reshape(len(new_array1),1)

k_min = new_array1.min()
k_max = new_array1.max()

kmeans = KMeans(init=(np.array( [ [k_min],[k_max] ] )), n_clusters=2, n_init=1).fit(array_choisi)
print("\ncluster_centers: " + str(kmeans.cluster_centers_))

########################################## EXO 4.2
def get_currentAttrib(df):
    ecart_x1 = np.std(df['data1'])
    ecart_x2 = np.std(df['data2'])
    if (ecart_x1 >= ecart_x2):
        return 1
    else:
        return 2

def get_array(df, num):
    if(num==1):
        data_choisi = df['data1']
    else:
        data_choisi = df['data2']
    tmp = data_choisi.to_numpy()
    array = tmp.reshape(len(tmp),1)
    return array

def get_data(num):
    if(num==1):
        data = 'data1'
    else:
        data = 'data2'
    return data

def get_kmean(array):
    k_min = array.min()
    k_max = array.max()
    kmeans = KMeans(init=(np.array( [ [k_min],[k_max] ] )), n_clusters=2, n_init=1).fit(array)
    return kmeans

get_kmean(array_choisi)

class Tree:
    def __init__(self, D, central, attribIdx):
        self.tree=self.buildDecisionTree(D, central, attribIdx)

    def buildDecisionTree(self, D, central, attribIdx):
        if (len(D)>=4) :
            if (len(attribIdx) >= 2) :
                currentAttrib = get_currentAttrib(df)
                array = get_array(D, currentAttrib)
                kmeans = get_kmean(array)

                a = float(kmeans.cluster_centers_[0])
                b = float(kmeans.cluster_centers_[1])
                data = get_data(currentAttrib)

                Dl= D[D[data]< a]
                Dm= D[(D[data] > a) & (D[data] < b)]
                Dr=D[D[data]>b]
                attribIdx.remove(currentAttrib)
                L = self.buildDecisionTree(Dl, False, attribIdx)
                M = self.buildDecisionTree(Dm, True, attribIdx)
                M.outlier= True
                R = self.buildDecisionTree(Dr, False, attribIdx)
                return Node(attribIdx,a,b,L,M,R)
            else:
                return Leaf(D, attribIdx)
        elif (len(D) < 4):
            if (central == True):
                return Node(outlier = True)
            else:
                return Node(outlier = False)


Tree = Tree(df, True,[1,2])

inlier = Tree.tree.node.inlier.donne
TN = len(inlier[inlier['class'] == 0])
FN = len(inlier[inlier['class'] == 1])

out_left = Tree.tree.node.out_left.donne
out_right = Tree.tree.node.out_right.donne
FP = len(out_left[out_left['class'] == 0]) + len(out_right[out_right['class'] == 0])
TP = len(out_left[out_left['class'] == 1]) + len(out_right[out_right['class'] == 1])

print("\nPour l'arbre de question 4.2:---------------------------------------------")
print("\nTN: " + str(TN) + "  FP: " + str(FP))
print("\nFN: " + str(FN) + "  TP: " + str(TP))

Exa_pond = (TN / (TN + FP) + TP / (FN + TP)) / 2
precision = TP/(TP + FP)
rapple = TP/(FN + TP)

print("L'exactitude pondere de cette model est: " + str(Exa_pond))
print("La précision de cette model est: " + str(precision))
print("Le rappel de cette model est: " + str(rapple))
#######################################""EXO4.3
class Tree2:
    def __init__(self, D, central, hauteur):
        self.tree=self.buildDecisionTree2(D, central, hauteur)
    #
    def buildDecisionTree2(self, D, central, hauteur):
        if (len(D)>=4) :
            if (hauteur >= 2) :
                currentAttrib = get_currentAttrib(df)
                array = get_array(D, currentAttrib)
                kmeans = get_kmean(array)

                a = float(kmeans.cluster_centers_[0])
                b = float(kmeans.cluster_centers_[1])
                data = get_data(currentAttrib)

                Dl= D[D[data]< a]
                Dm= D[(D[data] > a) & (D[data] < b)]
                Dr=D[D[data]>b]
                hauteur = hauteur - 1
                L = self.buildDecisionTree2(Dl, False, hauteur)
                M = self.buildDecisionTree2(Dm, True, hauteur)
                M.outlier= True
                R = self.buildDecisionTree2(Dr, False, hauteur)
                return Node(hauteur,a,b,L,M,R)
            else:
                return Leaf(D, hauteur)
        elif (len(D) < 4):
            if (central == True):
                return Node(outlier = True)
            else:
                return Node(outlier = False)



Tree = Tree2(df, True, 2)

inlier = Tree.tree.node.inlier.donne
TN = len(inlier[inlier['class'] == 0])
FN = len(inlier[inlier['class'] == 1])

out_left = Tree.tree.node.out_left.donne
out_right = Tree.tree.node.out_right.donne
FP = len(out_left[out_left['class'] == 0]) + len(out_right[out_right['class'] == 0])
TP = len(out_left[out_left['class'] == 1]) + len(out_right[out_right['class'] == 1])

print("\nPour l'arbre de question 4.3:---------------------------------------------")
print("\nTN: " + str(TN) + "  FP: " + str(FP))
print("\nFN: " + str(FN) + "  TP: " + str(TP))

Exa_pond = (TN / (TN + FP) + TP / (FN + TP)) / 2
precision = TP/(TP + FP)
rapple = TP/(FN + TP)
tab=[Exa_pond, precision, rapple]

print("L'exactitude pondere de cette model est: " + str(tab[0]))
print("La précision de cette model est: " + str(tab[1]))
print("Le rappel de cette model est: " + str(tab[2]))
