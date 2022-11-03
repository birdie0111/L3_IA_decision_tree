class Leaf:
    def __init__(self, D, attribIdx=1):
        self.outlier= False
        self.donne= D
        self.indice = attribIdx
        self.a = -1
        self.b = -1                
        self.out_left = None
        self.inlier = None
        self.out_right = None

class Node:
    def __init__(self,currentAttrib=None,a=None,b=None,L=None,M=None,R=None, outlier=None):
        self.node= Leaf(None,currentAttrib)
        ##
        self.outlier = outlier
        self.node.indice = currentAttrib
        self.node.a = a
        self.node.b = b        
        self.node.out_left = L
        self.node.inlier = M
        self.node.out_right = R
