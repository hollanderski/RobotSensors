
import numpy as np

# Calcul la distance du point pM au segment [p1, p2]
def point2seg(pM, p1, p2):
    vec = p2 - p1
    vec /= np.sqrt(vec[0]**2+vec[1]**2)
    vecT = np.array([-vec[1], vec[0]])
    return np.abs(np.sum((pM-p1)*vecT))


class Cluster:
    def __init__(self, points):
        self.points = np.array(points).T

        # Centre du cluster
        self.center = np.array([
            np.mean(self.points[0]),
            np.mean(self.points[1])
        ])

        # Largeur et longueur du cluster
        self.width = np.max(self.points[0])-np.min(self.points[0]) # along the x coordinate
        self.length = np.max(self.points[1])- np.min(self.points[1]) # along the y coordinate

        # Cluster polyline
        self.eps = 0.1 # 0.05
        self.poly = np.array(self.rdp(self.points, self.eps))


# Ramer-Douglas-Peucker algorithm :

    def rdp(self, points, eps):

        dmax=0
        jmax=0
        distances=[]

        for i in range(1, int(points.shape[1])): #-1
            if (points[:,0] - points[:,int(points.shape[1])-1]).sum()!=0 :
                distances.append(point2seg(points[:,i], points[:,0], points[:,int(points.shape[1])-1]))

        if distances:
            dmax=np.max(distances)
            jmax=np.argmax(distances)
        else :
            dmax=-1

        if dmax > eps and jmax!=0: 
            Pres1= self.rdp(points[:,:jmax], eps)
            Pres2= self.rdp(points[:,jmax:int(points.shape[1])], eps)
            Pres=Pres1[:len(Pres1)-1] + Pres2[:len(Pres2)]
            
        else:
            Pres=[points[:,0], points[:,int(points.shape[1])-1]]

        return Pres


def clustering(points):
    k = 3  
    D = 0.1
    # Clustering algorithm 
    G = np.zeros(len(points[0]), dtype=int)
    Gp = []
    g=0
    dmin=100000
    jmin=0
    
    for i in range(k, int(points.shape[1])):
        distances=[]
        for j in range(1, k+1):
            d=np.sqrt(np.square(points[0][i-j]-points[0][i])+np.square(points[1][i-j]-points[1][i]))
            distances.append(d)

        dmin=np.min(distances)
        jmin=np.argmin(distances)+1

     
        if dmin < D :
            if G[i-jmin] == 0:
                g+=1
                Gp.append([])
                G[i-jmin]=g
                Gp[g-1].append(points[:,i-jmin])
            
            G[i]=G[i-jmin]
            Gp[G[i-jmin]-1].append(points[:,i])
            



    clusters = [Cluster(np.array(cluster_p)) for cluster_p in Gp]

    return G, clusters
