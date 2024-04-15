"""
denclue.py

@author: mgarrett
changed by Euan
main change:  Gaussian to Spherical Kernel Function
kernelize(x, y, h, degree) = (c_d / vol_b_d) * (np.linalg.norm(x - y) <= h)
"""
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx
from scipy.special import gamma

def _hill_climb(x_t, X, W=None, h=0.1, eps=1e-7):
    """
    This function climbs the 'hill' of the kernel density function
    and finds the 'peak', which represents the density attractor
    """
    error = 99.
    prob = 0.
    x_l1 = np.copy(x_t)
    
    #Sum of the last three steps is used to establish radius
    #of neighborhood around attractor. Authors suggested two
    #steps works well, but I found three is more robust to
    #noisey datasets.
    radius_new = 0.
    radius_old = 0.
    radius_twiceold = 0.
    iters = 0.
    while True:
        radius_thriceold = radius_twiceold
        radius_twiceold = radius_old
        radius_old = radius_new
        x_l0 = np.copy(x_l1)
        x_l1, density = _step(x_l0, X, W=W, h=h)
        error = density - prob
        prob = density
        radius_new = np.linalg.norm(x_l1-x_l0)
        radius = radius_thriceold + radius_twiceold + radius_old + radius_new
        iters += 1
        if iters>3 and error < eps:
            break
    return [x_l1, prob, radius]

def _step(x_l0, X, W=None, h=0.1):
    n = X.shape[0]
    d = X.shape[1]
    superweight = 0.
    x_l1 = np.zeros((1, d))
    if W is None:
        W = np.ones((n, 1))
    else:
        W = W

    for j in range(n):
        kernel = kernelize(x_l0, X[j], h, d)
        if kernel > 1e-8:
            superweight += kernel
            x_l1 += (kernel * X[j])

    if superweight > 1e-8:
        x_l1 /= superweight
    else:
        x_l1 = x_l0

    density = superweight / np.sum(W)
    return [x_l1, density]
    
def kernelize(x, y, h, degree):
    c_d = np.pi ** (degree / 2) / gamma(degree / 2 + 1)
    vol_b_d = h ** degree
    kernel = (c_d / vol_b_d) * (np.linalg.norm(x - y) <= h)
    return kernel

class DENCLUE(BaseEstimator, ClusterMixin):
    """Perform DENCLUE clustering from vector array.

    Parameters
    ----------
    h : float, optional
        The smoothing parameter for the gaussian kernel. This is a hyper-
        parameter, and the optimal value depends on data. Default is the
        np.std(X)/5.

    eps : float, optional
        Convergence threshold parameter for density attractors

    min_density : float, optional
        The minimum kernel density required for a cluster attractor to be
        considered a cluster and not noise.  Cluster info will stil be kept
        but the label for the corresponding instances will be -1 for noise.
        Since what consitutes a high enough kernel density depends on the
        nature of the data, it's often best to fit the model first and 
        explore the results before deciding on the min_density, which can be
        set later with the 'set_minimum_density' method.
        Default is 0.

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. In this version, I've only tested 'euclidean' at this
        moment.

    Attributes
    -------
    cluster_info_ : dictionary [n_clusters]
        Contains relevant information of all clusters (i.e. density attractors)
        Information is retained even if the attractor is lower than the
        minimum density required to be labelled a cluster.

    labels_ : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    Notes
    -----


    References
    ----------
    Hinneburg A., Gabriel HH. "DENCLUE 2.0: Fast Clustering Based on Kernel 
    Density Estimation". In: R. Berthold M., Shawe-Taylor J., Lavrač N. (eds)
    Advances in Intelligent Data Analysis VII. IDA 2007
    """
    
    def __init__(self, h=None, eps=1e-8, min_density=0., metric='euclidean'):        
        self.h = h        
        self.eps = eps
        self.min_density = min_density
        self.metric = metric
        
    def fit(self, X, y=None, sample_weight=None):
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        density_attractors = np.zeros((self.n_samples, self.n_features))
        radii = np.zeros((self.n_samples, 1))
        density = np.zeros((self.n_samples, 1))

        # create default values
        if self.h is None:
            self.h = np.std(X) / 5
        if sample_weight is None:
            sample_weight = np.ones((self.n_samples, 1))
        else:
            sample_weight = sample_weight

        # initialize all labels to noise
        labels = -np.ones(X.shape[0])

        # climb each hill
        print("Fitting DENCLUE model...")
        with tqdm(total=self.n_samples, unit="sample") as pbar:
            for i in range(self.n_samples):
                try:
                    density_attractors[i], density[i], radii[i] = _hill_climb(X[i], X, W=sample_weight,
                                                         h=self.h, eps=self.eps)
                except ZeroDivisionError:
                    # Handle potential divide-by-zero errors
                    density_attractors[i] = X[i]
                    density[i] = 0.0
                    radii[i] = 0.0
                pbar.update(1)

        # initialize cluster graph to finalize clusters. Networkx graph is
        # used to verify clusters, which are connected components of the
        # graph. Edges are defined as density attractors being in the same
        # neighborhood as defined by our radii for each attractor.
        cluster_info = {}
        num_clusters = 0
        cluster_info[num_clusters] = {'instances': [0],
                                    'centroid': np.atleast_2d(density_attractors[0])}
        g_clusters = nx.Graph()
        for j1 in range(self.n_samples):
            g_clusters.add_node(j1, attractor=density_attractors[j1], radius=radii[j1], density=density[j1])
    
        # populate cluster graph
        print("Computing cluster centroids and sample similarities...")
        with tqdm(total=self.n_samples * (self.n_samples - 1) // 2, unit="edge") as pbar:
            for j1 in range(self.n_samples):
                for j2 in (x for x in range(self.n_samples) if x != j1):
                    if j2 in g_clusters.neighbors(j1):
                        continue
                    diff = np.linalg.norm(g_clusters.nodes[j1]['attractor'] - g_clusters.nodes[j2]['attractor'])
                    if diff <= (g_clusters.nodes[j1]['radius'] + g_clusters.nodes[j1]['radius']):
                        g_clusters.add_edge(j1, j2)
                    pbar.update(1)
    
       # connected components represent a cluster
        clusters = [g for g in nx.connected_components(g_clusters)]
        num_clusters = 0
    
        # loop through all connected components
        print("Finalizing clusters...")
        with tqdm(total=len(clusters), unit="cluster") as pbar:
            for clust in clusters:
                # get maximum density of attractors and location
                max_instance = max(clust, key=lambda x: g_clusters.nodes[x]['density'])
                max_density = g_clusters.nodes[max_instance]['density']
                max_centroid = g_clusters.nodes[max_instance]['attractor']
    
                # populate cluster_info dict
                cluster_info[num_clusters] = {'instances': list(clust),
                                            'size': len(clust),
                                            'centroid': max_centroid,
                                            'density': max_density,
                                            'complete': g_clusters.subgraph(clust).number_of_edges() == (len(clust) * (len(clust) - 1)) / 2.}
    
                # if the cluster density is not higher than the minimum,
                # instances are kept classified as noise
                if max_density >= self.min_density:
                    labels[list(clust)] = num_clusters
                num_clusters += 1
                pbar.update(1)
    
        self.clust_info_ = cluster_info
        self.labels_ = labels
        return self
        
    def get_density(self, x, X, y=None, sample_weight=None):
        superweight=0.
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if sample_weight is None:
            sample_weight = np.ones((n_samples,1))
        else:
            sample_weight = sample_weight
        for y in range(n_samples):
            kernel = kernelize(x, X[y], h=self.h, degree=n_features)
            kernel = kernel * sample_weight[y]/(self.h**n_features)
            superweight = superweight + kernel
        density = superweight/np.sum(sample_weight)
        return density
        
    def set_minimum_density(self, min_density):
        self.min_density = min_density
        labels_copy = np.copy(self.labels_)
        for k in self.clust_info_.keys():
            if self.clust_info_[k]['density']<min_density:
                labels_copy[self.clust_info_[k]['instances']]= -1
            else:
                labels_copy[self.clust_info_[k]['instances']]= k
        self.labels_ = labels_copy
        return self
