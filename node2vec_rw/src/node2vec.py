import numpy as np
import networkx as nx
import random
import os
import argparse
import re
from termcolor import colored
#import matplotlib.pyplot as plt
import time

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.G_is_directed = nx_G.is_directed()
                self.is_directed = is_directed
		self.p = p
		self.q = q
                self.walk = []
                
        def __iter__(self):
            return self.G.edges()
        
        
        def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges
		walk = [start_node]
                
                self.walk_length = walk_length

		while len(walk) < walk_length:
			cur =  walk[-1]

                        cur_nbrs = sorted(G.neighbors(cur))

                        
                        if len(cur_nbrs) > 0: #that means that the start node can be empty as well
				if len(walk) == 1: #this is the part which determines the walk
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                                else:
					prev = walk[-2]
                                        
                                        edges = alias_edges[(prev, cur)]
                                            
                                        n_idx =  alias_draw(
                                            
                                            [alias_edges[(prev, cur)][0]],
                                            [alias_edges[(prev, cur)][1]]

                                            )

                                        if n_idx >= len(cur_nbrs):
                                            break
                                        
                                        n = cur_nbrs[n_idx]
                                        """
                                        next = cur_nbrs[
                                                
                                                alias_draw(
                                                    alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					"""
                                        
                                        
                                        
                                        walk.append(n)
			else:
                            break

		self.walk = walk
                return self.walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print 'Walk iteration:'
		for walk_iter in range(num_walks):
			print str(walk_iter+1), '/', str(num_walks)
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

        def alias_edge_walk(self):
            G = self.G    
            
            graph = {}
            
            walk = []
             
            start_node = [self.walk[n/sum(self.walk)] for n in self.walk ]
            argmax  = np.argmax(self.walk)*start_node[-(len(start_node)/2)] 
            argmin = np.argmin((self.walk)*start_node[-(len(start_node)/2)])
            length = self.walk_length
            alias_nodes = self.alias_nodes
            alias_edges = self.alias_edges
            
            
            while len(walk) < length: 
                    
                if len(walk) == 0:
                    walk.append(self.walk[argmin])

                if len(walk) >= 1:


                    alias_edge1 = self.get_alias_edge(start_node[0], argmax)
                    alias_edge2 = self.get_alias_edge(self.walk[0], argmax) 
                
                    probs = alias_draw(alias_edge1, alias_edge2)
                    walk.append(self.walk[probs])
                    
                
            self.a_walk = walk
            
            return walk

    
        
        
        
        def walking_walk(self):
            walk = self.a_walk
            start_node = [walk[np.argmax(walk)]]
            the_walks = [start_node]
            
            
            while len(the_walks) <  self.walk_length:  
                n = [n for n in G.neighbors(start_node[-1])]
                drawn = alias_draw(alias_setup(n)[1], alias_setup(n)[0])
                the_walks.append(drawn)

            return the_walks
        
    
    

        def get_alias_edge(self, src, dst):
            '''
            Get the alias edge setup lists for a given edge.
            '''
            G = self.G
            p = self.p
            q = self.q

            unnormalized_probs = []
            
            return p,q
            
            for dst_nbr in sorted(G.neighbors(dst)):
                    if dst_nbr == src:
                            unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
                    elif G.has_edge(dst_nbr, src):
                            unnormalized_probs.append(G[dst][dst_nbr]['weight'])
                    else:
                            unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
            norm_const = sum(unnormalized_probs)
            

            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

            return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			
                        #return [float(u)/norm_const for u in unnormalized_probs]


                        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
                                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1]) #ME : this relation curating is the way
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

    

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
        K = len(J)
	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]: #need to understand the algorithm better
	    return kk
	else:
	    return J[kk]

def edgelist_graph(path, tp=nx.DiGraph()):
    if not os.path.exists(path):
        raise ValueError("file must exist :(") 
    if '.' not in os.path.splitext(path)[-1]:
        raise ValueError("A directory inserted") 
    if path.endswith('.edgelist'): 
        G = nx.read_edgelist(path, nodetype=int, create_using=tp)
    
    else:
        if path.endswith('.gexf'):
            G = nx.read_gexf(path, node_type=int)
            G = nx.read_pajek(path)
        # pro-processing graph for node2veghborsc 
        nx.write_edgelist(G)
    
    #from every node in edge combination, recursively add weight 1 in edge1->edge2 
    for edge in G.edges():
         G[edge[0]][edge[1]]['weight'] = 1 
    
    return G


def mod_graph(index):
    return
    
def read_log(fname):
    
    d={}
    
    if os.path.exists(fname):
        for line in open(fname, 'r').readlines():
            for i in range(0, len(line.split(' ')), 2):
                d[i] = d[i+1]
        return d

def mean(array):
    return np.mean(array)


def run_walks(G, num_times=1):
    
    j, q =  alias_setup(G.nodes())
    
    draw = False
    j, q = np.array(j).reshape(len(j)), np.array(q).reshape(len(q))
    
    #just for fun

    colors = ['red', 'green', 'blue']
    
    outs = []
    #while True: 
    while len(outs) < num_times:
        
        T= time.time()
        a=alias_draw(j, q)
        for k in range(3):
            b=alias_draw(j,q)
       

        nG = Graph(G, False, a,b)
        
        test = nG.preprocess_transition_probs()
        
        a_edge= nG.get_alias_edge(a,b)
        nG.node2vec_walk(10**3, 2)
        walk=  nG.alias_edge_walk()
       
        
        walk = nG.walking_walk()
        
        
        outs.append(walk)
    outs = np.array(outs)        
    return outs
if __name__ == '__main__':
    #working with creating a graph and random walking within it
    
    parsed = argparse.ArgumentParser()
    parsed.add_argument('graph', type=str, default=None)
    parsed.add_argument('iter_count', type=int, default=None)
    args = parsed.parse_args()
    
    
    pre_graph = {k:k*2 for k in range(10)} 
    
    G =nx.complete_graph(pre_graph)
    
    if args.graph:
        G = edgelist_graph(args.graph, nx.Graph())
    arr = []
    while len(arr) < args.iter_count:        
        walk =run_walks(G, args.iter_count) 
        print "shape returned %s" % str(walk.shape)
        print "end node %d" % walk.tolist()[-1][-1]
        arr.append(walk.tolist()[-1][-1])
        
                
                
                
    print mean(arr)
    other_G = {}
    for x,y in zip(G.nodes(), walk):
        other_G[x] = y
    print other_G[list((G.nodes()))[1]] 
