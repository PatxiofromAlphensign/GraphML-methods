import numpy as np
def rand_matrix(lenn):
    assert lenn%2 == 0
    rands = []
    while len(rands) < lenn:
        rands.append(np.random.randint(1,10))
    matrix = np.array(rands).reshape(int(lenn/2),2)
    return matrix

class Base_ADJ:
    def __init__(self, matrix, lenght=100):
        self.matrix = matrix 
        self.lenght = lenght
        self.forward = []
    def __getitem__(self, key):
        forward = np.array(self.forward[key])
        return forward[key]
        
    def update(self,ml):
        matrix = self.matrix
        adj = []
        forward = []
        for i in np.arange(matrix.shape[0]):
            if i*ml >= matrix.shape[0]:
                break
            forward.append(matrix[int(i*ml)])
            
        self.forward = forward
        return forward
            
            
    def create_adj(self):
        matrix = self.matrix
        adj = []
        
        for f in self.forward:
            x,y = f
            for j in range(matrix.shape[1]):
                adj.append(1 if matrix[x,j] <  matrix[y, j] else 0)
        self.adj = adj
        return adj

    def intervalues(self,ml):
        matrix = self.matrix
        adj = self.create_adj() 
        intervals = matrix.shape[0]/len(adj)
        interval_matrix  = []
        for i in range(int(intervals)):
            inter = matrix[i:len(adj)]
            for j, val in enumerate(adj):
                if val:
                    interval_matrix.append(inter[j])
            
        return np.array(interval_matrix)           



class Trainer(Base_ADJ):
    def __init__(self, m):
    
        self.matrix= rand_matrix(2**m)
        super(Trainer, self).__init__(self.matrix)
    """
    FIXME : add more features to the base class and compute them in train here 
    TODO : implement with tensorflow
    """    
    def train(self, count):
        for i in range(count):
            update_val = self.update(i)

            print(self.intervalues(10))


ini = Trainer(20)
print(ini.train(2))
