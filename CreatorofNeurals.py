import copy
import numpy as np

class Creator:
    """
    Creates neural networks based on given specs
    Network layers to be input
    enter() to be handled by parent class
    """
    
    
    def __init__(self, layer_data, inp):
        #creates random data that forms the weights and biases
        self.la = []#layer data
        self.wl = []#weight data
        self.bl = []#biases
        c = 0
        try:
            for v in layer_data:
                self.la.append(np.asmatrix(np.zeros(v).reshape(v, 1)))
                self.wl.append(self.rmat(layer_data[c + 1], v))
                self.bl.append(self.rmat(layer_data[c + 1], 1))
                c += 1
        except:
            pass
        del c
        self.ip(inp)
        self.mfunc = np.vectorize(self.sig)        
        
    def ip(self, inpu):
        #handles input for the network
        c2 = 0
        for data in inpu:
            self.la[0][c2, 0] = data
            c2 += 1
        del c2
    def cost(self, right):
        #Gives the neural network a score, 0 is the best score
        ll = self.la[len(self.la) - 1]
        Cost = 0
        for i in np.nditer(ll):
            if i != ll[right]:
                Cost += i**2
            else:
                Cost += (1 - i)**2
        return Cost
    
                
    def backprop(self, set_list):#setlist also has meta-lists with data and correct values
        #Backpropogation, finds the value for which the cost function is minimum (Actually a local minimum)
        state  = copy.deepcopy(self.exunt())
        for i in state:#list[wl, bl]
            for a in i:#list[number of layers]
                for b in np.nditer(a, op_flags=['readwrite']):#iterates over each value but at the same time allows for editing
                    b[...] += 0.01
                    di = 0.02
                    sc = self.compproc(set_list, state)
                    b[...] -= 0.02
                    sc2 = self.compproc(set_list, state)
                    sl = (sc2 -sc)/di
                    b[...] += (sl * 0.01 * 1.718281828) + 0.01
        self.enter(state)
                    
    def rmat(self, x, y):
        #returns a matrix with random values of specified shape
        lim = x * y
        retv = np.asmatrix(np.random.rand(lim))
        for i in range(lim):
            k = np.random.randint(0, 3)
            if k == 1:
                retv[0, i] *= -1
        retv = retv.reshape(x, y)
        return retv
    def sig(self, x):#sigmoid func, vectorized for application to arrays in the __init__
        y = float(1/(2.7182818285**(-x) + 1))
        return y               
        
    def exunt(self):#Outputs current biases and weights
        l = [self.bl, self. wl]
        return l
    def enter(self, ln):#Inputs a list containing new values for biases and weights
        self.bl = ln[0]
        self.wl = ln[1]        
    def ncomp(self, wl, bl):#computes the final layer and gives it out in the form of percentages 
        no = len(self.la)
        for i in range(no - 1):
            new = np.dot(wl[i], self.la[i])
            new = new  + bl[i]
            jok = self.mfunc(new)
            self.la[i + 1] = jok
        res = []
        for i in self.la[no - 1]:
            res.append(i)
        s = sum(res)
        ares = []
        for k in res:
            ares.append(100*k/s)
        return np.array(ares).reshape(len(ares), 1)
    def compute(self):#Wrapper function
        return self.ncomp(self.wl, self.bl)
    def compproc(self, set_list, state):#Processes cost of batches of data
        assert len(set_list[1]) ==len(set_list[0])
        s = 0
        for io in range(len(set_list[0])):
            self.ip(set_list[0][io])
            self.ncomp(state[1], state[0])
            s += self.cost(set_list[1][io])
        return s            
