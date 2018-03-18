import numpy as np

class Creator:
    
    #Creates neural networks based on given specs
    #Network layers to be input
    #enter() to be handled by parent class
    
    
    
    def __init__(self, layer_data, inp):
        self.la = []
        self.wl = []
        self.bl = []
        c = 0
        for v in layer_data:
            try:
                self.la.append(np.asmatrix(np.zeros(v).reshape(v, 1)))
                self.wl.append(self.rmat(layer_data[c + 1], v))
                self.bl.append(self.rmat(layer_data[c + 1], 1))
                c += 1
            except IndexError:
                pass
        del v, c
        self.ip(inp)
        
    def ip(self, inpu):
        c2 = 0
        for data in inpu:
            self.la[0][c2, 0] = data
            c2 += 1
        del c2, data
        
    def rmat(self, x, y):
        lim = x * y
        retv = np.asmatrix(np.random.rand(lim))
        for i in range(lim):
            k = np.random.randint(0, 3)
            if k == 1:
                retv[0, (i - 1)] *= -1
        retv = retv.reshape(x, y)
        return retv
    def sig(self, x):
        x = np.exp(x)/(np.exp(x) + 1)
        return x               
        
    def exunt(self):
            l = [self.bl, self. wl]
            return l
    def enter(self, ln):
            self.bl = ln[0]
            self.wl = ln[1]
        
    def compute(self):
        no = len(self.la)
        for i in range(len(self.la) - 1):
            new = np.dot(self.wl[i], self.la[i])
            new = new  + self.bl[i]
            mfunc = np.vectorize(self.sig, otypes=[np.float])
            new = mfunc(new)
            self.la[i + 1] = new
        return self.la[no - 1]
            
