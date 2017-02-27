
class StreamingDMD:
    
    def __init__(self,max_rank=0):
        self.count = 0
        self.max_rank = max_rank
        self.Qx = 0
        self.Qy = 0
        self.A = 0
        self.Gx = 0
        self.Gy = 0
    description = """Calculate DMD in streaming mode. 
                     Python class based on M.S. Hemati, 
                     M.O. Williams, C.W. Rowley, "Dynamic Mode Decomposition 
                     for Large and Streaming Datasets",
                     Physics of Fluids 26, 111701 (2014). 
                     This is a python translation of the Matlab class 
                     provided in the paper's supplementary material."""
    author = "Nico De Tullio"
    
    def update(self, x, y):
        import numpy as np
        import numpy.linalg as LA
        import scipy.linalg.blas as LAB
        from time import time
        
#       Parameters
        ngram = 5 #number of times to reapply Gram-Schmidt
        epsilon = np.finfo(np.float32).eps

        self.count += 1

        normx = LA.norm(x)
        normy = LA.norm(y)     
        
#"""       ------- Pre-processing step --------       """
        if self.count == 1:
#           Construct bases
            self.Qx = x/LA.norm(x)
            self.Qy = y/LA.norm(y)
            
#           Compute
            self.Gx = np.zeros([1,1]) + normx**2
            self.Gy = np.zeros([1,1]) + normy**2
            self.A = np.zeros([1,1]) + normx * normy
            
#"       ------- STEP 1 --------       "
        xtilde = np.zeros(shape=(self.Qx.shape[1],1))
        ytilde = np.zeros(shape=(self.Qy.shape[1],1))
        ex = x
        ey = y
        for igram in range(ngram):
            dx = np.transpose(self.Qx).dot(ex)
            dy = np.transpose(self.Qy).dot(ey)
            xtilde = xtilde + dx
            ytilde = ytilde + dy
            ex = ex - self.Qx.dot(dx)
            ey = ey - self.Qy.dot(dy)
                
#"""       ------- STEP 2 --------       """
#        Check basis for x and expand if needed
        if LA.norm(ex) / normx > epsilon:
#           Update basis for x
            self.Qx = np.hstack([self.Qx,ex/LA.norm(ex)])
#           Increase size of Gx and A by zero-padding
            self.Gx = np.hstack([self.Gx,np.zeros([self.Gx.shape[0],1])])
            self.Gx = np.vstack([self.Gx,np.zeros([1,self.Gx.shape[1]])])
            self.A  = np.hstack([self.A,np.zeros([self.A.shape[0],1])])
            
        if LA.norm(ey) /normy > epsilon:
#           Update basis for y
            self.Qy = np.hstack([self.Qy,ey/LA.norm(ey)])
#           Increase size of Gy and A by zero-padding
            self.Gy = np.hstack([self.Gy,np.zeros([self.Gy.shape[0],1])])
            self.Gy = np.vstack([self.Gy,np.zeros([1,self.Gy.shape[1]])])
            self.A  = np.vstack([self.A,np.zeros([1,self.A.shape[1]])]) 
            
#"""       ------- STEP 3 --------       """
#       Check if POD compression is needed
        r0 = self.max_rank
        if r0:
            if self.Qx.shape[1] > r0:
                eigval, eigvec = LA.eig(self.Gx)
                indx = np.argsort(-eigval) # get indices for sorting in descending order
                eigval = -np.sort(-eigval) # sort in descending order
                qx = eigvec[:,indx[:r0]]
                self.Qx = self.Qx.dot(qx)
                self.A = self.A.dot(qx)
                self.Gx = np.diag(eigval[:r0])
                
            if self.Qy.shape[1] > r0:
                eigval, eigvec = LA.eig(self.Gy)
                indx = np.argsort(-eigval) # get indices for sorting in descending order
                eigval = -np.sort(-eigval) # sort in descending order
                qy = eigvec[:,indx[:r0]]
                self.Qy = self.Qy.dot(qy)
                self.A = np.transpose(qy).dot(self.A)
                self.Gy = np.diag(eigval[:r0])
                
#"""       ------- STEP 4 --------       """
        xtilde = np.transpose(self.Qx).dot(x)
        ytilde = np.transpose(self.Qy).dot(y)
        
#       Update A, Gx and Gy 
        self.A  = self.A + ytilde.dot(np.transpose(xtilde))
        self.Gx = self.Gx + xtilde.dot(np.transpose(xtilde))
        self.Gy = self.Gy + ytilde.dot(np.transpose(ytilde))
        
        return self
        
    def compute_modes(self):
        """ Compute DMD modes and eigenvalues
        M,lam = StreamingDMD.compute_modes produces a vector lam of DMD 
        eigenvalues, and a matrix M whose columns are the DMD modes of the
        data set """
        
        import numpy as np
        import numpy.linalg as LA
        
        Ktilde = np.transpose(self.Qx).dot(self.Qy).dot(self.A).dot(LA.pinv(self.Gx))
        eigvals,eigvecK = LA.eig(Ktilde)
        modes = self.Qx.dot(eigvecK)
        
        return modes, eigvals
            
