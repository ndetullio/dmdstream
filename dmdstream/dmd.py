import numpy as np
import numpy.linalg as la

class StreamingDMD:
    
    """
    Calculate DMD in streaming mode. 
    Python class based on: 
    "Dynamic Mode Decomposition for Large and Streaming Datasets",
    Physics of Fluids 26, 111701 (2014). 
    """

    def __init__(self, max_rank=0):

        self.max_rank = max_rank
        self.NGRAM = 5 # Number of Gram_Schmidt iterations
        self.EPSILON = np.finfo(np.float32).eps
        self.Qx = 0
        self.Qy = 0
        self.A = 0
        self.Gx = 0
        self.Gy = 0

    def preprocess(self, x, y):
        
        # Construct bases
        normx = la.norm(x)
        normy = la.norm(y)
        self.Qx = x/normx
        self.Qy = y/normy

        # Compute
        self.Gx = np.zeros([1,1]) + normx**2
        self.Gy = np.zeros([1,1]) + normy**2
        self.A = np.zeros([1,1]) + normx * normy


    def update(self, x, y):

        normx = la.norm(x)
        normy = la.norm(y)     
            
#"       ------- STEP 1 --------       "
        xtilde = np.zeros(shape=(self.Qx.shape[1],1))
        ytilde = np.zeros(shape=(self.Qy.shape[1],1))
        # xtilde = 0
        # ytilde = 0
        ex = x
        ey = y
        for _ in range(self.NGRAM):
            dx = np.transpose(self.Qx).dot(ex)
            dy = np.transpose(self.Qy).dot(ey)
            xtilde = xtilde + dx
            ytilde = ytilde + dy
            ex = ex - self.Qx.dot(dx)
            ey = ey - self.Qy.dot(dy)
                
#"""       ------- STEP 2 --------       """
#        Check basis for x and expand if needed
        if la.norm(ex) / normx > self.EPSILON:
#           Update basis for x
            self.Qx = np.hstack([self.Qx,ex/la.norm(ex)])
#           Increase size of Gx and A by zero-padding
            self.Gx = np.hstack([self.Gx,np.zeros([self.Gx.shape[0],1])])
            self.Gx = np.vstack([self.Gx,np.zeros([1,self.Gx.shape[1]])])
            self.A  = np.hstack([self.A,np.zeros([self.A.shape[0],1])])
            
        if la.norm(ey) /normy > self.EPSILON:
#           Update basis for y
            self.Qy = np.hstack([self.Qy,ey/la.norm(ey)])
#           Increase size of Gy and A by zero-padding
            self.Gy = np.hstack([self.Gy,np.zeros([self.Gy.shape[0],1])])
            self.Gy = np.vstack([self.Gy,np.zeros([1,self.Gy.shape[1]])])
            self.A  = np.vstack([self.A,np.zeros([1,self.A.shape[1]])]) 
            
#"""       ------- STEP 3 --------       """
#       Check if POD compression is needed
        r0 = self.max_rank
        if r0:
            if self.Qx.shape[1] > r0:
                eigval, eigvec = la.eig(self.Gx)
                indx = np.argsort(-eigval) # get indices for sorting in descending order
                eigval = -np.sort(-eigval) # sort in descending order
                qx = eigvec[:,indx[:r0]]
                self.Qx = self.Qx.dot(qx)
                self.A = self.A.dot(qx)
                self.Gx = np.diag(eigval[:r0])
                
            if self.Qy.shape[0] > r0:
                eigval, eigvec = la.eig(self.Gy)
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
                
                
    def compute_modes(self):
        """ 
        Compute DMD modes and eigenvalues
        M, lam = StreamingDMD.compute_modes produces a vector lam of DMD 
        eigenvalues, and a matrix M whose columns are the DMD modes of the
        data set 
        """
        
        Ktilde = np.transpose(self.Qx).dot(self.Qy).dot(self.A).dot(la.pinv(self.Gx))
        eigvals,eigvecK = la.eig(Ktilde)
        modes = self.Qx.dot(eigvecK)
        
        return modes, eigvals
            
