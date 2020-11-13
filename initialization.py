import numpy as np
import random
import matplotlib.pyplot as plt

def define_structure(a,n): 
    structure = []
    
    for i in range(n):  
        for j in range(n): 
            for k in range(n):
                structure.append(
                np.multiply((i - (n-1)/2),np.array([a,0,0])) + \
                np.multiply((j - (n-1)/2),np.array([a/2,a*np.sqrt(3)/2,0])) + \
                np.multiply((k - (n-1)/2),np.array([a/2,a*np.sqrt(3)/6,a*np.sqrt(2/3)])))
        
    with open('structure.xyz', 'w') as f:
        f.write(str(n**3))
        f.write('\n')
        for i in range(n**3):
            f.write('\n')
            f.write('Ar')
            for j in range(3):
                f.write('\t')
                f.write(str(structure[i][j]))
                
    return np.array(structure)
            
        
def set_sign():
    return random.choice((-1, 1))
            
def define_momentum(T0, m, n):
    M = []
    for i in range(n**3):
        M.append(np.array([set_sign()*np.sqrt(2*m*(-1/2)*(8.31*(10**(-3)))*T0*np.log(random.random())), 
                           set_sign()*np.sqrt(2*m*(-1/2)*(8.31*(10**(-3)))*T0*np.log(random.random())), 
                           set_sign()*np.sqrt(2*m*(-1/2)*(8.31*(10**(-3)))*T0*np.log(random.random()))]))
        
    M_sum = np.array(M).sum(axis=0)
    M = np.array(M) - M_sum/(n**3)
        
    with open('momentum.dat', 'w') as f:
        for i in range(n**3):
            for j in range(3):
                f.write(str(M[i][j]))
                f.write('\t')
            f.write('\n')

            
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Momentum')
    ax1.hist(M[:,0], bins=20, color = 'green', label = 'x')
    ax2.hist(M[:,1], bins=20, color = 'red', label = 'y')
    ax3.hist(M[:,2], bins=20, color = 'blue', label = 'z')
    
    fig.set_figheight(9)
    fig.set_figwidth(16)
    fig.legend()
    fig.savefig('momentum.png')
        
    return M