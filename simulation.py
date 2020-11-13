import initialization
import numpy as np
import numba
import time
from decimal import Decimal

@numba.njit
def calc(R, epsilon, structure, f_par, n, L):   
    V = 0
    F_S = np.zeros((n**3, 3))
    F_P = np.zeros((n**3, n**3, 3))
    
    for i in np.arange(0, n**3):
        r_b = r_p = np.linalg.norm(structure[i])
        if r_b >= L:
            V += (f_par/2)*np.power((r_b - L), 2)
            F_S[i] = f_par*(L - r_b)*structure[i]/r_b
        if i > 0:
            for j in np.arange(0, i):
                r_p = np.linalg.norm(structure[i]-structure[j])
                V += epsilon*((R/r_p)**12 - 2 * (R/r_p)**6)
                F_P[i, j] = 12 * epsilon * ((R/r_p)**12 - (R/r_p)**6) * ((structure[i] - structure[j])/(r_p**2))
                F_P[j, i] = - F_P[i, j]
    
    F = np.sum(F_P, axis=1) + F_S
    P = np.sum(np.sqrt(np.sum(F_S**2, axis=0))) / (4 * np.pi * L**2)
    
    return (V, P, F)

def sim(n, structure, momentum, m, f_par, S0, Sd, R, epsilon, S_xyz, S_out, L, tau):
    Temp_cul = P_cul = H_cul = 0
    T_avr = P_avr =  H_avr = 0
    
    r = structure
    p = momentum
    
    with open('wyniki/topnienie/86K/sim.atom', 'w+') as f, open('wyniki/topnienie/86K/sim.dat', 'w+') as f2:
        f2.write('t' + '\t' + 'H' + '\t' + 'H_avr' + '\t' + 'V' + '\t' + 'E_kin' + '\t' + 'T' + '\t' + 'T_avr' + '\t' + 'P' + '\t' + 'P_avr' + '\n')
            
        calc_products = calc(R, epsilon, structure, f_par, n, L)
        for i in range(S0+Sd):
            p = p + 0.5 * calc_products[2] * tau
            r = r + (1/m) * p * tau
            calc_products_curr = calc(R, epsilon, r, f_par, n, L)
            p = p + 0.5 * calc_products_curr[2] * tau

            E_kin = np.sum(np.linalg.norm(p)**2, axis=0)/(2*m)
            H = E_kin + calc_products_curr[0]
            Temp = E_kin*2/(3*(n**3)*8.31*(10**(-3)))

            calc_products = calc_products_curr

            if i > S0:
                Temp_cul = Temp_cul + Temp
                P_cul = P_cul + calc_products_curr[1]
                H_cul = H_cul + H
                
                T_avr = Temp_cul/(i-S0)
                P_avr = P_cul/(i-S0)
                H_avr = H_cul/(i-S0)
            
            if i%S_xyz == 0:
                """
                f.write('ITEM: TIMESTEP \n')
                f.write(str(i) + '\n')
                f.write('ITEM: NUMBER OF ATOMS \n')
                f.write(str(n**3) + '\n')
                f.write('ITEM: BOX BOUNDS pp pp pp \n')
                f.write(str(-L) + '\t' + str(L) + '\n')
                f.write(str(-L) + '\t' + str(L) + '\n')
                f.write(str(-L) + '\t' + str(L) + '\n')
                f.write('ITEM: ATOMS id type xs ys zs' + '\n')
                
                for k in range(n**3):
                    f.write(str(k+1) + ' ')
                    f.write('1' + ' ')
                    for j in range(3):
                        f.write(str(r[k][j]))
                        f.write(' ')
                    f.write('\n')  
                """
                for k in range(n**3):
                    #f.write(str(k+1) + ' ')
                    #f.write('1' + ' ')
                    for j in range(3):
                        f.write(str(r[k][j]))
                        f.write(' ')
                    f.write('\n')    

            if i%S_out == 0:    
                f2.write("{:.3f}".format(i*tau) + '\t')
                f2.write("{:.6E}".format(Decimal(H)) + '\t')
                f2.write("{:.6E}".format(Decimal(H_avr)) + '\t')
                f2.write("{:.6E}".format(Decimal(calc_products[0])) + '\t')
                f2.write("{:.6E}".format(Decimal(E_kin)) + '\t')
                f2.write("{:.6E}".format(Decimal(Temp)) + '\t')
                f2.write("{:.6E}".format(Decimal(T_avr)) + '\t')
                f2.write("{:.6E}".format(Decimal(calc_products[1])) + '\t')
                f2.write("{:.6E}".format(Decimal(P_avr)) + '\t')
                f2.write('\n')

                
if __name__ == "__main__":
    start_time = time.time()
    
    file = open('parameters.input')
    params = {}
    for line in file:
        line = line.strip()
        key_value = line.split('#')
        if len(key_value) == 2:
            params[key_value[1].strip()] = np.float(key_value[0].strip())
    
    
    structure = initialization.define_structure(params['a'], int(params['n']))
    momentum = initialization.define_momentum(params['Temp0'], params['m'], int(params['n']))
    sim(int(params['n']) , structure, momentum, params['m'], params['f_par'], int(params['S0']), int(params['Sd']), params['R'], 
        params['e'], int(params['S_xyz']), int(params['S_out']), params['L'], params['tau'])
    
    print("Simulation took", round(time.time() - start_time,2) , "s to run")