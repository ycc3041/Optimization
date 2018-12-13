import numpy as np
from numpy.linalg import inv
from numpy.linalg import matrix_rank


def check_redundant(A, b, c):
    """
    Stojković and Stanimirović Method
    # Input
    A: m * n matrix; coefficients of constraints
    b: m * 1 vector; right-hand-side values
    c: n * 1 vector; coefficients of objective function 
    # Return
    (True, i): True if there is redundant constraint, along with the index of the redundant constraint
    (False, None): False if there is no redundant constraint 
    """    
    
    m = A.shape[0]
    n = A.shape[1]
    Ab = np.append(A, b[:,None], axis=1)
    
    if matrix_rank(Ab) != m: # matrix Ab doesn't have full rank -> exist redundant constraint
        d = np.ones((m, n))
        
        # compute each element in matrix d
        for i in range(m):
            for j in range(n):
                if b[i] == 0 or c[j] == 0:
                    d[i,j] = float('inf')
                else:
                    d[i,j] = A[i, j] / (b[i] * c[j])
        
        # find a row i_1 in d that has smaller value than another row i_2 for each element 
        for i_1 in range(m):
            for i_2 in range(m):
                if i_1 != i_2: # don't need to compare with itself
                    target_row = d[i_1, :]
                    comparing_row = d[i_2, :]
                    if not False in np.less_equal(target_row, comparing_row):
                        return (True, i_1)
                    else: # should not happen
                        return (True, None)
    else:
        return (False, None)


def simplex_method(A, b, c, B_index, N_index):
    """
    # Input
    A: m * n matrix; coefficients of constraints
    b: m * 1 vector; right-hand-side values
    c: n * 1 vector; coefficients of objective function
    B_index: list of m elements; initial basic variable indexes
    N_index: list of (n-m) elements; initial non-basic variable indexes
    
    # Return
    B_index: list of m elements; final indexes of optimal variables
    x_B: m * 1 vector; the optimal basic variables values
    objFunValue: scalar; optimal objective function value
    final_table: m * n matrix; final table in tableau
    
    """
    
    all_index = set(B_index + N_index)
    
    c_B = np.array([c[i] for i in B_index]) # initial c_B
    B = A[:, B_index] # initial B

    print('### Checking before simplex method')

    m = A.shape[0]
    n = A.shape[1]
    
    # check initial B is identity matrix or not
    if np.all(B != np.identity(m)):
        print('The initial B is not an identity matrix. Stop the simplex.')
        return None
    else:
        print('The initial B is an identity matrix.')
    
    B_inv = inv(B)
    N = A[:, N_index]
        
    # check initial basic solution is feasible or not
    x_B = B_inv @ b
    if np.any(True in x_B < 0):
        print('The initial basic solution is not feasible.')
        return None
    else:
        print('The initial basic solution is feasible.')
    
    objFunValue = np.transpose(c_B) @ B_inv @ b # initial objective function value


    # everything is ready -> start the simplex method
    print('\n ### Start the simplex method')    
    stop = False
    iteration = 0
    
    while stop == False:
        
        c_B = np.array([c[i] for i in B_index]) 
        c_N = np.array([c[i] for i in N_index]) 
    
        B = A[:, B_index]
        B_inv = inv(B)
        N = A[:, N_index]
        
        x_B = B_inv @ b
        objFunValue = np.transpose(c_B) @ B_inv @ b
        
        print('\n # Iteration {}'.format(iteration))
        print('Basic variables and values')
        for i in range(len(B_index)):
            print('* variable {}: {}'.format(B_index[i], x_B[i]))
        print('=> objective function value: {}'.format(objFunValue))        
            
        # calcuate reduced costs for all nonbasic variables       
        reduced_costs = np.transpose(c_B) @ B_inv @ N  - c_N

        # check whether exist redueced cost that > 0
        # if some is > 0 -> keep going
        # if all are <= 0 -> stop; optimal found
        boolean = reduced_costs > 0
        if True in boolean:
            print('...exists reduced cost that > 0')
            
            # decide which variable will enter the basis
            # Bland's rule to prevent cycling
            # if contain same reduced cost, automatically choose minimum index one
            k = N_index[np.argmax(reduced_costs)]
        
            # check if unbounded
            x_B = B_inv @ b
            y = B_inv @ A[:, k]
            
            boolean = y > 0
            if True in boolean:
                # decide which variable will leave basis
                # ratio test with Bland's rule to prevent cycling
                # if contain same ratio, automatically choose minimum index one
                r = None
                min_ratio = float('inf')
                for i in range(len(y)):
                    if boolean[i] == True:
                        ratio = x_B[i] / y[i]
                        if ratio < min_ratio:
                            r = i
                            min_ratio = ratio
    
                print('...variable {} will enter the basis'.format(k))
                print('...variable {} will leave the basis'.format(B_index[r]))
     
                # update B and N indexes
                B_index[r] = k
                N_index = list(all_index - set(B_index))
                
                iteration += 1
                
            else:
                print('This problem is unbounded! Stop the simplex method.')
                stop = True
                return None
                
        else:
            stop = True
            print('All reduced cost are <= 0 -> optimal found!')
            return None
        
    final_table = np.ones((m, n))
    final_table[:, B_index] = np.identity(m)
    final_table[:, N_index] = B_inv @ A[:, N_index]
    
    return (B_index, x_B, objFunValue, final_table)
   

# =======================================================
# linear programming model 13
# https://sites.math.washington.edu/~burke/crs/407/models/m13.html

# check redundant constraints in canonical form
A = np.array([[17, 14, 10, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 15, 16, 12, 11, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 13, 12, 14, 8, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 8, 6],
              [-17, 0, 0, 0, -15, 0, 0, 0, -13, 0, 0, 0, -10, 0, 0, 0],
              [0, -14, 0, 0, 0, -16, 0, 0, 0, -12, 0, 0, 0, -11, 0, 0],
              [0, 0, -10, 0, 0, 0, -12, 0, 0, 0, -14, 0, 0, 0, -8, 0],
              [0, 0, 0, -9, 0, 0, 0, -11, 0, 0, 0, -8, 0, 0, 0, -6]])
b = np.array([1500, 1700, 900, 600, -22.5, -9, -4.8, -3.5])
c = np.array([16, 12, 20, 18, 14, 13, 24, 20, 17, 10, 28, 20, 12, 11, 18, 17])
check_redundant(A, b, c)


# phase (I)
# check feasibility
# if objective function value == 0 -> LP problem is feasible
A = np.array([[17, 14, 10, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 15, 16, 12, 11,0, 0, 0, 0, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 13, 12, 14, 8, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 8, 6, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [17, 0, 0, 0, 15, 0, 0, 0, 13, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0],
              [0, 14, 0, 0, 0, 16, 0, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0],
              [0, 0, 10, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0 ,0, 1, 0],
              [0, 0, 0, 9, 0, 0, 0, 11, 0, 0,0, 8, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, -1, 0 ,0 ,0, 1]])
b = np.array([1500, 1700, 900, 600, 22.5, 9, 4.8, 3.5])
B_index = [16, 17, 18, 19, 24, 25, 26, 27] 
N_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23]
c_1 = np.array([0]*24 + [1]*4)
B_index, x_B, objFunValue, final_table = simplex_method(A, b, c_1, B_index, N_index)


if objFunValue == 0:
    print('Objective function value is equal to 0 -> The original problem is feasible.')
else:
    print('Objective function value is not equal to 0 -> The original problem is not feasible.')


# phase (II)
A = final_table[:, 0:24]
b = x_B
c_2 = np.array([-16, -12, -20, -18, -14, -13, -24, -20, -17, -10, -28, -20, -12, -11, -18, -17] + [0]*8)
N_index = list(set(range(24)) - set(B_index))
B_index_2, x_B_2, objFunValue_2, final_table_2 = simplex_method(A, b, c_2, B_index, N_index)
