# Authors:
# - Adham Ashraf 
# - Seif Mahdy
# - Yousef Moustafa
# Created for educational purposes under the supervision of Dr. Ahmed Abdelsamea and Eng. Hossam Fathy :)

import numpy as np

def LPP_Form(A_ub, b_ub, A_eq, b_eq):
    n = 0 # number of rows
    m = 0 # number of columns
    
    # Convert A_ub and b_ub to numpy arrays and count the rows and columns for them (including number of slack, surplus and artificial variables)
    if A_ub != None:
        A_ub = np.array(A_ub, dtype=float)
        b_ub = np.array(b_ub, dtype=float)
        n += A_ub.shape[0]
        m += A_ub.shape[1] + A_ub.shape[0] * 2
    
    # Convert A_eq and b_eq to numpy arrays and count the rows and columns for them (including number of artificial variables)
    if A_eq != None:
        A_eq = np.array(A_eq, dtype=float)
        b_eq = np.array(b_eq, dtype=float)
        n += A_eq.shape[0]
        m += A_eq.shape[0]
    
    # Intialize A and b
    A = np.zeros((n, m), dtype=float)
    b = np.zeros(n, dtype=float)
    
    # Put A_ub and b_ub into A and b with their slack, surplus and artificial variables (Checking that b >= 0)
    if type(A_ub) == np.ndarray:
        A[:A_ub.shape[0], :A_ub.shape[1]] = A_ub
        A[:A_ub.shape[0], A_ub.shape[1] : A_ub.shape[1] + A_ub.shape[0]] = np.eye(A_ub.shape[0])
        b[:A_ub.shape[0]] = b_ub
        A[np.argwhere(b < 0).reshape(-1)] *= -1
        b[np.argwhere(b < 0).reshape(-1)] *= -1
        A[:A_ub.shape[0], -A_ub.shape[0] - (A_eq.shape[0] if type(A_eq) == np.ndarray else 0) : (-A_eq.shape[0] if type(A_eq) == np.ndarray else None)] = np.eye(A_ub.shape[0])
        
    # Put A_eq and b_eq into A and b with their artificial variables (Checking that b >= 0)    
    if type(A_eq) == np.ndarray:
        A[-A_eq.shape[0]:, :A_eq.shape[1]] = A_eq
        b[-A_eq.shape[0]:] = b_eq
        A[np.argwhere(b < 0).reshape(-1)] *= -1
        b[np.argwhere(b < 0).reshape(-1)] *= -1
        A[-A_eq.shape[0]:, -A_eq.shape[0]:] = np.eye(A_eq.shape[0])
        
    return A, b
     
    
class RevisedSimplexOptimize:
    def __init__(self, cost, A_ub = None, b_ub = None, A_eq = None, b_eq = None):
        if any([A_ub and not b_ub, not A_ub and b_ub, A_eq and not b_eq, not A_eq and b_eq, not A_eq and not A_ub]):
            raise AssertionError("You must include equality or inequality constraint, any of them with both (A) matrix and (B) vector!")
        self.A, self.b = LPP_Form(A_ub, b_ub, A_eq, b_eq)
        self.b_v = np.arange(self.A.shape[1] - self.A.shape[0], self.A.shape[1])
        self.B_inv = np.eye(self.b_v.shape[0])
        self.cost = np.zeros(self.A.shape[1], dtype=float)
        self.cost[:len(cost)] = cost
        self.x0 = self.b[:] # x0: Basic variables values
        self.vars = len(cost) # Vars: number of original variables
        self.status = 0 # Status: solution status 0 if in progress 1 if converged 2 if unbounded 3 if infeasible 4 if degenerate
        
    def solve(self):
        # Finding basic feasible solution
        self._phase_one()
        
        # Remove artificial variables columns from A matrix
        self.A = self.A[:, :-self.A.shape[0]]
        
        # Check for infeasibility
        if np.any(self.b_v >= self.A.shape[1]) and self.status == 0:
            self.status = 3
            
        # Finding the optimal solution    
        self._phase_two()
        
        # Change the status to convergence
        if self.status == 0: 
            self.status = 1 
            
        return self.__result_dict() # Result dictionary generation according to status
        
    
    def _phase_one(self):
        # Build temporary cost function from summation of artificial variables and minimize it using phase 2
        cost = np.zeros(self.A.shape[1], dtype=float)
        cost[-self.b_v.shape[0]:] = np.ones(self.b_v.shape[0])
        self._phase_two(cost)
    
    def _phase_two(self, cost = None):
        # Termination if status not in progress
        if self.status != 0:
            return
        # If the cost function is not given then it will use problem cost function else it will minimize cost function came from argument
        if type(cost) != np.ndarray:
            cost = self.cost
            
        # Store our non basic variables and calculate the reduced cost for each one      
        d = np.setdiff1d(np.arange(self.A.shape[1]), self.b_v)
        r_d = cost[d] - np.matmul(np.matmul(cost[self.b_v], self.B_inv), self.A[:, d])
    
        # This loop will work until the reduced cost for each non basic variable >= 0
        while r_d.min() < 0:
            # Determine which will be the new basic variable and calculate its x_k = B^-1 @ A_k
            new_basic_var =d[np.argmin(r_d)]
            x1 = np.matmul(self.B_inv, self.A[:, new_basic_var])
            
            # Calculate ratio between x0 and x1, determine which less positive ratio to be our pivot and check for unbounded solution 
            ratio = self.x0 / x1
            ratio[np.argwhere(ratio < 0).reshape(-1)] = np.inf
            if ratio.min() == np.inf:
                self.status = 2
                return            
            pivot = np.argmin(ratio)
            
            # Pivot row update for B^-1 and x0
            self.B_inv[pivot] /= x1[pivot]
            self.x0[pivot] /= x1[pivot]
            x1[pivot] = 1
            
            # Remaining rows update for B^-1 and x0
            for i in range(x1.shape[0]):
                if i == pivot: continue
                self.B_inv[i] += self.B_inv[pivot] * -x1[i]
                self.x0[i] += self.x0[pivot] * -x1[i]     
                x1[i] = 0
            
            # Update basic variables and non basic variables sets and calculate reduced cost for non basic variables
            self.b_v[pivot] = new_basic_var
            d = np.setdiff1d(np.arange(self.A.shape[1]), self.b_v)
            r_d = cost[d] - np.matmul(np.matmul(cost[self.b_v], self.B_inv), self.A[:, d])
            
            # Check for degenerate solution
            if 0 in self.x0:
                self.status = 4
                return
            
    def __result_dict(self):
        # Special cases output based on status
        if self.status == 2:
            return dict(sol = "Unbounded")
        if self.status == 3:
            return dict(sol = "Infeasible")
        if self.status == 4:
            return dict(sol = "Degeneracy")
        
        # Construct result dictionary of original variable from basic variables and their values
        results = dict()
        for x, val in zip(self.b_v, self.x0):
            if x < self.vars:
                results[f"x{x + 1}"] = val

        # Adding original variable from non basic variables and their values and add function value after calculating it using cTx
        x = np.zeros(self.vars, dtype=float)
        for i in range(self.vars):
            if f"x{i + 1}" not in results:
                results[f"x{i + 1}"] = 0.
            x[i] = results[f"x{i + 1}"]
        results["fun"] = np.dot(self.cost[:self.vars], x)
        
        return results
            
            
            
        