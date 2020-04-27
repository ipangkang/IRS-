from cvxpy import *
import cvxpy as cvx

'''
problem :
minimize (x-y)^2
subject to
        x+y=1
        x-y>=1
'''
from cvxpy import *
# Create two scalar optimization variables.
x = Variable()
y = Variable()
# Create two constraints.
constraints = [x + y == 1,
              x - y >= 1]
obj = Minimize(square(x - y))
prob = Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)
prob2 = cvx.Problem(cvx.Maximize(x + y), prob.constraints)
print("optimal p1 value", prob2.solve())

constraints = [x + y <= 3] + prob.constraints[1:]
prob2 = cvx.Problem(prob.objective, constraints)
print("optimal P2 value", prob2.solve())
