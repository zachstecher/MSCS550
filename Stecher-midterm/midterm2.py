import math

N = 1000
n = 0
it = 0

#While the difference is more than 1, we haven't yet converged.
while((N - n) > 1):
  it += 1
  n = N
  N = (8/(0.05**2))*math.log((4*((2*N)**10 + 1)/0.05))
  
  #Just in case something goes wrong, to prevent an infinite loop.
  if(it == 20):
    break

print N
