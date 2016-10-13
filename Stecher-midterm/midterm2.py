import math

N = 1000
n = 0
it = 0

while((N - n) > 1):
  it += 1
  n = N
  N = (8/(0.05**2))*math.log((4*((2*N)**10 + 1)/0.05))
  
  if(it == 20):
    break

print N
