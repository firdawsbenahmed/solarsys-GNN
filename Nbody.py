
import numpy as np 

G = 1 

def force(pos , mass , eps) : 
    N = len(mass) ### the number of the bodies 

    F = np.zeros((N , N , 2 ))## this an empty table n lines and n colm + each row has a cell of (x,y)

    for i in range(N) : ## i is the one that is being pulled 
       for j in range(N) : ## j is the one pulled to 
           if i == j : ## the body shouldnt pull itself
            continue

           d = pos[j] - pos[i] ## (- xj from xi and same for y )

           r2 = d[0]**2 + d[1]**2 + eps**2 ## we added the eps to avoid 0 like safety stuff

           r = np.sqrt(r2)  

           F[i , j] = G * mass[i] * mass [j] / r2 * (d/r) # is the arrow from body i to body j and i added this for the importance of the pulling direction 
    return F

def acceleration(pos , mass , eps): 
   N = len(mass)
   F = force(pos , mass , eps)
   acc = np.zeros(N , 2) ## NumPy array with just n rows and 2 colums and it has one (x, y) row per body 

   for i in range(N) : 
       total_force = np.zeros(2) ## here it is a numpy array that has only 2 numbers to store the total force on i object xi and yi cz we have force on x and on y 
       for j in range(N): 
          total_force += F[i , j]
       acc[i] = total_force / mass[i]

   return acc , F