
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

def leapfrog(pos , vel, mass , dt , eps): ## the pur of this func move the whole system forward by one time step which will update the vel and the F 
   ### dt is the length of one full time step 
   acc , _ = acceleration(pos , mass, eps) 
   vel_half = vel + 0.5 * dt * acc ## the rule is vel_half = old vel + acc * time ,,, in this case the time = 0.5* dt 
   pos = pos + dt * vel_half
   acc , F = acceleration(pos , mass , eps)
   vel = vel_half + 0.5 * vel * acc 

   return pos , vel , F 

#####  important note in rl the total energy of the system = kinetic energy + potential energy is an constant value which means it never changees in the simulator we will be using the result 
### the result of theis func as a health check if the number drifts the integrator is inventing or destroying energy
def energy (pos , vel , mass ,eps ) : 
   N = len(mass)

   kinetic = 0.0 ### the kinetic is an energy that and object has due to its motion 
   for i in range(N) :
      speed_square = vel[i][0]**2 + vel[i][0]**2  ## total speed of a vector 
      kinetic += 0.5 * mass[i] * speed_square
   potential = 0.0 
   for i in range(N) : ## chosing the 1st body of the pair to start with
      for j in range(i+1 , N): ## the 2nd body is the bodies that came after i ofcrs this will ensure this and also avoid the case where i = j ofcs 
          arrow = pos[j] - pos[i] 
          distance = np.sqrt(arrow[0]**2+ arrow[1]**2 + arrow[2]**2 + eps**2)
          potential -= G * mass[i] * mass[j] / distance 
   return kinetic + potential

