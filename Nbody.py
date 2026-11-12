
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

def make_sys(rng , n_planet , star_mass = 1.0 , r_range =(0.5 , 3.0), mass_range=(1e-4 , 1e-2), noise = 0.02 , incl_max = 0.3 ): 
   
   n = n_planet+1 ## +1 is for the sun
   mass = np.zeros(n) ## empty numpy array that has n numbers
   pos = np.zeros((n , 3)) ## n lines so each planet object has its row and the 2 is for the (x,y)
   vel = np.zeros((n , 3)) ## same 

   mass[0] = star_mass ## it is putting the origin at the sun
   log_of_mass = rng.uniform(np.log(mass_range[0]), np.log(mass_range[1] ), n_planet)
   ## Picking directly would make most planets heavy, while picking in log space gives light and heavy planets in equal proportion
   r = rng.uniform(*r_range , n_planet) ### one random distance from the star per planet between 0.5 and 3.0
   theta = rng.uniform(0, 2 * np.pi, n_planet) ## one random angle per planet anywhere around a full circle 0 to 2π
   wobble = rng.standard_normal(n_planet) ### one small random number per planet, used to make the orbit speed slightly off
   tilt = rng.uniform( - incl_max , incl_max , n_planet)  ### tilt of each orbit out of the x-y plane this was a major changing for the 3D 
   for k in range(n_planet): 
      body = k + 1 ## k= is the start which is the star so planet k is body k+1 
      mass[body] = np.exp(log_of_mass[k])

      ### 1st sstep on the nrml plane x y 
      x , y = r[k] * np.cos(theta[k]), r[k]*np.sin(theta[k]) ## we use the random distance and the angle 
      dx , dy = -np.sin(theta[k]) , np.cos(theta[k])  ## the direction of the movement ofcs it will be sideways to the line torward the star 
      speed = np.sqrt(G * star_mass / r[k] * (1 + noise * wobble[k])) ### the 1st part of this equa is a low the 2nd part that has the noise i added it so it changes the 1st part by 2% so the orbit don't be like a perfect circle cz this isn't reality 
