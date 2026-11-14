
import argparse
import numpy as np 

### i like commenting my work every line so the any usr later find it easy to understnd and modify <3
G = 1 

def force(pos , mass , eps) : 
    N = len(mass) ### the number of the bodies 

    F = np.zeros((N , N , 3 ))## this an empty table n lines and n colm + each row has a cell of (x,y)

    for i in range(N) : ## i is the one that is being pulled 
       for j in range(N) : ## j is the one pulled to 
           if i == j : ## the body shouldnt pull itself
            continue

           d = pos[j] - pos[i] ## (- xj from xi and same for y )

           r2 = d[0]**2 + d[1]**2 +  d[2]**2 + eps**2 ## we added the eps to avoid 0 like safety stuff

           r = np.sqrt(r2)  

           F[i , j] = G * mass[i] * mass [j] / r2 * (d/r) # is the arrow from body i to body j and i added this for the importance of the pulling direction 
    return F

def acceleration(pos , mass , eps): 
   N = len(mass)
   F = force(pos , mass , eps)
   acc = np.zeros((N,3)) ## NumPy array with just n rows and 2 colums and it has one (x, y) row per body 

   for i in range(N) : 
       total_force = np.zeros(3) ## here it is a numpy array that has only 2 numbers to store the total force on i object xi and yi cz we have force on x and on y 
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
   vel = vel_half + 0.5 * dt * acc 

   return pos , vel , F 

#####  important note in rl the total energy of the system = kinetic energy + potential energy is an constant value which means it never changees in the simulator we will be using the result 
### the result of theis func as a health check if the number drifts the integrator is inventing or destroying energy
def energy (pos , vel , mass ,eps ) : 
   N = len(mass)

   kinetic = 0.0 ### the kinetic is an energy that and object has due to its motion 
   for i in range(N) :
      speed_square = vel[i][0]**2 + vel[i][1]**2 + vel[i][2]**2  ## total speed of a vector 
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

      ### the tilt HELLA IMPORTANT  so now we be tilting the plane work around the x axis logically here a part of the y will turn into z 
      c , s = np.cos(tilt[k]), np.sin(tilt[k])
      pos[body] = [x , y*c , y*s] ## we are rotating around the x x wont change 
      vel[body] = speed * np.array([dx, dy * c , dy * s])

   total_momentum = np.zeros(3) ## for the whole system 
   for b in range(n) : 
      total_momentum += mass[b] * vel[b]
   drift = total_momentum / mass.sum() ### the get at which speed the sys is drifting 
   for b in range(n) : 
      vel[b] -= drift ### hna the substruction was so the whole system stands still 
        ### more details look since the sun started at 0,0 and the whole planets are movind so the sys has some momument and this sys will be drifting to one direction 
        ### to prevent from this drift we substract the avg drift vel from every body INCLUDING THE STAR now the star gets small vel in the ops direction to the plnts's combined momentum this is why the star wibbles lil bit btw this is true irl 
   return pos , vel , mass 

def simulate(pos , vel , mass , n_steps , dt=0.01 ,eps = 0.05 , save_every=1): 
   
   ### we need lists to collect snapshotss for the vel and pos and the force sizes
   P , V ,FM = [pos.copy()] , [ vel.copy()] , [np.linalg.norm(force(pos , mass, eps),  axis=-1 )] #### turns each arrow into its length

   for s in range(1 , n_steps+1) : 
      pos , vel , F  = leapfrog(pos , vel , mass , dt , eps) ### F is the force table at the new positions
      if s % save_every == 0 : 
         P.append(pos.copy()); V.append(vel.copy()) ; FM.append(np.linalg.norm(F, axis=-1))
   return np.array(P) , np.array(V) , np.array(FM)

def generate_database(n_traj , n_planets , n_steps , dt , eps , save_every , seed , **sys_kw) : 
   #### n_traj is nb of trajectories and the steps per trajectory , step length and a seed the others are clear 
   ### **sys_kw is for any optional extras 

   rng = np.random.default_rng(seed)
   #### this makes a rndm nb generator and once the seed fixes the whole ds so the same seed always gives the same data

   Ps , Vs , Ms , FMs = [] , [] , [] , []

   ### empty list to collect the results of each trajectory pos , vels . mass , force size

   for _ in range(n_traj) : 

      pos , vel , mass = make_sys(rng , n_planets , **sys_kw)

      #### this creates a new random starting system star + planets and random properties like evertime it gives a diff sys but the whole seqnc is kinda repeatable from the seed 

      P , V , FM = simulate(pos , vel ,mass , n_steps , dt , eps , save_every)

      #### this runs that system forward in time and gets back its snapshots

      Ps.append(P) ; Vs.append(V) ; Ms.append(mass) ; FMs.append(FM)
        ## and this adds this trajectory’s results to the four lists
   return dict(pos = np.array(Ps) , vel = np.array(Vs) , mass = np.array(Ms) , force_mag = np.array(FMs) , dt = dt * save_every , eps = eps) ### This stacks each list into one array and returns everything in a dictionary



if __name__ == "__main__" : 
   ap = argparse.ArgumentParser() ####this creates the option reader our menu 
   ap.add_argument("--check" , action="store_true" , help="run energy-conversation check and exist") 
   ### This option is a switch not a value. If you type --check it becomes True and if you dont it stays False
   ### python Nbody.py --check runs the energy test the switch is on
   ####python Nbody.py generates the dataset the switch is off
   ap.add_argument("--n_traj" , type=int , default=500)### each of these lines adds one option to the menu and these nb or info are needed in the functioning later 
   ap.add_argument("--n_planets" , type=int , default=4)
   ap.add_argument("--n_steps" , type=int , default=2000)
   ap.add_argument("--dt" , type=float , default=0.005)
   ap.add_argument("--eps" , type=float , default=0.05)
   ap.add_argument("--save_every" , type=int , default=5)
   ap.add_argument("--seed", type=int, default=0)
   ap.add_argument("--out",  default="nbody_train.npz")
   a = ap.parse_args() ### reads the given data and store it in a

   if a.check: 
      rng = np.random.default_rng(a.seed)
      pos , vel , mass = make_sys(rng , a.n_planets)
      e0 = energy(pos , vel , mass , a.eps) ### it stores the total energy at the very start
      P , V , _ = simulate(pos , vel , mass , a.n_steps , a.dt , a.eps )
      e1 = energy(P[-1] , V[-1] , mass , a.eps) ## -1 means the last time and e1 is the energy at the end
      print(f"E0={e0:.6f}  E_end={e1:.6f}  rel drift={abs((e1 - e0) / e0):.2e}")
   else : 
      d = generate_database(a.n_traj , a.n_planets , a.n_steps , a.dt , a.eps , a.save_every , a.seed)
      np.savez_compressed(a.out , **d) ### it saves the dictionary to a file called a.out the **d unpacks the dictionary so each entry (pos, vel, mass, force_mag, ...) is stored as its own named array in the file
      print(f"saved {a.out}: pos {d['pos'].shape}, force_mag {d['force_mag'].shape}")

    



