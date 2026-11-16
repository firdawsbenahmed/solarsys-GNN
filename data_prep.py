import numpy as np 
import argparse

from Nbody import acceleration

ap = argparse.ArgumentParser()
ap.add_argument("--inp" , default="nbody_train.npz") ### dataset to read 
ap.add_argument("--out" , default="nbody_train_acc.npz") ### dataset to write 

a = ap.parse_args() ### the variable that will hold wht u type 

d = np.load(a.inp) 
### here it opens a .npz file and it behaves like a doctionary d["pos"]gives us an array tht stored under the name of pos 
pos , vel , mass = d["pos"] , d["vel"] , d["mass"]
eps , dt_save = float(d["eps"]) , float(["dt"]) ### dt is the time between the snapshots 

n_traj , T , N , _ = pos.shape ### so here pos has the shape of (n_traj , T , N , 3)
### these 4 sizes are : trjectories and moments in time and nb of bodies and lastly the 3 coordonates , no need for _ but it is used as a name for the 3 cz python reqires a name for every value so 

acc = np.zeros((n_traj, T , N , 3))
###empty numpy array having same shape as the pos and it hols one acc arrow x, y, z for every body at every moment in ever traj 
for k in range(n_traj):
    for t in range(T):
        #### in the both loops we r going through every traj in every saved moment 
        acc_now , _ = acceleration(pos[k , t] ,  mass[k] , eps) ## to throw the forces away 
        acc[k , t] = acc_now ### It stores the acc in the big table each moment gets the correct answer for its own positions
    if (k + 1 ) % 50 == 0 or k+1 == n_traj : 
        print(f"{k+1} / {n_traj} trajectories are done") ### a prog msg is sent after every 50 traj or at the end

errors = [] ### empty list to collect the errors 

for k in range(n_traj) : 
    for t in range(T-1) : ## every pair of neighbouring moments so the loop stops one early at t-1
        for b in range(1 , N) : ### every N body starting from 1 not 0 so only the planets are checked 
            dv = vel[ k , t + 1 , b] - vel[k , t , b] ### we r using the neighboring pairs here we r counting the vel changes in between the t and t + 1
            acc_avg = 0.5 * (acc[k , t , b] + acc[k , t+1 , b])
            error = np.linalg.norm(dv / dt_save - acc_avg)
            ##### dv / dt_saved is vel change per unit time sub the avg acc gives how far off the two are  and np.linalg.norm turns diff arrow into a length
            errors.append(error / np.linalg.norm(acc_avg)) ### it divides by the size of the acceleration itself to get a relative error this is convention
print(f"label check, median relative error (planets only): {np.median(errors):.1e}")
### printing the middle value of all those errors 
#### i used the median instead of the mean bcz a few rare close encounters could throw an avg off
