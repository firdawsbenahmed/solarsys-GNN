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

### true acc of every body at every saved moment 
acc = np.zeros((n_traj, T , N , 3))
###empty numpy array having same shape as the pos and it hols one acc arrow x, y, z for every body at every moment in ever traj 
for k in range(n_traj):
    for t in range(T):
        #### in the both loops we r going through every traj in every saved moment 
        acc_now , _ = acceleration(pos[k , t] ,  mass[k] , eps) ## to throw the forces away 
        acc[k , t] = acc_now ### It stores the acc in the big table each moment gets the correct answer for its own positions
    if (k + 1 ) % 50 == 0 or k+1 == n_traj : 
        print(f"{k+1} / {n_traj} trajectories are done") ### a prog msg is sent after every 50 traj or at the end


####   acc should explain how the vel changed between snapshots
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

#### how big are the star vs planet
star_size = []
planet_size = []
for k in range(n_traj) : 
    for t in range(T): 
        star_size.append(np.linalg.norm(acc[k , t, 0])) #### the star goes 1st means the size lngth of the star’s acc arrow
        for b in range(1 , N) : 
            planet_size.append(np.linalg.norm(acc[k , t , b])) ### then of all the planets’ arrows go in the 2nd
print(f"typical |acceleration|:  star {np.median(star_size):.1e}   planets {np.median(planet_size):.1e}") #### the typed value of the star is 200 times smaller this is so important later for the training setup


### save everything from the input file + the labels
out = {}

for name in d.files : #### d.files is the list of names stored in the input file it copies each of them in new dictionary
    out[name] = d[name]
out["acc"] = acc ### it adds the new labels under the name acc
np.save_compressed(a.out , **out) ### save dictionary as compressed file + ** is here to unpacks it into separete arrays 
print(f"saved {a.out}: acc {acc.shape}")