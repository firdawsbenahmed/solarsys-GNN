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

