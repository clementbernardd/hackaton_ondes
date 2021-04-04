"""
Script python pour ouvrir les fichiers de traces de clavier

"""

import matplotlib.pyplot as plt
import numpy as np
import time

def read_int(f):
    ba = bytearray(4)
    f.readinto(ba)
    prm = np.frombuffer(ba, dtype=np.int32)
    return prm[0]
    
def read_double(f):
    ba = bytearray(8)
    f.readinto(ba)
    prm = np.frombuffer(ba, dtype=np.double)
    return prm[0]

def read_double_tab(f, n):
    ba = bytearray(8*n)
    nr = f.readinto(ba)
    if nr != len(ba):
        return []
    else:
        prm = np.frombuffer(ba, dtype=np.double)
        return prm
    
def get_pics_from_file(filename):
    # Lecture du fichier d'infos + pics detectes (post-processing KeyFinder)
    # print("Ouverture du fichier de pics "+filename)
    f_pic = open(filename, "rb")
    info = dict()
    info["nb_pics"] = read_int(f_pic)
    # print("Nb pics par trame: " + str(info["nb_pics"]))
    info["freq_sampling_khz"] = read_double(f_pic)
    # print("Frequence d'echantillonnage: " + str(info["freq_sampling_khz"]) + " kHz")
    info["freq_trame_hz"] = read_double(f_pic)
    # print("Frequence trame: " + str(info["freq_trame_hz"]) + " Hz")
    info["freq_pic_khz"] = read_double(f_pic)
    # print("Frequence pic: " + str(info["freq_pic_khz"]) + " kHz")
    info["norm_fact"] = read_double(f_pic)
    # print("Facteur de normalisation: " + str(info["norm_fact"]))
    tab_pics = []
    pics = read_double_tab(f_pic, info["nb_pics"])
    nb_trames = 1
    while len(pics) > 0:
        nb_trames = nb_trames+1
        tab_pics.append(pics)
        pics = read_double_tab(f_pic, info["nb_pics"])
    # print("Nb trames: " + str(nb_trames))
    f_pic.close()
    return tab_pics, info

if __name__ == "__main__":
    pics_nokey, info = get_pics_from_file("../data/pics_NOKEY.bin")
    pics_pad0, info = get_pics_from_file("../data/pics_0.bin")

    ######### Pics ############
    # NO KEY
    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(1,info["nb_pics"]+1), pics_nokey[0], 'ko')
    plt.xlabel('numéro de pic')
    plt.ylabel('valeur du pic')
    plt.title('no key')
    plt.ylim(0, 1.5)
    plt.grid(b=True, which='both')
    # PAD-0
    plt.subplot(212)
    plt.plot(range(1,info["nb_pics"]+1), pics_pad0[0], 'ko')
    plt.xlabel('numéro de pic')
    plt.ylabel('valeur du pic')
    plt.title('PAD-0')
    plt.ylim(0, 1.5)
    plt.grid(b=True, which='both')
    #
    plt.show()
    
