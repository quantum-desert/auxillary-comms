#SPDC Coincidence Count Data
import os
import time
import matplotlib.pyplot as plt

# read in
os.chdir(r"C:\Users\liamw\OneDrive\GraduateSchool\UM\QE_LAB\Quantum Illumination\data")
fname = "Bidirectional_histogram_2023-11-29_133047.txt"
deltaT = [] # ns
count = [] # counts
with open(fname,"r") as f:
    data = f.readlines()
    ti = round(time.time()*1e3)
    print("Reading Data...")
    header = data.pop(0)
    for line in data:
        dsplit =(line.strip("\n")).split("\t")
        deltaT.append((int(dsplit[0])/1e3))
        count.append(int(dsplit[1]))
    tf = round(time.time()*1e3)

print("Done: %i ms"%(int(tf-ti)))

# max idx
max_idx = count.index(max(count))
# plot data
print("Plotting...")
plt.plot(deltaT,count)
plt.ylabel("Count")
plt.xlabel("Delta T (ns)")
#plt.xlim([count[max_idx-100],count[max_idx+100]])
plt.xlim([100,200])
plt.grid("on")
plt.show()

