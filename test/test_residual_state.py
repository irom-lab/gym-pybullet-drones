import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12


csv_path = 'lcmlog-2022-09-13.11.csv'
data = pd.read_csv(csv_path)  

t = data['timestamp']
x = data['x']
y = data['y'] 
z = data['z']

pres = data['pres']
qres = data['qres'] 
rres = data['rres']

# z,roll,pitch,yaw,vx,vy,vz,p,q,r,psp,qsp,rsp,pres,qres,rres,thrust_sp, thrust_res
fig, axarr = plt.subplots(3, 6, figsize=(32,16))
axarr[0,0].plot(t, data['x'])
axarr[1,0].plot(t, data['y'])
axarr[2,0].plot(t, data['z'])
axarr[0,1].plot(t, data['pres'])
axarr[1,1].plot(t, data['qres'])
axarr[2,1].plot(t, data['rres'])
axarr[0,2].plot(t, data['roll'])
axarr[1,2].plot(t, data['pitch'])
axarr[2,2].plot(t, data['yaw'])
axarr[0,3].plot(t, data['vx'])
axarr[1,3].plot(t, data['vy'])
axarr[2,3].plot(t, data['vz'])
axarr[0,4].plot(t, data['p'])
axarr[1,4].plot(t, data['q'])
axarr[2,4].plot(t, data['r'])
# plt.show()
plt.savefig('test.png')
