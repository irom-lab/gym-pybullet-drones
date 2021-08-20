from urdfpy import URDF
object = URDF.load('gym_pybullet_drones/assets/cf2x.urdf')
for link in object.links:
    print(link.name)
object.show(cfg={})
