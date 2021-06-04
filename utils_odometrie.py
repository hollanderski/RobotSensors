#! /usr/bin/env python

# CSV
import pandas as pd

# Math library
import numpy as np

# Plotting library
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math


df_encoder = pd.read_csv('sensor_state.csv')
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(df_encoder['%time'].tolist(), df_encoder['field.right_encoder'].tolist(), label='right encoder')
ax.plot(df_encoder['%time'].tolist(), df_encoder['field.left_encoder'].tolist(), label='left encoder')
ax.legend()
plt.xlabel('time (s)')
plt.show()



# Calcul de la vitesse : 

maliste= df_encoder['field.right_encoder'] 

N=4096 # resolution de l'encodeur
T=20   # periode calculee a partir de la frequence
frequence=0.05
# Attention conversion de mm en m 
rayon = 0.033 # 33 mm
L= 0.080 # 80mm demi axe

# Pas de temps (calcul frequence difference entre mesure)
# a. df_encoder['left_encoder'].diff()
# b. si elle est constante, utiliser la frequence d'echantillonage   
# Dans notre cas, la frequence est constante et vaut 20Hz
																	
   
# Vitesse du centre des roues gauche et droite :
phi1=df_encoder['field.left_encoder'].diff() /frequence * 2*math.pi/N
phi2=df_encoder['field.right_encoder'].diff() /frequence * 2*math.pi/N

print(phi1, phi2)

# Vitesse au centre des deux roues
# Vitesse moyenne donne vitesse longitudinale 
# on multiplie les vitesses angulaires par le rayon 

# Vitesse longitudinale : moyenne des vitesse au centre 
v=rayon*(phi1+phi2)/2  #speed 
# que des valeurs positives car le robot ne fait qu'avancer

# Vitesse de rotation i.e vitesse angulaire
w=rayon*(phi2-phi1)/(2*L) #yaw rate
# negative quand on tourne a droite, positive quand on tourne a gauche

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(df_encoder['%time'].tolist(), v, label='vitesse longitudinale')
ax.plot(df_encoder['%time'].tolist(), w, label='vitesse de rotation angulaire')
#plt.ylim(-0.2, 0.2)
ax.legend()
plt.xlabel('time (s)')
plt.show()

# on calcule le deplacement longitudinal et on le projette dans le repere 

df_encoder['rotation']=w


angles=w*frequence

angles_cs=np.cumsum(angles)

x=v*np.cos(angles_cs)*frequence
y=v*np.sin(angles_cs)*frequence

x_cs=np.cumsum(x)
y_cs=np.cumsum(y) 



fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x_cs, y_cs , label='Odometrie')
#plt.ylim(-0.2, 0.2)
ax.legend()
plt.xlabel('time (s)')
plt.show()



df_imu = pd.read_csv('imu.csv')

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(df_encoder['%time'].tolist(), w, label='odometrie')
ax.plot(df_imu['%time'].tolist(), df_imu['field.angular_velocity.z'].tolist(), label='gyrometre')
#plt.ylim(-0.2, 0.2)
ax.legend()
plt.xlabel('time (s)')
plt.show()

# Les donnees du gyrometre sont bruitees lors des deplacements du turtlebot
# La frequence de la centrale inertielle est 100Hz
# Le magnetometre pointe vers le nord ou le sud, 
# c'est un vecteur qui nous permet de deduire l'orientation du vecteur et donc de l'angle 


df_magneto = pd.read_csv('magnetic_field.csv')
angle_magn=np.arctan2(df_magneto['field.magnetic_field.y'] ,df_magneto['field.magnetic_field.x'])-1.5

# Ajuster le zero a ete necessaire pour le magnetometre


angles_gyr=df_imu['field.angular_velocity.z']*0.01
angles_gyr_cs=np.cumsum(angles_gyr)

# Rotations de lacet, roulis et tangage :
# Lacet autour de l'axe Z
# Tangage autour de l'axe Y 
# Roulis autour de l'axe X

# On utilise la vitesse de rotation angulaire autour de l'axe Z 
# donc il s'agit de l'angle de lacet

# Comparaison de l angle obtenu pour le gyrometre, le magnetometre et l'ondometre :

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(df_imu['%time'].tolist(), angles_gyr_cs, label='gyrometre')
ax.plot(df_magneto['%time'].tolist(), angle_magn, label='magnetometre')
ax.plot(df_encoder['%time'].tolist(), angles_cs, label='ondometrie')

ax.legend()
plt.xlabel('time (s)')
plt.show()

