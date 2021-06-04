from matplotlib import pyplot as plt
import cv2 
import numpy as np
from scipy.optimize import least_squares


number_images=1168
img=None
poses=[]


# /!\ La calibration a été réalisée sur des images 640 x 480 pixels et non 320 x 240 pixels  :

fx=308.03303845/2
fy=308.73431122/2
cx=323.34527019/2
cy= 294.55858154/2

# Projection Matrix : 

A = np.array([
    [fx,            0.,         cx],
    [  0.,          fy,         cy],
    [  0.,          0.,         1.]])


# Resolution (1920 x 1080) pixels 
# Taille 48x150x49 mm 
# sensor size of 12MP 


# Rappel : 
# u = fx * x' + cx => x' = ( u - cx )/fx
# v = fy * y' + cy => y' = ( v - cy )/fy

# Focale (pixel) : (fx,fy) 
# Centre de l'image : (cx,cy) 

# cf.  https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html



# Distorsion coefficients : 

D = np.array([[ 0.02696619, -0.07472997,  0.00203149, -0.0030282,   0.02261343]])




# On pose les amers suivants : 
# Balise verte 1 = A
# Balise verte 2 = B
# Balise rouge = C



# Fonction permettant de trouver l'intersection de deux cercles C0 et C1
# de centres respectifs (x0,y0) et (x1,y1)
# et rayons respectifs r0 et r1

def intersectionCircles(x0, y0, r0, x1, y1, r1):

    d=np.sqrt((x1-x0)**2 + (y1-y0)**2)

    # Pas d'intersection
    if d > r0 + r1 :
        return None
    # Un cercle à l'intérieur d'un autre
    if d < abs(r0-r1):
        return None
    # Cercles identiques
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=np.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d

        return [(x3, y3), (x4, y4)]



def intersectionPoint(centre_ARB, centre_ARC, centre_BRC, R_ARB, R_ARC, R_BRC):


    def eq(g):
        x, y = g

        return (

            np.square(x- centre_ARB[0])+ np.square(y- centre_ARB[1])-R_ARB,
            np.square(x- centre_ARC[0])+ np.square(y- centre_ARC[1])-R_ARC,
            np.square(x- centre_BRC[0])+ np.square(y- centre_BRC[1])-R_BRC
            )

    guess = (centre_ARB[0], centre_ARB[1] + np.sqrt(R_ARB))

    ans = least_squares(eq, guess, ftol=None, xtol=None)

    return ans.x



def distance2d(xA,yA,xB,yB):
    return np.sqrt(np.square(xB-xA)+np.square(yB-yA))


# On créé des masques permettant de filtrer les balises sur les images :


for i in range(number_images):

    angleA_min=angleA_max=angleB_min=angleB_max=angleC_min=angleC_max=0
    centre_ARB=centre_ARC=centre_BRC=R_ARB=R_ARC=R_BRC=np.nan

    src="images/frame"+str(i).zfill(4)+".jpg"
    image = cv2.imread(src)#[:,:,::-1]
    image = cv2.GaussianBlur(image, (5, 5), 2)
    h=len(image)
    l=len(image[0])
    centre=[int(l/2), int(h/2)]
    FOV=120#89.6
    angle_par_pixel=FOV/l


    img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,80,80]) 
    upper_red = np.array([10,255,255]) 
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    lower_red = np.array([160,80,80])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(img_hsv, lower_red, upper_red)
    mask=mask1+mask2

    # Le sol est dans la moitie inferieure : 
    mask[int(len(mask)/2):]=0
    mask=cv2.blur(mask,(5,5))
    r = cv2.bitwise_and(image, image, mask=mask)
    if np.count_nonzero(mask)==0:
        balise_min=balise_max=-1
    else:


        width=len(mask[0])

        imgray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)

        # cf. https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html 

        imageR, contoursR, hierarchyR = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for index, item in enumerate(contoursR):
            x, y, w, h = cv2.boundingRect(item)
            #center = Point((x + w)/2, (y + h)/2)
            # Attention verifier ordre coordonnees 
            #cv2.circle(v, (x,y), 5, (255,0,0), 1) 

            cv2.rectangle(r, (x, y), (x + w, y + h), (0, 0, 255), 2) 

            # Calcul du centre de la balise : 
            x=np.float32(x+w/2)
            y=np.float32(y+h/2) 

            balise_min=[x,y] 

            # Amer C dans repere camera : 
            xminC=(balise_min[0]-cx)/fx
            yminC=(balise_min[1]-cy)/fy

            angleC_min=xminC*angle_par_pixel 
            #angleC_min=(centre[0]-xminC)*angle_par_pixel

            cv2.imshow('display', r)
            cv2.waitKey(1)


       

    # attention il faut convertir balise_min dans repere camera 
    # Rappel : 
    # u = fx * x' + cx => x' = ( u - cx )/fx
    # v = fy * y' + cy => y' = ( v - cy )/fy

    

    # ajouter le theta de l'odometrie? decalage robot/camera 
    # On suppose qu'il n'y a pas de décalage entre le robot et la caméra
    # Cette hypothèse est probablement fausse, mais on s'en contentera pour le projet

    # Vert : (36,0,0) ~ (86,255,255)
    mask = cv2.inRange(img_hsv, (45,75,50), (86,255,180)) 
    v = cv2.bitwise_and(image, image, mask=mask)


    if np.count_nonzero(mask)==0:
        balise_min=balise_max=-1
    else:
        width=len(mask[0])


        imgray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)

        # cf. https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html 

        image, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for index, item in enumerate(contours):
            x, y, w, h = cv2.boundingRect(item)
            cv2.rectangle(v, (x, y), (x + w, y + h), (0, 255, 0), 2) 

            # Calcul du centre de la balise : 
            x=np.float32(x+w/2)
            y=np.float32(y+h/2)
          
            if index==0: 
                balise1_min=[x,y]
                balise1_max=[h,w] 

                # Amer A dans repere camera :  
                xminA=(balise1_min[0]-cx)/fx
                yminA=(balise1_min[1]-cy)/fy

                #angleA_min=(centre[0]-xminA)*angle_par_pixel
                angleA_min=xminA*angle_par_pixel

            if index==1:
                balise2_min=[x,y]
                balise2_max=[w,h]

                #Amer B dans repere camera : 
                xminB=(balise2_min[0]-cx)/fx
                yminB=(balise2_min[1]-cy)/fy

                #angleB_min=(centre[0]-xminB)*angle_par_pixel
                angleB_min=xminB*angle_par_pixel


    # Calcul de l'angle ARB :
    if angleA_min and angleB_min:
        ARB_min=angleB_min-angleA_min

        # D'apres la loi des sinus :
        AB=distance2d(xminA,yminA,xminB,yminB)
        #AB=distance2d(yminA,xminA,yminB,xminB)
        #AB=xminB-xminA
        R_ARB=np.abs(AB/(2*np.sin(ARB_min))) # Le rayon du cercle ARB

        Inter_ARB=intersectionCircles(xminA, yminA, R_ARB, xminB, yminB, R_ARB)
        
        # Recherche des deux points d'intersection des cercles de centre A et B 
        if Inter_ARB[0][1]>yminA:
            centre_ARB=Inter_ARB[0]
        else:
            centre_ARB=Inter_ARB[1]
        # Centre choisi 

    # Calcul de l'angle ARC : 
    if angleA_min and angleC_min:
        ARC_min=angleC_min-angleA_min

        AC=distance2d(xminA,yminA,xminC,yminC)
        #AC=distance2d(yminA,xminA,yminC,xminC)
        #AC=xminC-xminA
        print("AC", AC, xminC-xminA)
        R_ARC=np.abs(AC/(2*np.sin(ARC_min))) # Le rayon du cercle ARC

        # Recherche des deux points d'intersection des cercles de centre A et C
        Inter_ARC=intersectionCircles(xminA, yminA, R_ARC, xminC, yminC, R_ARC)

        if Inter_ARC[0][1]>yminA:
            centre_ARC=Inter_ARC[0]
        else:
            centre_ARC=Inter_ARC[1]

    # Calcul de l'angle BRC :
    if angleB_min and angleC_min:
        BRC_min=angleC_min-angleB_min
        BC=distance2d(xminB,yminB,xminC,yminC)
        #BC=distance2d(yminB,xminB,yminC,xminC)
        #BC=xminC-xminB
        R_BRC=np.abs(BC/(2*np.sin(BRC_min))) # Le rayon du cercle ARC 

        # Recherche des deux points d'intersection des cercles de centre B et C 
        Inter_BRC=intersectionCircles(xminB, yminB, R_BRC, xminC, yminC, R_BRC)

        if Inter_BRC[0][1]>yminB:
            centre_BRC=Inter_BRC[0]
        else:
            centre_BRC=Inter_BRC[1]

    # Seulement si on a trouvé les valeurs pour les 3 cercles ARB, ARC et BRC : 
    if not np.isnan(centre_ARB).any() and not np.isnan(centre_ARC).any() and  not np.isnan(centre_BRC).any() and not np.isnan(R_ARB) and not np.isnan(R_ARC) and not np.isnan(R_BRC):
        

        # On cherche la pose du robot à partir de l'intersection des 3 cercles
        pose = intersectionPoint(centre_ARB, centre_ARC, centre_BRC, R_ARB, R_ARC, R_BRC)
        if pose.all():
            poses.append(pose)
            print("Pose du robot", pose)

        cv2.imshow('display', v)
        cv2.waitKey(1)


cv2.destroyAllWindows()


fig = plt.figure()
ax = plt.axes()
poses=np.array(poses)

# /!\ OpenCV et Pyplot n'ont pas les mêmes systèmes de coordonnées 
# J'effectue tous les calculs dans le repère d'OpenCV, 
# puis j'échange à la fin lignes et colonnes pour l'affichage avec Pyplot

x=poses[:,0]
y=poses[:,1]
ax.plot(y,x);
plt.xlabel('x')
plt.ylabel('y')
plt.title('Position')
plt.show()







# Je n'ai pas réussi faire marcher l'algorithme ensembliste SIVIA
# à la place je trouve l'intersection des 3 cercles par la méthode des moindres carrés


# Set Inversion Via Interval Analysis  

#dimension pièce
x0=np.array([120, 210])
epsilon=1
emptySet=np.array([])#set() 

def SIVIA(x0, epsilon):

    print("Start SIVIA")
    L=np.array([x0])

    while len(L):
        x, L = L[-1], L[:-1]
        print("x", x)
        # Fonction d'inclusion 
        testC1 = np.square(x[0]- centre_ARB[0])+ np.square(x[1]- centre_ARB[1])-R_ARB
        testC2 = np.square(x[0]- centre_ARC[0])+ np.square(x[1]- centre_ARC[1])-R_ARC
        testC3 = np.square(x[0]- centre_BRC[0])+ np.square(x[1]- centre_BRC[1])-R_BRC

        print(testC1, testC2, testC3)
        if testC1 in [0,0] and testC2 in [0,0] and testC3 in [0,0]:
            print(x, "est solution")
            Xmoins.append(x)
        else:
            if len(np.intersect1d(testC1,emptySet)) or len(np.intersect1d(testC2,emptySet)) or len(np.intersect1d(testC2,emptySet)) :
                Xmoins.append(x)
                print("pas solution")
            else:
                print("bissect")
                if len(x)>epsilon:
                    #x1, x2 = np.array_split(x, 2)
                    print(L, x1, x2)
                    L=np.append(L,x1)
                    L=np.append(L,x2)
                else:
                    print(x, "peut etre solution")
                    Xplus.append(x)





