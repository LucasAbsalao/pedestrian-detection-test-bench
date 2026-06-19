import cv2
import numpy as np
import random

def write_lines(image, point_d, point_u, width):
    point_r1 = point_d
    point_r2 = (width - point_d[0], point_d[1])

    point_g1 = point_u
    point_g2 = (width - point_u[0], point_u[1])

    point_r3 = ((point_u[0]-point_d[0])//3 + point_d[0], point_d[1] - (point_d[1]-point_u[1])//3)
    point_r4 = (width - ((point_u[0]-point_d[0])//3 + point_d[0]), point_d[1] - (point_d[1]-point_u[1])//3)

    point_o1 = (2*(point_u[0]-point_d[0])//3 + point_d[0], point_d[1] - 2*(point_d[1]-point_u[1])//3)  
    point_o2 = (width - (2 * (point_u[0]-point_d[0])//3 + point_d[0]), point_d[1] - 2*(point_d[1]-point_u[1])//3)
    
    image = cv2.line(image, point_r1, point_r2, color=(0,0,255), thickness=3)
    image = cv2.line(image, point_r3, point_r4, color = (0,0,255), thickness=3)
    image = cv2.line(image, point_r1, point_r3, color = (0,0,255), thickness=3)
    image = cv2.line(image, point_r2, point_r4, color = (0,0,255), thickness=3)

    image = cv2.line(image, point_r3, point_o1, color=(0,150,255), thickness=3)
    image = cv2.line(image, point_r4, point_o2, color=(0,150,255), thickness=3)
    image = cv2.line(image, point_o1, point_o2, color=(0,150,255), thickness=3)

    image = cv2.line(image, point_o1, point_g1, color=(0,255,0), thickness=3)
    image = cv2.line(image, point_o2, point_g2, color=(0,255,0), thickness=3)
    image = cv2.line(image, point_g1, point_g2, color=(0,255,0), thickness=3)

    trapezes = [[point_r1, point_r3, point_r4, point_r2], 
                [point_r3, point_o1, point_o2, point_r4], 
                [point_o1, point_g1, point_g2, point_o2]]
    
    return image, trapezes 

def intersect(point, trapezes, colors):
    i = 0
    for trapeze in trapezes:
        ld, lu, ru, rd = trapeze
        print(ld, lu, ru, rd)
        print(point)
        if (point[0]>ld[0] and point[0]<rd[0]) and (point[1]<ld[1] and point[1]>lu[1]):
            ang_coef = (lu[1]-ld[1])/(lu[0]-ld[0])
            if lu[0] <= point[0] <= ru[0]:
                print("a")
                return colors[i]
            elif point[0] < lu[0]:
                print("b")
                lateral_limit_l = ld[1] + ang_coef * (point[0]-ld[0])
                if point[1]>lateral_limit_l:
                    return colors[i] 
            elif point[0]>ru[0]:
                print("c")
                lateral_limit_l = ru[1] - ang_coef * (point[0]-ru[0])
                if point[1]>lateral_limit_l:
                    return colors[i] 
        i+=1
                
    return colors[-1]



width = 1080
height = 720

image = np.zeros((height, width,3), dtype = np.uint8)

point_1 = (300, 700)
point_2 = (500, 300)

image, trapezes = write_lines(image, point_1, point_2, width)

colors = [(0,0,255), (0,150,255), (0,255,0), (255,0,0)]

centers = np.random.randint(0, [width, height], size=(50,2))

for center in centers:
    color = intersect(center, trapezes, colors)
    cv2.circle(image, center=center, radius=1, color=color, thickness=-1)

    cv2.imshow("Teste", image)
    cv2.waitKey(2)

# center = (int(random.uniform(0,width)), int(random.uniform(0,height)))
# print(center)

print("trapezes: ", trapezes) #[[(300, 700), (366, 567), (714, 567), (780, 700)], [(366, 567), (433, 434), (647, 434), (714, 567)], [(433, 434), (500, 300), (580, 300), (647, 434)]]

cv2.imshow("Teste", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

