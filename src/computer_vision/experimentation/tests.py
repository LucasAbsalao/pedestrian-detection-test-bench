import cv2
import numpy as np
import matplotlib
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

cap = cv2.VideoCapture("/home/lucas/Documents/computer_vision/videos/distortion/marcher_fog.mp4")

assert cap.isOpened(), "Error reading video file"


w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
#video_writer = cv2.VideoWriter(str(project_path / f"{str(args.name)}.avi"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

cmap = matplotlib.colormaps.get_cmap("Spectral")

while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break
       

    new_img = (im0 - np.min(im0)) / (np.max(im0) - np.min(im0))

    new_img = (cmap(new_img[:,:,1].squeeze())[:,:,:3] * 255)[:,:,::-1].astype(np.uint8)

    print(new_img.shape)

    cv2.imshow("teste", new_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        #video_writer.write(im0)

cap.release()
#video_writer.release()
cv2.destroyAllWindows()
