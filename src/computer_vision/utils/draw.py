import cv2

ZONE_COLORS = [(0, 0, 255), (0, 150, 255), (0, 255, 0), (255, 0, 0)]



def intersect(point, trapezes, verbose = False):
    i = 0
    for trapeze in trapezes:
        ld, lu, ru, rd = trapeze
        if verbose:
            print("Trapeze: ", ld, lu, ru, rd, "| Point: ", point)
        if (point[0]>ld[0] and point[0]<rd[0]) and (point[1]<ld[1] and point[1]>lu[1]):
            ang_coef = (lu[1]-ld[1])/(lu[0]-ld[0])
            if lu[0] <= point[0] <= ru[0]:
                return i
            elif point[0] < lu[0]:
                lateral_limit_l = ld[1] + ang_coef * (point[0]-ld[0])
                if point[1]>lateral_limit_l:
                    return i
            elif point[0]>ru[0]:
                lateral_limit_l = ru[1] - ang_coef * (point[0]-ru[0])
                if point[1]>lateral_limit_l:
                    return i
        i+=1
                
    return i


def draw_bbox(image, bbox, trapezes, rectangle = True):
    colors = [(0,0,255), (0,150,255), (0,255,0), (255,0,0)]
    zones = []
    for box in bbox:
        points = [(box[0], box[1]), (box[0], box[3]), (box[2], box[3]), (box[2], box[1])]
        colors_circles = []
        for p in points:
            colors_circles.append(intersect(p, trapezes))

        if rectangle:
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=colors[min(colors_circles)], thickness=3)
        else:
            cv2.circle(image, center=(int(box[0]), int(box[1])), radius=6, color=colors[colors_circles[0]], thickness=-1)
            cv2.circle(image, center=(int(box[0]), int(box[3])), radius=6, color=colors[colors_circles[1]], thickness=-1)
            cv2.circle(image, center=(int(box[2]), int(box[3])), radius=6, color=colors[colors_circles[2]], thickness=-1)
            cv2.circle(image, center=(int(box[2]), int(box[1])), radius=6, color=colors[colors_circles[3]], thickness=-1)

        zones.append(min(colors_circles))

    return zones

def draw_bboxes_from_data(image, frame_data):
    """Draw bounding boxes on the image colored by zone."""
    for zone, bbox in frame_data:
        x1, y1, x2, y2 = map(int, bbox)
        color = ZONE_COLORS[min(zone, 3)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=3)
        label = f"Zone: {zone}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image

def generate_trapezes(point_d, point_u, width):
    point_r1 = point_d
    point_r2 = (width - point_d[0], point_d[1])

    point_g1 = point_u
    point_g2 = (width - point_u[0], point_u[1])

    point_r3 = ((point_u[0]-point_d[0])//3 + point_d[0], point_d[1] - (point_d[1]-point_u[1])//3)
    point_r4 = (width - ((point_u[0]-point_d[0])//3 + point_d[0]), point_d[1] - (point_d[1]-point_u[1])//3)

    point_o1 = (2*(point_u[0]-point_d[0])//3 + point_d[0], point_d[1] - 2*(point_d[1]-point_u[1])//3)  
    point_o2 = (width - (2 * (point_u[0]-point_d[0])//3 + point_d[0]), point_d[1] - 2*(point_d[1]-point_u[1])//3)

    trapezes = [[point_r1, point_r3, point_r4, point_r2], 
                [point_r3, point_o1, point_o2, point_r4], 
                [point_o1, point_g1, point_g2, point_o2]]

    return trapezes

def write_lines(image, trapezes):
    point_r1 = trapezes[0][0]
    point_r2 = trapezes[0][3]

    point_g1 = trapezes[2][1]
    point_g2 = trapezes[2][2]

    point_r3 = trapezes[0][1]
    point_r4 = trapezes[0][2]

    point_o1 = trapezes[1][1]
    point_o2 = trapezes[1][2]
    
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
    
    return image 


def write_lines_from_points(image, point_d, point_u, width):
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
    
    return image 