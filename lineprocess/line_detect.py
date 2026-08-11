import cv2
import numpy as np
import random


# ===================== 1. RANSAC直线拟合 =====================

def ransac_line_fit(points, iter_num=150, dist_thresh=2.2):

    if points is None or len(points) < 2:
        return None

    best_inliers = 0
    best_model = None

    n = len(points)

    for _ in range(iter_num):

        idx = random.sample(range(n), 2)

        p1 = points[idx[0]]
        p2 = points[idx[1]]

        x1, y1 = p1
        x2, y2 = p2

        # 两点重合
        if x1 == x2 and y1 == y2:
            continue

        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2


        denom = np.sqrt(a*a+b*b)

        if denom < 1e-6:
            continue


        dists = np.abs(
            a*points[:,0] +
            b*points[:,1] +
            c
        ) / denom


        inliers = np.sum(dists < dist_thresh)


        if inliers > best_inliers:

            best_inliers = inliers
            best_model = (a,b,c)


    return best_model



# ===================== 2. 透视变换 =====================

def get_perspective_matrix(img_h,img_w):

    src=np.float32([
        [img_w*0.12,img_h],
        [img_w*0.45,img_h*0.62],
        [img_w*0.55,img_h*0.62],
        [img_w*0.88,img_h]
    ])


    dst=np.float32([
        [img_w*0.15,img_h],
        [img_w*0.15,0],
        [img_w*0.85,0],
        [img_w*0.85,img_h]
    ])


    M=cv2.getPerspectiveTransform(src,dst)

    M_inv=cv2.getPerspectiveTransform(dst,src)


    return M,M_inv



# ===================== 3. 颜色阈值 =====================

def lane_color_threshold(img):

    hsv=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )


    # 白色
    lower_white=np.array([0,0,200])
    upper_white=np.array([180,30,255])


    mask_white=cv2.inRange(
        hsv,
        lower_white,
        upper_white
    )


    # 黄色

    lower_yellow=np.array([10,60,100])
    upper_yellow=np.array([40,255,255])


    mask_yellow=cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )


    return cv2.bitwise_or(
        mask_white,
        mask_yellow
    )



# ===================== 4. 绘制拟合线 =====================

def draw_line_get_points(
        img,
        line_coeff,
        y_bottom,
        y_top,
        color):


    if line_coeff is None:
        return None,None


    a,b,c=line_coeff


    # 防止水平线

    if abs(a)<1e-6:
        return None,None



    x1=int((-b*y_bottom-c)/a)
    x2=int((-b*y_top-c)/a)


    h,w=img.shape[:2]


    x1=np.clip(
        x1,
        0,
        w-1
    )

    x2=np.clip(
        x2,
        0,
        w-1
    )


    p1=(x1,y_bottom)
    p2=(x2,y_top)


    cv2.line(
        img,
        p1,
        p2,
        color,
        4
    )


    return p1,p2




# ===================== 主程序 =====================


if __name__=="__main__":


    frame=cv2.imread("1.jpeg")


    if frame is None:

        h,w=500,800

        frame=np.zeros(
            (h,w,3),
            dtype=np.uint8
        )


        cv2.fillPoly(
            frame,
            [
                np.array(
                    [
                        [80,h],
                        [320,310],
                        [480,310],
                        [720,h]
                    ])
            ],
            (60,60,60)
        )


        cv2.line(
            frame,
            (120,h),
            (340,310),
            (255,255,255),
            6
        )


        cv2.line(
            frame,
            (680,h),
            (460,310),
            (255,255,0),
            6
        )



    h_img,w_img=frame.shape[:2]


    M,M_inv=get_perspective_matrix(
        h_img,
        w_img
    )



    # 1 高斯滤波

    blur=cv2.GaussianBlur(
        frame,
        (5,5),
        1.3
    )



    # 2 颜色

    mask=lane_color_threshold(
        blur
    )



    # 3 Canny

    edges=cv2.Canny(
        mask,
        50,
        160
    )



    # 4 鸟瞰图

    bird=cv2.warpPerspective(
        edges,
        M,
        (w_img,h_img)
    )



    # 5 轮廓过滤

    contours,_=cv2.findContours(
        bird,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    clean=np.zeros_like(bird)


    for cnt in contours:

        area=cv2.contourArea(cnt)

        if 15 < area < 800:

            cv2.drawContours(
                clean,
                [cnt],
                -1,
                255,
                -1
            )



    # 6 Hough

    hough_lines=cv2.HoughLinesP(
        clean,
        1,
        np.pi/180,
        threshold=35,
        minLineLength=25,
        maxLineGap=12
    )



    left_points=[]
    right_points=[]


    hough_draw=cv2.cvtColor(
        clean,
        cv2.COLOR_GRAY2BGR
    )



    if hough_lines is not None:


        for line in hough_lines:


            # ★关键修复
            x1,y1,x2,y2=np.array(line).reshape(-1)


            if abs(x2-x1)<1:
                continue



            slope=(y2-y1)/(x2-x1)


            mid_x=(x1+x2)/2



            cv2.line(
                hough_draw,
                (x1,y1),
                (x2,y2),
                (0,150,255),
                2
            )



            if slope>0.2 and mid_x<w_img//2:

                left_points.append(
                    [x1,y1]
                )

                left_points.append(
                    [x2,y2]
                )


            elif slope<-0.2 and mid_x>w_img//2:


                right_points.append(
                    [x1,y1]
                )

                right_points.append(
                    [x2,y2]
                )



    # 7 RANSAC


    left_line=None
    right_line=None



    if len(left_points)>3:

        left_line=ransac_line_fit(
            np.array(left_points)
        )



    if len(right_points)>3:

        right_line=ransac_line_fit(
            np.array(right_points)
        )



    lane=cv2.cvtColor(
        clean,
        cv2.COLOR_GRAY2BGR
    )


    p_left1,p_left2=draw_line_get_points(
        lane,
        left_line,
        h_img,
        int(h_img*0.25),
        (0,255,0)
    )


    p_right1,p_right2=draw_line_get_points(
        lane,
        right_line,
        h_img,
        int(h_img*0.25),
        (0,255,0)
    )



    # 8 车道区域

    result=frame.copy()


    if (
        p_left1 and
        p_left2 and
        p_right1 and
        p_right2
    ):


        poly=np.array(
            [
                p_left1,
                p_left2,
                p_right2,
                p_right1
            ],
            np.int32
        )


        overlay=np.zeros_like(frame)


        cv2.fillPoly(
            overlay,
            [poly],
            (0,180,80)
        )


        overlay=cv2.warpPerspective(
            overlay,
            M_inv,
            (w_img,h_img)
        )


        result=cv2.addWeighted(
            frame,
            1,
            overlay,
            0.4,
            0
        )



    # 显示

    row1=np.hstack(
        [
            blur,
            cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        ]
    )


    row2=np.hstack(
        [
            cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(bird,cv2.COLOR_GRAY2BGR)
        ]
    )


    row3=np.hstack(
        [
            hough_draw,
            lane
        ]
    )


    panel=np.vstack(
        [
            row1,
            row2,
            row3
        ]
    )


    panel=cv2.resize(
        panel,
        (w_img,h_img)
    )


    show=np.hstack(
        [
            panel,
            result
        ]
    )



    cv2.imwrite(
        "lane detection.jpg",
        show
    )


    # cv2.waitKey(0)

    # cv2.destroyAllWindows()