import cv2
import numpy as np
import subprocess
import os
import time

video_path = '/home/lucas/Documents/computer_vision/videos/marcher_180.mp4'
video_codec_fourcc = ['DIVX', 'XVID', 'MJPG', 'X264', 'WMV1', 'WMV2','FFV1', 'libx264']

def mse(im1, im2):
     return np.mean((im1-im2)**2)

for fourcc in ['FFV1']:

    cap = cv2.VideoCapture(video_path)
    temp_video_path = f'/home/lucas/Documents/computer_vision/src/computer_vision/experimentation/marcher_180_temp_{fourcc}.avi'
    final_video_path = f'/home/lucas/Documents/computer_vision/src/computer_vision/experimentation/marcher_180_final_{fourcc}.mp4'

    assert cap.isOpened(), "Error reading file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fourcc != 'libx264':
        video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w,h))
    else:
        comando_ffmpeg = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',       
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',             
            '-pix_fmt', 'bgr24',          
            '-r', str(fps),               
            '-i', '-',                    
            

            '-color_primaries', 'bt709',
            '-colorspace', 'bt709',
            '-color_trc', 'bt709',
            '-sws_flags', 'accurate_rnd+bitexact',
            
            '-vcodec', 'libx264',
            '-crf', '17',
            '-pix_fmt', 'yuv420p',       
            final_video_path
        ]
        
        process = subprocess.Popen(comando_ffmpeg, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_count = 0
    start_time = time.perf_counter()
    while cap.isOpened():
            success, im0 = cap.read()
            
            if not success:
                break
            
            if fourcc != 'libx264':
                video_writer.write(im0)
            else:
                process.stdin.write(im0.tobytes())

            frame_count+=1

            if cv2.waitKey(10) & 0xff == ord('q'):
                break

    end_time = time.perf_counter()
    print(frame_count, " Frames analysed")
    elapsed_time = end_time - start_time
    print("Time elapsed: ", elapsed_time)

    cap.release()
    if fourcc != 'libx264':
        video_writer.release()
    else:
        process.stdin.close() 
        process.wait()

    if fourcc != 'libx264':
        command = [
            'ffmpeg', 
            '-y', # Sobrescreve o arquivo final se ele já existir
            '-loglevel', 'error', # Esconde os textos chatos do ffmpeg, mostra só erros
            '-i', temp_video_path, 
            '-vcodec', 'libx264', 

            '-pix_fmt', 'yuv444p',
            '-crf', '17',
            final_video_path
        ]

        print("Running subprocess")
        start_time = time.perf_counter()
        subprocess.run(command, check=True)
        end_time = time.perf_counter()
        print("Subprocess finished")

        print(f"Elapsed time for transforming {fourcc} in h264 codec: ", end_time - start_time)

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        print("Total duration ", elapsed_time + (end_time - start_time))

    cap = cv2.VideoCapture(video_path)
    cap1 = cv2.VideoCapture(final_video_path)

    if not cap.isOpened() or not cap1.isOpened():
        print("Codec not available")
        continue

    mse_by_frames = []

    print("Comparing...")

    frame_count = 0
    while cap.isOpened() and cap1.isOpened():
        success, im = cap.read()
        success1, im1 = cap1.read()


        if not success or not success1:
            break
        
        mse_actual = mse(im,im1)
        mse_by_frames.append(mse(im, im1))

        print(frame_count, ": Different |" if mse_actual != 0 else "", end = ' ')

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Aborting...")
            break
        
        frame_count+=1
    print("\n\n")
    with open('results.txt', 'a') as file:
        file.write(f"Mean Squared Error (fourcc = {fourcc}, extension: {final_video_path.split('.')[-1]}): {np.mean(mse_by_frames)}\n")