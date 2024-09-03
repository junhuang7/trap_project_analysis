from tqdm import tqdm  # for progress bar
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
import subprocess
import sys

def ask_for_video_path():
    # Initialize Tkinter root
    root = tk.Tk()
    # Hide the main window (optional)
    root.withdraw()
    # Open the file selection dialog and store the selected file path
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
    )
    # Return the selected file path
    return file_path


def select_crop_regions(frame):
    # Initialize resizable window for crop region selection
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  # Create a window that can be resized
    cv2.resizeWindow('Frame', 800, 600)  # Default window size

    regions = []  # To store regions defined by the user
    current_region = []  # To store the current region being drawn

    def mouse_click(event, x, y, flags, param):
        nonlocal current_region
        if event == cv2.EVENT_LBUTTONDOWN:  # Left button click
            current_region.append((x, y))  # Add point to current region
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Mark the clicked point
            if len(current_region) > 1:
                # Draw lines between points to show the polygon being formed
                cv2.polylines(frame, [np.array(current_region)], False, (255, 0, 0), 2)
            cv2.imshow('Frame', frame)  # Show the updated frame

    cv2.setMouseCallback('Frame', mouse_click)

    print("Draw a region. Press 'ENTER' to complete the region, 'ESC' to finish.")
    while True:
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('\n') or key == ord('\r'):  # Handles Enter key for different OS
            if len(current_region) > 2:  # A valid polygon requires at least 3 points
                # Automatically close the polygon by connecting the last point to the first
                if current_region[-1] != current_region[0]:
                    current_region.append(current_region[0])
                regions.append(current_region)
                # Draw the complete polygon
                cv2.polylines(frame, [np.array(current_region)], True, (0, 255, 255), 2)
                current_region = []  # Reset for the next region
                cv2.imshow('Frame', frame)  # Show the frame with the completed polygon
            else:
                print("A polygon must have at least 3 points.")
        elif key == 27:  # ESC key to finish
            break

    cv2.destroyAllWindows()
    return regions


def initialize_ffmpeg_process(region,output_path, fps):
    x, y, w, h = cv2.boundingRect(np.array(region))
    # Adjust dimensions for H.264 compatibility
    w -= w % 2
    h -= h % 2

    # check if it's  PC or Mac
    if os.name == 'nt':
        ffmpeg_command = "M:\\Software\FFmpeg\\bin\\ffmpeg.exe"
    else:
        ffmpeg_command = 'ffmpeg' # Assuming 'ffmpeg' is in PATH
  
    command = [ffmpeg_command,
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', '{}x{}'.format(w, h),  # Original dimensions for input frames
           '-r', str(fps),
           '-i', '-',
           '-vf', "scale='min(1080,iw)':-1,pad=1080:1080:(1080-iw)/2:(1080-ih)/2",
           '-an',
           '-vcodec', 'libx264',
           '-preset', 'fast',
           '-crf', '25',
           '-pix_fmt', 'yuv420p',
           output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def write_cropped_frame_to_ffmpeg(frame, region, process):
    x, y, w, h = cv2.boundingRect(np.array(region))
    # Adjust dimensions
    w -= w % 2
    h -= h % 2
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(region)], 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    cropped_frame = masked_frame[y:y+h, x:x+w]

    if cropped_frame.size > 0 and w > 0 and h > 0:
        process.stdin.write(cropped_frame.tobytes())


def main():
    video_path = ask_for_video_path()
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
    regions = select_crop_regions(frame)
    animal_IDs = [input(f"Enter the ID for animal #{i+1} in the selected regions: ") for i in range(len(regions))]
    
    output_dir = os.path.join('D:/temp_cropped_videos/', 'cropped')
    os.makedirs(output_dir, exist_ok=True)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize FFmpeg processes for each region
    ffmpeg_processes = [initialize_ffmpeg_process(region, os.path.join(output_dir, f'cropped_{animal_IDs[i]}.mp4'), fps) 
                        for i, region in enumerate(regions)]
    
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#    for _ in tqdm(range(total_frames), desc="Processing Video"):
    for _ in tqdm(range(total_frames), desc="Processing Video", file=sys.stdout, ncols=100, bar_format='{l_bar}{bar}{r_bar}'):

        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply mask, crop, and write frames to their corresponding FFmpeg process
        [write_cropped_frame_to_ffmpeg(frame, regions[idx], process) for idx, process in enumerate(ffmpeg_processes)]
    
    # Finalize FFmpeg processes
    [process.stdin.close() for process in ffmpeg_processes]
    [process.wait() for process in ffmpeg_processes]
    
    cap.release()

# implement parallel processing
#    import concurrent.futures

def process_video(video_path, regions, animal_IDs, output_dir):
    # video processing logic here, adjusted to work with the parameters provided
    # This is a simplified version of your 'main' function that processes a single video
    x = []


def main_parallel():
    videos = ["path/to/video1.mp4", "path/to/video2.mp4"]  # Example list of videos
    regions = [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]]  # Example regions for each video
    animal_IDs = [["ID1", "ID2"], ["ID3", "ID4"]]  # Example IDs for animals in each video
    output_dir = 'D:/temp_cropped_videos/cropped'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each video in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for video_path, video_regions, video_animal_IDs in zip(videos, regions, animal_IDs):
            futures.append(executor.submit(process_video, video_path, video_regions, video_animal_IDs, output_dir))
        # Wait for all futures to complete
        concurrent.futures.wait(futures)

## call main function
if __name__ == "__main__":
    main()
    # main_parallel()

