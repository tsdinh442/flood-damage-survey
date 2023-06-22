import cv2
import os

folder_path = './frames/6/'
file_names = os.listdir(folder_path)
# Sort the file names numerically, excluding non-numeric file names
sorted_file_names = sorted(
    [f for f in file_names if f.split(".")[0].isdigit()],
    key=lambda x: int(x.split(".")[0])
)

output_file = "./frames/6/ output_video.mp4"
frame_width, frame_height = 1920, 1080  # Specify the desired frame size
fps = 22  # Specify the frames per second
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the video codec (e.g., "mp4v", "XVID", etc.)
output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

for frame_file in sorted_file_names:
    frame_path = os.path.join(folder_path, frame_file)
    frame = cv2.imread(frame_path)

    # Resize the frame if necessary
    #frame = cv2.resize(frame, (frame_width, frame_height))

    output_video.write(frame)

output_video.release()
cv2.destroyAllWindows()