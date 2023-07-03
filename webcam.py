import cv2
from spectral_analyzer import SpectralAnalyzer

window_name = "Spectrometer"

region_start_x = 800
region_start_y = 300
region_end_x = 1270
region_end_y = 700
rs_x = region_start_x + 2
rs_y = region_start_y + 2
re_x = region_end_x - 2
re_y = region_end_y - 2
  
sa = SpectralAnalyzer()
  
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = capture.read()
cv2.imshow(window_name, frame)
  
while cv2.getWindowProperty(window_name, 0) >= 0:
    ret, frame = capture.read()

    frame = cv2.rectangle(
        frame, 
        (region_start_x, region_start_y), 
        (region_end_x, region_end_y), 
        (0, 255, 0), 
        2
    )
  
    cv2.imshow(window_name, frame)
      
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'): # escape or q
        break
    if ret and key == 32: # space
        sa.analyze(frame[rs_y:re_y, rs_x:re_x])


capture.release()
cv2.destroyAllWindows()