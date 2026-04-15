import cv2
img = cv2.imread("detected-ans6/02_warped.jpg")
# Rightmost column x=522-669, rows 21-24 cy=837-919
# Add some context buffer
crop = img[820:960, 510:690]
cv2.imwrite("detected-ans6/debug_q23_q25_crop.jpg", crop)
print("Saved crop for Q23-Q25 region")
