import cv2 as cv

video = cv.VideoCapture('../Dataset/1.mp4')
con, img = video.read()
output = cv.VideoWriter('../Result/opencv.mp4', cv.VideoWriter.fourcc('m', 'p', '4', 'v'), 30,
                        (img.shape[1], img.shape[0]))
bbox = cv.selectROI(img, False)
tracker = cv.TrackerKCF.create()
tracker.init(img, bbox)

while True:
    con, img = video.read()
    if not con:
        break
    success, bbox = tracker.update(img)
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv.putText(img, "Tracking failure detected", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.imshow('img', img)
    output.write(img)
    cv.waitKey(1)

video.release()
cv.destroyAllWindows()
