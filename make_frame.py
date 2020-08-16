import cv2

vidcap = cv2.VideoCapture('movie/test1.mp4') ### change video point here
 
count = 0
 
while(vidcap.isOpened()):
    ret, image = vidcap.read()
 
    if(int(vidcap.get(1)) % 20 == 0): ### change frame number here
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        cv2.imwrite('cropped_trainset/fake/frame%d.jpg' % count, image) ### change save point here
        print('Saved frame%d.jpg' % count)
        count += 1

vidcap.release()
