import cv2

WIDTH = 1280
HEIGHT = 720

carCascade = cv2.CascadeClassifier('myhaar.xml')
carCascade = cv2.CascadeClassifier('./xml/two_wheeler.xml')

image = cv2.imread("./imagens/moto.png")
#image = cv2.resize(image, (WIDTH, HEIGHT))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

for (_x, _y, _w, _h) in cars:
    x = int(_x)
    y = int(_y)
    w = int(_w)
    h = int(_h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('image', image)

cv2.waitKey(0)
    
cv2.destroyAllWindows()
