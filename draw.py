import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 1000
WINDOWSIZEY = 600
boundryinc = 5
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
imagesave = False
model = load_model('C:/TDAK/Mnist Handwritting Digit Recognition/bestweightmnist.h5')
labels = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
predict = True

# Initialize pygame
pygame.init()
font = pygame.font.Font('freesansbold.ttf', 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption('Digit Board')
isdrawing = False
number_xcoord = []
number_ycoord = []

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEMOTION and isdrawing:
            xcoord, ycoord = event.pos
            pygame.draw.circle(DISPLAYSURF, white, (xcoord, ycoord), 4, 0)
            number_xcoord.append(xcoord)
            number_ycoord.append(ycoord)
        if event.type == pygame.MOUSEBUTTONDOWN:
            isdrawing = True
        if event.type == pygame.MOUSEBUTTONUP:
            isdrawing = False
            number_xcoord = sorted(number_xcoord)
            number_ycoord = sorted(number_ycoord)

            rect_min_x, rect_max_x = max(number_xcoord[0] - boundryinc, 0), min(WINDOWSIZEX, number_xcoord[-1] + boundryinc)
            rect_min_y, rect_max_y = max(number_ycoord[0] - boundryinc, 0), min(number_ycoord[-1] + boundryinc, WINDOWSIZEY)

            number_xcoord = []
            number_ycoord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if predict:
                # image = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                label = str(labels[np.argmax(model.predict(image.reshape(1, 28, 28, 1)))])

                textSurface = font.render(label, True, red, white)
                textRect = textSurface.get_rect()
                textRect.left, textRect.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRect)
                bounding_box = pygame.Rect(rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y)
                pygame.draw.rect(DISPLAYSURF, red, bounding_box, 2)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                DISPLAYSURF.fill(black)

    pygame.display.update()
