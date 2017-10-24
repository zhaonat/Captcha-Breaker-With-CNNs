import numpy as np
from io import BytesIO
from captcha.image import ImageCaptcha
import os
from random import randint

path = 'D:\\Documents\\CS229\\Project\\Fonts\\'
database = 'D:\\Documents\\CS229\\Project\\CaptchaDatabase\\'
fonts = [];
for file in os.listdir(path):
    fonts.append(path+file);
print(fonts)
image = ImageCaptcha(fonts=fonts)

captchlength = 4;
numGenerate = 500;
for i in range(numGenerate):
    #generate random number and convert to character
    sample = [randint(65,90) for x in range(captchlength)];
    #convert to charac6ters
    string = [chr(x) for x in sample];
    string = ''.join(string);
    print(string)
    data = image.generate(string)
    assert isinstance(data, BytesIO)
    image.write(string, database+string+'.png')