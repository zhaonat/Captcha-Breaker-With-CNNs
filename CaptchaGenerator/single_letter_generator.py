## Script for generating single characters

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os
import dev_constants

path = dev_constants.MY_PROJECT_PATH
fontDir = 'Fonts'
outDir = 'Single letter training set'

fontName = 'calibri'
font = ImageFont.truetype(os.path.join(path, fontDir, (fontName + '.ttf')), 40)

W, H = (40, 60)

for i in range(65, 91):
    letter = chr(i)
    img = Image.new("RGBA", (W, H), "white")
    draw = ImageDraw.Draw(img)
    image_size = img.size
    image_height = img.size[1]
    image_width = img.size[0]
    text_size = draw.textsize(letter,font=font)

    x = (image_width / 2) - (text_size[0]/2 )
    y= (image_height / 2) - (text_size[1]/2 )
    draw.text((x,y), letter, font=font, fill='black')

    # draw.text((0,0), letter, fill="black",font = font)
    img.save(os.path.join(path, outDir, (letter + '.png')))
    # plt.imshow(np.array(img))
