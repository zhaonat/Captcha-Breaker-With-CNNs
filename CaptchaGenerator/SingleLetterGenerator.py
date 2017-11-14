## script for generating single characeters

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
import numpy as np

path = 'D:\\Documents\\CS229\\Project\\Fonts\\'
font = ImageFont.truetype(path+"stocky.ttf", 40)
outdir = 'D:\\Documents\\CS229\\Project\\SingleLetterDatabase\\';
W, H = (40,60)

for i in range(65,90):
    letter = chr(i);
    img = Image.new("RGBA", (W, H), "white")
    draw = ImageDraw.Draw(img)
    image_size = img.size
    image_height = img.size[1]
    image_width = img.size[0]
    text_size = draw.textsize(letter,font=font)
    print(image_width); print(text_size)

    x = (image_width / 2) - (text_size[0]/2 )
    y= (image_height / 2) - (text_size[1]/2 );
    draw.text((x,y), letter, font=font, fill='black')

    #draw.text((0,0), letter, fill="black",font = font)
    img.save(outdir+letter + '.png')
    plt.imshow(np.array(img));
plt.show()
