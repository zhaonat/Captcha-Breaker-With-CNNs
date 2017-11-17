## generate captcha from letters in the single_letter database

## this matters later, just focus on single digit recognition
import dev_constants as dev
import os
import matplotlib.pyplot as plt
from PIL import Image;

single_letter_database = dev.MY_PROJECT_PATH+'\\SingleLetterDatabase'

for letterFile in os.listdir(single_letter_database):
    img = Image.open(single_letter_database+'\\'+letterFile);
    plt.imshow(img)
    plt.show()