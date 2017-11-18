## function which takes an input image and outputs 60x40 windows

def MovingWindow(img):
    W = 40;
    d = img.shape;
    columns = d[1];
    start = 0;
    imgStore = list();
    for i in range(0,columns, 10):
        window = img[:,start+i:W+i];
        imgStore.append(window);

    return imgStore;
