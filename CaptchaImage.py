def RandomCharSet(charCount=4):
    import Definitions
    import random as rand

    charChoices = []

    for i in range(charCount):
        charChoices.append(str(rand.choice(Definitions.charSet)))

    return charChoices

def GetCaptchaTextAndImage(charCount=4):
    from captcha.image import ImageCaptcha
    import numpy as np
    from PIL import Image

    captcha = ImageCaptcha()
    charSet = RandomCharSet(charCount)
    image = captcha.generate(charSet)
    image = Image.open(image)
    imagenp = np.array(image)

    # from imagesplit import image
    return charSet, imagenp, image

def ReadImage(path):
    import numpy as np
    from PIL import Image

    image = Image.open(path)
    image = np.array(image)

    return image

def TestCaptchaImage():
    from PIL import Image
    from matplotlib.pyplot import imshow
    text, imagenp, image = GetCaptchaTextAndImage(4)

    print(text)
    imshow(imagenp)
    print(type(imagenp))
    print(imagenp.shape)
