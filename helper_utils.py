import numpy as np
import matplotlib.pyplot as plt
import torchvision

class HelperUtils():
    def image_show(image):
        image = torchvision.utils.make_grid(image)
        image = image / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.show()
