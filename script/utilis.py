from matplotlib import pyplot as plt


def show_image(image, title="image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()