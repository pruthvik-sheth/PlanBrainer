from dataloader.dataloader import DataLoader
import os
from matplotlib import pyplot as plt

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    loader = DataLoader()
    real_img, input_img = loader.load_image("./data/ready/combined/1.png")
    plt.figure()
    # Reason to / 255.0 is:
    # Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    plt.imshow(real_img / 255.0)
    plt.show()

    plt.imshow(input_img / 255.0)
    plt.show()


if __name__ == "__main__":
    main()