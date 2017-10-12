import cv2


def show_img(path):
    img = cv2.imread(path, 0)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    show_img('../example_keras/image/n07591961_paella/001be5fb5535fe7636b32faff7b81f06aec6ebc8.jpg')


if __name__ == '__main__':
    main()
