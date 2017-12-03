from .. import util


def _each_pixel(top, bottom, func):
    # has color channel
    if len(bottom.shape) == 3:
        for r in range(bottom.shape[0]):
            for c in range(bottom.shape[1]):
                for channel in range(bottom.shape[2]):
                    bottom[r, c, channel] = func(top[r, c, channel], bottom[r, c, channel])
    else:
        for r in range(top.shape[0]):
            for c in range(top.shape[1]):
                bottom[r, c] = func(top[r, c], bottom[r, c])
    return bottom


def _each_pixel_single_color_top(top, bottom, func):
    # has color channel
    if len(bottom.shape) == 3:
        for channel in range(bottom.shape[2]):
            bottom[:, :, channel] = func(top[channel], bottom[:, :, channel])
    else:
        for r in range(top.shape[0]):
            for c in range(top.shape[1]):
                bottom[r, c] = func(top, bottom[r, c])
    return bottom


def _multiply(top, bottom):
    bottom = util.copy(bottom, dtype='uint16')
    # ((t/255.0) * (b/255.0)) * 255.0
    bottom = top * bottom // 255
    return bottom


def _multiply_single_color_top(top, bottom):
    bottom = util.copy(bottom, dtype='uint16')
    # ((t/255.0) * (b/255.0)) * 255.0
    for channel in range(bottom.shape[2]):
        bottom[:, :, channel] = top[channel] * bottom[:, :, channel] // 255
    return bottom


def _screen(top, bottom):
    # (1 - (1 - t/255)(1 - b/255)) 255
    return 255 - (255 - top) * (255 - bottom)


def blending(top, bottom, blend_type):
    if blend_type == 'normal':
        return top
    elif blend_type == 'dissolve':
        # TODO
        return bottom
    elif blend_type == 'multiply':
        print('top: {0}'.format(top))
        # if top is single color image
        # gray scale
        if len(top) == 1 and len(bottom.shape) == 2:
            # TODO
            return bottom
        # RGB
        elif len(top) == bottom.shape[2]:
            return _multiply_single_color_top(top, bottom)
        else:
            return _multiply(top, bottom)
    elif blend_type == 'screen':
        # TODO
        return _each_pixel(top, bottom, _screen)
    else:
        # TODO
        return top
