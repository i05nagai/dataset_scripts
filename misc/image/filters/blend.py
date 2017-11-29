

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
        for r in range(bottom.shape[0]):
            for c in range(bottom.shape[1]):
                for channel in range(bottom.shape[2]):
                    bottom[r, c, channel] = func(top[channel], bottom[r, c, channel])
    else:
        for r in range(top.shape[0]):
            for c in range(top.shape[1]):
                bottom[r, c] = func(top, bottom[r, c])
    return bottom


def _multiply(top, bottom):
    # ((t/255.0) * (b/255.0)) * 255.0
    return int((top * bottom) / 255.0)


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
        # if top is single color image
        # gray scale
        if len(top) == 1 and len(bottom.shape) == 2:
            # TODO
            return bottom
        # RGB
        elif len(top) == len(bottom.shape[2]):
            return _each_pixel_single_color_top(top, bottom, _multiply)
        else:
            return _each_pixel(top, bottom, _multiply)
    elif blend_type == 'screen':
        # TODO
        return _each_pixel(top, bottom, _screen)
    else:
        # TODO
        return top
