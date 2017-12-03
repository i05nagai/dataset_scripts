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
    return util.to_valid_image(bottom)


def _screen(top, bottom):
    # (1 - (1 - t/255)(1 - b/255)) 255
    return 255 - (255 - top) * (255 - bottom)


def _overlay(top, bottom):
    # 0 <= a <= 1, 0 <= b <= 1
    # 2ab if a < 0.5
    # 1 - 2(1 - a)(1 - b) otherwise
    ind = top < 0.5 * 255
    bottom[ind] = 2.0 * top[ind] * bottom[ind] // 255
    ind = top >= 0.5 * 255
    bottom[ind] = 255.0 - 2.0 * (255 - top[ind]) * (255 - bottom[ind])
    return bottom


def _overlay_single_color(top, bottom):
    # 0 <= a <= 1, 0 <= b <= 1
    # 2ab if a < 0.5
    # 1 - 2(1 - a)(1 - b) otherwise
    ind = top < 0.5 * 255
    bottom[:, :, ind] = 2.0 * top[ind] * bottom[:, :, ind] // 255
    ind = top >= 0.5 * 255
    bottom[:, :, ind] = (255.0
                         - 2.0 * (255 - top[ind]) * (255 - bottom[:, :, ind]))
    return bottom


def _overlay_alpha(top, bottom):
    # 0 <= a <= 1, 0 <= b <= 1
    alpha_top = top[:, :, 3] / 255.0
    alpha_bottom = bottom[:, :, 3] / 255.0
    alpha_out = (alpha_top * 1.0
                 + alpha_bottom * (1.0 - alpha_top))
    for c in range(0, 3):
        bottom[:, :, c] = ((top[:, :, c] * alpha_top * 1.0
                            + bottom[:, :, c] * alpha_bottom * (1.0 - alpha_top))
                           / alpha_out)
    bottom[:, :, 3] = alpha_out * 255.0
    return bottom


def _overlay_alpha_single_color(top, bottom):
    # top = (r, g, b, a)
    # 0 <= a <= 1, 0 <= b <= 1
    # 2ab if a < 0.5
    # 1 - 2(1 - a)(1 - b) otherwise
    alpha_top = top[3]
    alpha_bottom = bottom[:, :, 3]
    alpha_out = (alpha_top * 1.0
                 + alpha_bottom * (1.0 - alpha_top))
    bottom[:, :, 0:3] = ((top[0:3] * alpha_top * 1.0
                          + bottom[:, :, 0:3] * alpha_bottom * (1.0 - alpha_top))
                         / alpha_out)
    return bottom


def overlay(top, bottom):
    # grayscale
    if len(top) == 1 and len(bottom.shape) == 2:
        # TODO
        return bottom
    # RGB/RGBA but top is single color image
    elif len(top) == bottom.shape[2]:
        # RGB
        if len(top) == 3:
            return _overlay_single_color(top, bottom)
        # RGBA
        elif len(top) == 4:
            bottom = util.rgb_to_rgba(bottom)
            return _overlay_alpha_single_color(top, bottom)
    else:
        # RGB
        if top.shape[2] == 3:
            return _overlay_single_color(top, bottom)
        # RGBA
        elif top.shape[2] == 4:
            bottom = util.rgb_to_rgba(bottom)
            return _overlay_alpha(top, bottom)
        return _overlay(top, bottom)


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
    elif blend_type == 'overlay':
        return overlay(top, bottom)
    else:
        # TODO
        return top
