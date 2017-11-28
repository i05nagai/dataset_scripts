import holoviews as hv
import skimage


hv.extension('bokeh')


def draw_histogram(image, each_channel=True):
    if each_channel:
        hist, nbins = skimage.exposure.histogram(image[:, :, 0])
        values = list(zip(nbins, hist))
        chart = hv.Histogram(values, kdims='channel 0')
        for c in range(1, 3):
            hist, nbins = skimage.exposure.histogram(image[:, :, c])
            values = list(zip(nbins, hist))
            chart += hv.Histogram(values, kdims='channel {0}'.format(c))
        # brightness
        hist, nbins = skimage.exposure.histogram(image)
        values = list(zip(nbins, hist))
        chart = (chart + hv.Histogram(values, kdims='channel lightness')).cols(3)
    else:
        hist, nbins = skimage.exposure.histogram(image)
        values = list(zip(nbins, hist))
        chart = hv.Histogram(values)
    return chart
