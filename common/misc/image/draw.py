import holoviews as hv
import skimage

from . import histogram


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
        # intensity
        intensity = histogram.get_intensity(image)
        hist, nbins = skimage.exposure.histogram(intensity)
        values = list(zip(nbins, hist))
        chart = (chart + hv.Histogram(values, kdims='channel lightness')).cols(3)
    else:
        hist, nbins = skimage.exposure.histogram(image)
        values = list(zip(nbins, hist))
        chart = hv.Histogram(values)
    return chart


def draw_cdf(image, each_channel=True):
    if each_channel:
        hist, nbins = skimage.exposure.cumulative_distribution(image[:, :, 0])
        values = list(zip(nbins, hist))
        chart = hv.Curve(values, kdims='channel 0')
        for c in range(1, 3):
            hist, nbins = skimage.exposure.cumulative_distribution(image[:, :, c])
            values = list(zip(nbins, hist))
            chart += hv.Curve(values, kdims='channel {0}'.format(c))
        # intensity
        intensity = histogram.get_intensity(image)
        hist, nbins = skimage.exposure.cumulative_distribution(intensity)
        values = list(zip(nbins, hist))
        chart = (chart + hv.Curve(values, kdims='channel lightness')).cols(3)
    else:
        hist, nbins = skimage.exposure.cumulative_distribution(image)
        values = list(zip(nbins, hist))
        chart = hv.Curve(values)
    return chart


def draw_pdf(image, each_channel=True):
    if each_channel:
        density, nbins = histogram.get_pdf(image[:, :, 0])
        values = list(zip(nbins, density))
        chart = hv.Curve(values, kdims='channel 0')
        for c in range(1, 3):
            density, nbins = histogram.get_pdf(image[:, :, c])
            values = list(zip(nbins, density))
            chart += hv.Curve(values, kdims='channel {0}'.format(c))
        # intensity
        intensity = histogram.get_intensity(image)
        density, nbins = histogram.get_pdf(intensity)
        values = list(zip(nbins, density))
        chart = (chart + hv.Curve(values, kdims='channel lightness')).cols(3)
    else:
        density, nbins = histogram.get_pdf(image)
        values = list(zip(nbins, density))
        chart = hv.Curve(values)
    return chart
