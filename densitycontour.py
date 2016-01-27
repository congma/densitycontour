"""densitycontour -- draw density contours from sample points."""
# Disable Pylint false positives due to numpy/scipy's over-smart import
# pylint: disable=E1101
import weakref
import numpy
import scipy.optimize
import scipy.ndimage
from scipy.special import cbrt
import pylab


__all__ = ["ScatterData", "RasterizedData", "ContourVisualizerBase",
           "NaiveContourVisualizer", "ZoomedContourVisualizer",
           "MFZoomedContourVisualizer"]


def _getkeybysig(nbins_x, nbins_y, clipping, datarefx, datarefy):
    """Calculate raster interning key by the signature of its creation call."""
    if clipping is not None:
        # Unpack the range parameters and store as Python native type.
        ranges = tuple(float(x) for x in numpy.asarray(clipping).flat)
        cpv = clipping
    else:
        ranges = None
        cpv = (None, None)
    if nbins_x is None:
        nbx = _freedman_diaconis_nbins(datarefx, cpv[0])
    else:
        nbx = int(nbins_x)
    if nbins_y is None:
        nby = _freedman_diaconis_nbins(datarefy, cpv[1])
    else:
        nby = int(nbins_y)
    return (nbx, nby, ranges)


# pylint: disable=C0103
def _find_confidence_interval(x, pdf, confidence_level):
    """Function whose root, as the desired image-value level for
    the input confidence level, is to be found.
    """
    return pdf[pdf > x].sum() - confidence_level
# pylint: enable=C0103


def _destroy_dict(what, num, totally):
    """Destroy up to num dictionary items, unless totally = True, in which case
    the dictionary is cleared.  Items are destroyed in no particular order.

    Arguments
    ---------
    what: Dictionary whose items are to be destroyed.
    num: Integer, maximal (inclusive) number of items to destroy.  If not
         given as an integer, will be floor-rounded as one.
    totally: Boolean flat that, when specified, will cause the entire
             dictionary to be destroyed.

    Return value
    ------------
    The actual number of items destroyed.
    """
    assert num >= 0
    num = int(num)
    if totally:
        count = len(what)
        what.clear()
    else:
        count = 0
        for i in xrange(num):   # pylint: disable=W0612
            try:
                what.popitem()
            except KeyError:    # Nothing left
                break           # Exit the "for i in xrange()" loop
            count += 1
    return count


def _freedman_diaconis_nbins(data1d, clipping):
    """Calculate the number of bins for 1-d data according to
    Freedman-Diaconis's rule.
    """
    if clipping is not None:
        upper = numpy.max(clipping)
        lower = numpy.min(clipping)
        mask = (data1d >= lower) & (data1d <= upper)
        dataview = data1d[mask]
    else:
        dataview = data1d
    pwidth = numpy.percentile(dataview, 75) - numpy.percentile(dataview, 25)
    datarange = dataview.max() - dataview.min()
    return int(datarange * numpy.sqrt(numpy.sqrt((dataview.shape[0])))
               / 0.9 / pwidth)


class ScatterData(object):
    """Class holding reference to the scatter-point data."""
    def __init__(self, xdata, ydata):
        """Initialize with x- and y-data.  Both must be same-length
        (i.e. parallel) numpy arrays.
        """
        self.xdata = xdata
        self.ydata = ydata
        # Weak dict as memo table of rasterizations.
        self._rasters = weakref.WeakValueDictionary()
        return None

    def rasterize(self, nbins_x=None, nbins_y=None, clipping=None):
        """Rasterize self by binning the scattered points.

        Arguments
        --------
        nbins_x, nbins_y: Number of bins in the x- and y- dimensions.
                          If None, apply the Freedman-Diaconis rule.
        clipping: Valid ranges of data, points outside of which are considered
                  outliers and discarded in further processings.

        Return value
        ------------
        A RasterizedData instance, possibly a reference to an already-memoized
        one.
        """
        # Produce the memo key by normalizing the input parameters.
        memokey = _getkeybysig(nbins_x, nbins_y, clipping,
                               self.xdata, self.ydata)
        try:  # Consult the memo table first.
            result = self._rasters[memokey]
        except KeyError:  # Not yet in memo, so create it.
            result = RasterizedData(self, memokey[0], memokey[1],
                                    clipping=clipping)
            self._rasters[memokey] = result
        return result

    def forget_rasters(self, num=1, totally=False):
        """Forget memoized rasters in no particular order.

        Arguments
        ---------
        num: Maximal number of rasters to forget.  The actual number of rasters
        forgotten may be less than num.
        totally: If True, forget all rasters, ignoring num.

        Return value
        ------------
        The number of rasters actually forgotten.
        """
        return _destroy_dict(self._rasters, num, totally)


class RasterizedData(object):
    """Class holding the rasterized data."""
    def __init__(self, other, nbins_x, nbins_y, clipping=None):
        """Bin the data and create empirical PDF from other.

        Arguments
        ---------
        other: The ScatterData instance to which this raster instance will
               belong.
        nbins_x, nbins_y: Number of bins in the x- and y- dimensions.
        clipping: Valid ranges of data, points outside of which are considered
                  outliers and discarded in further processings.
        """
        hist, self.xedges, self.yedges = numpy.histogram2d(
            other.xdata, other.ydata,
            bins=(nbins_x, nbins_y),
            range=clipping, normed=True)
        x_bin_sizes = (self.xedges[1:] -
                       self.xedges[:-1]).reshape((1, nbins_x))
        y_bin_sizes = (self.yedges[1:] -
                       self.yedges[:-1]).reshape((nbins_y, 1))
        self.pdf = hist.T * (x_bin_sizes * y_bin_sizes)
        self.levels = {}
        return None

    def level_by_confidence(self, confidence_level):
        """Convert confidence to "image" level value.

        The result is memoized in self.levels during computation, if necessary.

        Arguments
        ---------
        confidence_level: the confidence level (0 <= p <= 1).

        Return value
        ------------
        The level value of given input.
        """
        assert 0.0 <= confidence_level <= 1.0
        # Use natively typed value as key.
        conf_native = float(confidence_level)
        try:
            result = self.levels[conf_native]
        except KeyError:
            result = scipy.optimize.brentq(_find_confidence_interval, 0., 1.,
                                           args=(self.pdf, confidence_level))
            self.levels[conf_native] = result
        return result

    def forget_levels(self, num=1, totally=False):
        """Forget up to num plotting levels, unless totally = True,
        in which case all levels are cleared.

        Arguments
        ---------
        num: Maximal number of rasters to forget.  The actual number of rasters
        forgotten may be less than num.
        totally: If True, forget all rasters, ignoring num.

        Return value
        ------------
        The actual levels forgotten.
        """
        return _destroy_dict(self.levels, num, totally)


# pylint: disable=R0903
class ContourVisualizerBase(object):
    """Base class of contour visualizer for rasterized data.

    Not meant to be instantiated directly.
    """
    def __init__(self, other, *args, **kwargs):  # pylint: disable=W0613
        """Initialize visualizer by a RasterizedData instance.

        Arguments
        ---------
        other: The RasterizedData instance for which visualization
               will be done.

        Subclasses will determine the use of *args and **kwargs.
        """
        self._xorig = 0.5 * (other.xedges[1:] + other.xedges[:-1])
        self._yorig = 0.5 * (other.yedges[1:] + other.yedges[:-1])
        self._zorig = other.pdf
        self.levelfinder = other.level_by_confidence
        return None

    def plot(self, confidence_levels, axes=None, filled=False,
             **contour_kwargs):
        """Plot as matplotlib contour.

        Arguments
        ---------
        confidence_levels: Container of confidence levels
        (0 <= p <= 1) for which the contours shall be drawn.
        axes: If given, will be used as the matplotlib "Axes"
              to which the contour will be added.
        Keyword arguments: Passed to matplotlib directly, except "levels".

        Return value
        ------------
        matplotlib's QuadContourSet object.
        """
        imglevels = [self.levelfinder(x) for x in confidence_levels]
        if axes is None:
            axes = pylab
        if filled:
            cmd = "contourf"
        else:
            cmd = "contour"
        contour = axes.__getattribute__(cmd)(self._xdat, self._ydat,
                                             self._zdat, levels=imglevels,
                                             origin="lower", **contour_kwargs)
        return contour


class NaiveContourVisualizer(ContourVisualizerBase):
    """The "naive" contour visualizer.

    This visualizer simply accpets the raster as it is, without any
    post-processing.

    """
    def __init__(self, other, *args, **kwargs):
        super(NaiveContourVisualizer, self).__init__(other, *args, **kwargs)
        self._xdat = self._xorig
        self._ydat = self._yorig
        self._zdat = self._zorig
        return None


class ZoomedContourVisualizer(ContourVisualizerBase):
    """This visualizer uses interpolated upsampling to provide
    better-shaped contours.
    """
    def __init__(self, other, zoom=3, zoom_prefilter=False, *args, **kwargs):
        """Initialize a ZoomedContourVisualizer instance.

        Additional arguments
        --------------------
        zoom: Zoom factor, default 3.
        zoom_prefilter: Boolean, whether the raster image should be processed
                        by the spline filter (scipy.ndimage.spline_filter)
                        before enlarging.  The filter is a high-pass filter
                        that is intended to enhance details, but may cause
                        undesirable artefacts in the processed image.  See
                        documentation of scipy.ndimage.zoom.
        *args are delegated to parent class.
        **kwargs are passed to scipy.ndimage.zoom.
        """
        super(ZoomedContourVisualizer,
              self).__init__(other, *args, **kwargs)
        self._xdat = scipy.ndimage.zoom(self._xorig, zoom,
                                        prefilter=zoom_prefilter, **kwargs)
        self._ydat = scipy.ndimage.zoom(self._yorig, zoom,
                                        prefilter=zoom_prefilter, **kwargs)
        self._zdat = scipy.ndimage.zoom(self._zorig, zoom,
                                        prefilter=zoom_prefilter, **kwargs)
        return None


class MFZoomedContourVisualizer(ZoomedContourVisualizer):
    """Upsampling visualizer with post-blurring using median filter."""
    def __init__(self, other, zoom=3, size=4, zoom_prefilter=True,
                 *args, **kwargs):
        """Initialize a MFZoomedContourVisualizer instance.

        Additional arguments
        --------------------
        size: The size of median filter, see documentation of
              scipy.ndimage.median_filter.  Default 4.
        *args are delegated to parent class.
        **kwargs are passed to scipy.ndimage.zoom.
        """
        super(MFZoomedContourVisualizer,
              self).__init__(other, zoom_prefilter=zoom_prefilter,
                             *args, **kwargs)
        self._zdat = scipy.ndimage.median_filter(self._zdat,
                                                 size=size, **kwargs)
        return None
# pylint: enable=R0903


def _test():
    """Module self-test."""
    data_array = numpy.random.multivariate_normal([0, 0],
                                                  [[1, 0], [0, 1]], 20000)
    data = ScatterData(data_array[:, 0], data_array[:, 1])
    raster = data.rasterize(32, 32)
    vis_backends = (NaiveContourVisualizer, ZoomedContourVisualizer,
                    MFZoomedContourVisualizer)
    titles = ("NaiveContourVisualizer", "ZoomedContourVisualizer",
              "MFZoomedContourVisualizer")
    # pylint: disable=W0612
    fig, axs = pylab.subplots(1, 3, sharex=True, sharey=True)
    titleft = {"fontsize": 10}
    # pylint: enable=W0612
    for i in xrange(len(vis_backends)):
        vbackend = vis_backends[i]
        sub = axs[i]
        sub.scatter(data.xdata[::4], data.ydata[::4],  # thinned by 4
                    alpha=0.1, facecolor="green",
                    edgecolor="green", marker=".", antialiased=True)
        cont = vbackend(raster, mode="nearest")
        cont.plot([0.9, 0.5], axes=sub, colors="black",
                  linestyles=["--", "-"], antialiased=True, alpha=0.8)
        sub.set_aspect("equal", adjustable="box-forced", anchor="C")
        sub.set_ylim((-3.5, 3.5))
        sub.set_xlim((-3.5, 3.5))
        sub.set_title(titles[i], fontdict=titleft)
    pylab.show()
    return None


if __name__ == "__main__":
    _test()
