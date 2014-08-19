# densitycontour

`densitycontour` -- draw density contours from sample points.


## USAGE

`densitycontour` is a Python module that helps with creating contour plots from
a sample of points.  It is useful for visualizing the output of Markov Chain
Monte Carlo (MCMC) sampling.

Typical usage is like follows:

```python
import pylab	# Import matplotlib environment.
import densitycontour

# Create scatter-data and rasterized image objects.
# x_array and y_array are "raw" inputs.
sample_data = densitycontour.ScatterData(x_array, y_array)

# Create a raster array for plotting, using default binning.
raster = sample_data.rasterize()

# Use the ZoomedContourVisualizer post-processor on the raster array.
contours = densitycontour.ZoomedContourVisualizer(raster, mode="nearest")

# Plot the contours for confidence levels 50% and 90% respectively,
# using default settings.
contours.plot((0.5, 0.9))

# Show the figure.
pylab.show()
```

The resulting figure should look like the image showed in one of the
following panels:

![Test output of densitycontour](densitycontour-test.png?raw=true "Test output of densitycontour")

You can run the module as a Python script to see the test diagrams.


## DEPENDENCY

`densitycontour` requires the [`numpy`][numpy], [`scipy`][scipy],
and [`matplotlib`][matplotlib] packages.


## COPYRIGHT

Copyright © 2014 Cong Ma.  License BSD: See the COPYING file.

This is free software: you are free to change and redistribute it.  
There is NO WARRANTY, to the extent permitted by law.


## AVAILABILITY

Available from [https://github.com/congma/densitycontour][githubrepo].


[githubrepo]: https://github.com/congma/densitycontour "GitHub repository page for densitycontour"
[numpy]: http://www.numpy.org/ "NumPy"
[scipy]: http://www.scipy.org/scipylib/index.html "SciPy library"
[matplotlib]: http://matplotlib.org/ "matplotlib"
