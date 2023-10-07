# Surface Curvature in Python

Several implementations of calculating mean, gaussian, and normal surface curvature along with principal curvature and respective directions.

The functions allow the surface to be defined either symbolically (with sympy) or discretely.
You can of course take a discrete surface and fit a polynomial to the data to create a symbolic expression of the surface. Or vice-vera, by taking a symbolic function and picking out a set of data points along the surface to use for discrete evaluation.

The data can be either explicit `z = f(x,y)` or parametric `(u,v) --> < x(uv), y(u,v), z(u,v) >`
There is yet another way to define a surface, implicitly `F(x,y,z) = 0`, but I have not created the functions for it yet.

These are most likely not the fastest or optimized ways to do the calculations, instead, they are meant to be a medium to learn differential geometry and how the formulas are implemented into computer code.

![examples/DiffGeoOps.ipynb](docs/gaussian_torus.png)

![examples/discrete_shape.ipynb](docs/image.png)

- [x] Symbolic
  - [x] Explicit
    - [x] Mean
    - [x] Gaussian
    - [x] Principal
    - [x] Principal vectors
  - [x] Parametric
    - [x] Mean
    - [x] Gaussian
    - [x] Principal
    - [x] Principal vectors
- [x] Discrete
  - [x] Explicit orthogonal (monge patch)
    - [x] Mean
    - [x] Gaussian
    - [x] Principal
    - [x] Principal vectors
  - [x] Parametric
    - [x] Mean
    - [x] Gaussian
    - [x] Principal
    - [x] Principal vectors
  - [ ] Arbitrary Mesh
    - [ ] [Mark Meyer, Mathieu Desbrun, Peter Schröder and Alan H. Barr. Discrete Differential-Geometry Operators for Triangulated 2-Manifolds. Caltech, USC, 2002.](http://www.multires.caltech.edu/pubs/diffGeoOps.pdf)
      - see `example/DiffGeoOps.ipynb`
    - [ ] [Rusinkiewicz, Szymon. Estimating Curvatures and Their Derivatives on Triangle Meshes. Princeton University, 2004. ](https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2004_ECA/curvpaper.pdf)
    - [ ] [D. Panozzo, E. Puppo, L. Rocca. Efficient Multi-scale Curvature and Crease Estimation. DISI - Universita di Genova, 2010. (Using a localized fitting of surface to each vertex normal for computing curvature. polynomial regression, quadratic fitting or quadratic interpolation on the nearby vertices around the point to calculate principal curvatures.)](https://cims.nyu.edu/gcl/papers/GraVisMa10-PanozzoPuppoRocca.pdf)

## Testing

```
python -m unittest discover .\tests\
```

## References:

### Definitions

- https://en.wikipedia.org/wiki/Differential_geometry_of_surfaces
- https://en.wikipedia.org/wiki/Curvature
- https://en.wikipedia.org/wiki/Principal_curvature
- https://en.wikipedia.org/wiki/Mean_curvature
- https://en.wikipedia.org/wiki/Gaussian_curvature
- https://en.wikipedia.org/wiki/Parametric_surface#Curvature
- https://e.math.cornell.edu/people/belk/differentialgeometry/Outline%20-%20The%20Gauss%20Map.pdf

### References

- Keenan Crane (see [KeenanCraneLect15.ipynb](examples/KeenanCraneLect15.ipynb))
  - https://brickisland.net/DDGFall2017/wp-content/uploads/2017/10/CMU_DDG_Fall2017_08_Surfaces.pdf
  - https://www.youtube.com/watch?v=e-erMrqBd1w
  - http://wordpress.discretization.de/geometryprocessingandapplicationsws19/a-quick-and-dirty-introduction-to-the-curvature-of-surfaces/
- https://mathworld.wolfram.com/MeanCurvature.html
- https://liavas.net/courses/math430/files/Surfaces.pdf
- https://jhavaldar.github.io/assets/2017-07-16-diffgeo-notes5.pdf
- https://github.com/sujithTSR/surface-curvature/blob/master/surface.py
- https://www.mathworks.com/matlabcentral/fileexchange/11168-surface-curvature
- `Curvedness = sqrt((k1**2 + k2**2)/2)`
  - https://www.researchgate.net/publication/324908170_Description_and_Retrieval_of_Geometric_Patterns_on_Surface_Meshes_using_an_edge-based_LBP_approach
- https://machinelearningmastery.com/a-gentle-introduction-to-the-laplacian/

### Discrete Algorithms:

- 2004 Caltech Paper over Discrete Surface Curvature approach

  - https://thesis.library.caltech.edu/2186/1/phd.pdf
  - http://www.multires.caltech.edu/pubs/diffGeoOps.pdf
  - https://github.com/justachetan/DiffGeoOps/blob/master/DiffGeoOps.py

- 2004 Princeton Paper for "Estimating Curvatures and Their Derivatives on Triangle Meshes"

  - https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2004_ECA/curvpaper.pdf
  - https://stackoverflow.com/questions/14234127/how-to-get-principal-curvature-of-a-given-mesh
  - https://github.com/Forceflow/trimesh2/blob/main/libsrc/TriMesh_curvature.cc

- Keenan Crane

  - https://www.cs.cmu.edu/~kmcrane/Projects/DGPDEC/paper.pdf
  -

- 2010 Panozzo - Quadratic Fitting

  - https://cims.nyu.edu/gcl/papers/GraVisMa10-PanozzoPuppoRocca.pdf
  - https://github.com/libigl/libigl/blob/main/include/igl/principal_curvature.cpp
  - https://libigl.github.io/libigl-python-bindings/tut-chapter1/

- 2021

  - https://hal.science/hal-03272493/document
  - https://github.com/STORM-IRIT/algebraic-shape-operator/tree/main/examples

- https://stackoverflow.com/a/14234542
- https://github.com/alecjacobson/geometry-processing-curvature (See 2010 Panozzo)
- https://github.com/cuge1995/curvature-calculation-python/blob/main/plcurvature.py
- https://blender.stackexchange.com/questions/146819/is-there-a-way-to-calculate-mean-curvature-of-a-triangular-mesh/147371#147371
- https://www.cs.purdue.edu/homes/cs53100/slides/geom.pdf
- https://jhavaldar.github.io/assets/2017-07-16-diffgeo-notes5.pdf
- https://github.com/pmp-library/pmp-library/blob/main/src/pmp/algorithms/curvature.cpp
- https://github.com/AbhilashReddyM/curvpack
- https://booksite.elsevier.com/samplechapters/9780120887354/9780120887354.PDF
-
