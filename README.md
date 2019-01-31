<h2>Machine Learning-Support Vector Machines</h2>
<h3>Description:</h3>
<ul style="list-style-type:disc">
<li>Python script to estimate Support Vector Machines for linear, polynomial and Gaussian kernels utilising quadratic programming optimisation algorithm from library CVXOPT.</li>
<li>Support Vector Machines implemented from scratch and compared to scikit-learn.</li>
<li>All estimations using simulated data.</li>
<li>Linear hard margin fits linearly separable data. Linear soft margin fits linearly separable data with some overlap in class examples. Both Gaussian and polynomial kernels estimate non-linearly separable examples.</li>
</ul>

<p float="left">
  <img src="/SupportVectorMachinesCustomLibraryLinearHardMargin.png" width="400" alt="Linear SVM hard margin. Using custom library."/>
  <img src="/SupportVectorMachinesSklearnLibraryLinearHardMargin.png" width="460"alt="Linear SVM hard margin. Using SKLearn."/>
  
  <img src="/SupportVectorMachinesCustomLibraryLinearSoftMargin.png" width="400" alt="Linear SVM soft margin."/>
  <img src="/SupportVectorMachinesSklearnLibraryLinearSoftMargin.png" width="460"alt="Linear SVM soft margin."/>
  
  <img src="/SupportVectorMachinesCustomLibraryPolynomial.png" width="400" alt="Polynomial SVM. Custom library."/>
  <img src="/SupportVectorMachinesSklearnLibraryPolynomialMargin.png" width="460"alt="Polynomial SVM. Using SKLearn."/>

  <img src="/SupportVectorMachinesCustomLibraryGaussianMargin.png" width="400" alt="Gaussian SVM. Custom library."/>
  <img src="/SupportVectorMachinesSklearnLibraryGaussianMargin.png" width="460"alt="Gaussian SVM. Using SKLearn."/>
</p>


 
 
<h3>Model</h3>
<p>Simulate labelled training data <img src="svgs/4388ea036963a2791929a7365e301c7a.svg" align=middle width=294.09701144999997pt height=27.91243950000002pt/> where there are N couples of <img src="svgs/45f2dbf90251d9796e488d974777c8c8.svg" align=middle width=48.491328599999996pt height=24.65753399999998pt/>, k is the number (dimension) of x variables.</p>
<br>
The Wolfe dual soft margin formula with kernel is given by

<p align="center"><img src="svgs/0acbd9783d20c53d1e9f750f2665520d.svg" align=middle width=333.89845664999996pt height=131.37932775pt/></p>

Where



<h4>Linear Kernel</h4>
<p align="center"><img src="svgs/3229be5d1587352e6b4dddbb725a2468.svg" align=middle width=114.9997167pt height=17.2895712pt/></p>
Where x and x' are two vectors.

<h4>Polynomial Kernel</h4>
<p align="center"><img src="svgs/130c930cf5becce140bc3628f8d6d787.svg" align=middle width=168.4659702pt height=18.88772655pt/></p>
Where C is a constant and d is the degree of the kernel.

<h4>Gaussian Kernel (aka Radial Basis Function (RBF)) </h4>
<p align="center"><img src="svgs/427b753bc670d106e67a2c8c5e77febf.svg" align=middle width=213.56621055pt height=21.1544223pt/></p>
<p>Where <img src="svgs/243cf87857232b4de4bc600c26d9d7cb.svg" align=middle width=22.20931349999999pt height=19.1781018pt/> is a free scalar parameter chosen based on the data and defines the influence of each training example.</p>


<h3>CVXOPT Library</h3>
The CVXOPT library solves the Wolfe dual soft margin constrained optimisation with the following API:
 
<p align="center"><img src="svgs/cda1046657bbb251aa30586f24569572.svg" align=middle width=418.849332pt height=78.26216475pt/></p>
<p>Note: <img src="svgs/ceddacf03a28d83100c38150c1076c1f.svg" align=middle width=12.785434199999989pt height=20.931464400000007pt/> indicates component-wise vector inequalities. It means that each row of the matrix <img src="svgs/b5087617bd5bed26b1da99fefb5353f1.svg" align=middle width=23.50114799999999pt height=22.465723500000017pt/> represents an inequality that must be satisfied.</p>
 
To use the CVXOPT convex solver API. The Wolfe dual soft margin formula is re-written as follows

<p align="center"><img src="svgs/a364906d0854671fe9b9718ce4ce1ec3.svg" align=middle width=212.12443724999997pt height=81.45851505pt/></p>

Where 
<br>
<p>G is a Gram matrix of all possible dot products of vectors <img src="svgs/d7084ce258ffe96f77e4f3647b250bbf.svg" align=middle width=17.521011749999992pt height=14.15524440000002pt/>.</p>

<p align="center"><img src="svgs/5ceca286e4d3c1cb407465d5db863df5.svg" align=middle width=357.85148685pt height=88.76800184999999pt/></p>

<p align="center"><img src="svgs/6a661323605601f1953ed25ec42f6807.svg" align=middle width=543.27321015pt height=148.99362225pt/></p>

 
<h3>How to use</h3>
<pre>
python supportVectorMachines.py
</pre>
		
		
<h3>Expected Output</h3>
<pre>
Estimating kernel: linear
     pcost       dcost       gap    pres   dres
 0: -1.4469e+01 -2.6709e+01  5e+02  2e+01  2e+00
 1: -1.6173e+01 -1.1054e+01  1e+02  6e+00  6e-01
 2: -2.1081e+01 -1.0259e+01  1e+02  4e+00  3e-01
 3: -6.7928e+00 -4.0658e+00  1e+01  4e-01  3e-02
 4: -3.1094e+00 -3.4085e+00  9e-01  2e-02  2e-03
 5: -3.2416e+00 -3.2970e+00  7e-02  4e-04  4e-05
 6: -3.2908e+00 -3.2914e+00  7e-04  4e-06  4e-07
 7: -3.2913e+00 -3.2913e+00  7e-06  4e-08  4e-09
 8: -3.2913e+00 -3.2913e+00  7e-08  4e-10  4e-11
 9: -3.2913e+00 -3.2913e+00  7e-10  4e-12  4e-13
10: -3.2913e+00 -3.2913e+00  7e-12  4e-14  7e-15
Optimal solution found.
============================================================
        SUPPORT VECTOR MACHINE TERMINATION RESULTS
============================================================
               **** In-Sample: ****
3 support vectors found from 160 examples.
               **** Predictions: ****
40 out of 40 predictions correct.
============================================================
Finished
</pre>

<h3>Highlights</h3>
<ul style="list-style-type:disc">
<li></li>
</ul>

<h3>Requirements</h3>
<p><a href="https://www.python.org/">Python (>2.7)</a>, <a href="http://www.numpy.org/">Numpy</a>, <a href="https://cvxopt.org/">CVXOPT</a>, <a href="https://scikit-learn.org">sklearn</a> and <a href="https://matplotlib.org/">matplotlib</a>.</p>
 



