attributes <- data.frame(
	x1=c(-4, 0, 4, 0, -1, 1, 1, -1),
	x2=c(0, 4, 0, -4, 1, 1, -1, -1)
)
labels <- c(0, 0, 0, 0, 1, 1, 1, 1)

# quadratic homogeneous kernel (degree 2, constant 0)
K <- new('PolynomialKernel', 2, 0)

# 2 classes, 2 dimensions
SVC <- new('OAOPolynomialKernel', 2, 2, K)

# C=1.0
SVC$train(attributes, labels, 1.0)

#contour plot
plus <- attributes[labels==1,]
minus <- attributes[labels==0,]

X <- seq(-5, 5, length=800)
Y <- seq(-5, 5, length=800)

filled.contour(X, Y, outer(X, Y, SVC$predict), asp=1,
	plot.title={
		title(main='Polynomial Kernel Demo', sub='Degree=2 Constant=0 C=1.0')
	},
	plot.axes={
		axis(1);
		axis(2);
		points(plus, pch='+', lwd=1.5, cex=2);
		points(minus, pch='-', lwd=1.5, cex=2);
	}
)



