attributes <- data.frame(
	x1=c(-4, -3, -2, -2, -1, 1, -1, 1, 2, 3, 3),
	x2=c(-4, -1, -2, 2, -1, -4, 4, 1, -3, -1, 2)
)
labels <- c(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1)

# 2 classes, 2 dimensions + linear kernel
SVC <- new('OneAgainstOneCSVC', 2, 2, new('LinearKernel'))

# C=1.0
SVC$train(attributes, labels, 1.0)

#contour plot
plus <- attributes[labels==1,]
minus <- attributes[labels==0,]

X <- seq(-5, 5, length=800)
Y <- seq(-5, 5, length=800)

filled.contour(X, Y, outer(X, Y, SVC$predict), asp=1,
	plot.title={
		title(main='Linear Kernel Demo', sub='C=1.0')
	},
	plot.axes={
		axis(1);
		axis(2);
		points(plus, pch='+', lwd=1.5, cex=2);
		points(minus, pch='-', lwd=1.5, cex=2);
	}
)
