attributes <- data.frame(
	x1=c(-4, -3, -2, -1, 0, 1, 2, 3, 4, -2, 0, 2, -3, 3),
	x2=c(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, -1)
)
labels <- c(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0)

# 2 classes, 2 dimensions + RBF kernel (gamma 2.0)
SVC <- new('OneAgainstOneCSVC', 2, 2, new('RbfKernel', 2.0))

# C=2.0
SVC$train(attributes, labels, 2.0)

#contour plot
plus <- attributes[labels==1,]
minus <- attributes[labels==0,]

X <- seq(-5, 5, length=800)
Y <- seq(-5, 5, length=800)

filled.contour(X, Y, outer(X, Y, SVC$predict), asp=1,
	plot.title={
		title(main='RBF Kernel Demo', sub='Gamma=2.0 C=2.0');
	},
	plot.axes={
		axis(1);
		axis(2);
		points(plus, pch='+', lwd=1.5, cex=2);
		points(minus, pch='-', lwd=1.5, cex=2);
	}
)



