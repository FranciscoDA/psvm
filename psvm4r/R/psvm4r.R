
setClass('Kernel', representation(pointer='externalptr'))
setMethod('names', 'Kernel', function(x) {
	c('get')
})
setMethod('$', 'Kernel', function(x, name) {
	if (name == 'get') {
		function(...) .Kernel__get(x@pointer, ...)
	}
})

#' LinearKernel class.
setClass('LinearKernel', contains=c('Kernel'))

#' Create a LinearKernel. No parameters are necessary.
setMethod('initialize', 'LinearKernel', function(.Object, ...) {
	.Object@pointer <- .LinearKernel__new(...)
	.Object
})

#' PolynomialKernel class.
setClass('PolynomialKernel', contains=c('Kernel'))

#' Create a PolynomialKernel with the given parameters.
#'
#' @param degree degree of the exponent
#' @param constant additive constant
setMethod('initialize', 'PolynomialKernel', function(.Object, ...) {
	.Object@pointer <- .PolynomialKernel__new(...)
	.Object
})

#' RbfKernel class.
setClass('RbfKernel', contains=c('Kernel'))

#' Create a RbfKernel with the given parameter.
#'
#' @param gamma Parameter to control the radius. Note that gamma=1/(2*sigma^2) so a higher gamma yields a smaller radius
setMethod('initialize', 'RbfKernel', function(.Object, ...) {
	.Object@pointer <- .RbfKernel__new(...)
	.Object
})


# convert a multidimensional entity (dataframe or matrix) into a 1d vector
.reorder <- function(attributes) {
	if (!is.null(dim(attributes)))
		as.numeric(t(attributes))
	else
		attributes
}

setClass('CSVC', representation(pointer='externalptr'))
setMethod('show', 'CSVC', function(object) {
	cat(class(object), ' (', object$num_dimensions, ' dimensions, ', object$num_classes, ' classes, ', sum(object$num_sv), ' total SVs)\n', sep='')
})
setMethod('print', 'CSVC', function(x) {
	show(x)
	invisible(x)
})
setMethod('names', 'CSVC', function(x) c('train', 'predict', 'num_sv', 'num_classes', 'num_dimensions'))
setMethod('$', 'CSVC', function(x, name) {
	if (name == 'train') {
		function(attributes, labels, C) {
			.CSVC__train(x@pointer, .reorder(attributes), labels, C);
			invisible(x)
		}
	}
	else if (name == 'predict') {
		function(...) {
			if (...length() == 1)
				attributes <- .reorder(..1)
			else if (...length() > 1)
				attributes <- c(rbind(...))
			else
				stop('No attributes to predict on')
			.CSVC__predict(x@pointer, attributes)
		}
	}
	else if (name == 'num_sv') {
		.CSVC__num_sv(x@pointer)
	}
	else if (name == 'num_classes') {
		.CSVC__num_classes(x@pointer)
	}
	else if (name == 'num_dimensions') {
		.CSVC__num_dimensions(x@pointer)
	}
})

#' OneAgainstAllCSVC class.
setClass('OneAgainstAllCSVC', contains=c('CSVC'))

#' Create a One-against-all C-SVC with the given parameters
#'
#' @param num_classes number of distinct classes in the label vector of the problem
#' @param num_dimensions number of dimensions in the attributes table of the problem
#' @param kernel kernel object to use
setMethod('initialize', 'OneAgainstAllCSVC', function(.Object, num_classes, num_dimensions, kernel) {
	if (!inherits(kernel, 'Kernel'))
		stop('`kernel` argument is not a proper kernel object')
	.Object@pointer <- .OneAgainstAllCSVC__new(num_classes, num_dimensions, kernel@pointer)
	.Object
})

#' OneAgainstOneCSVC class.
setClass('OneAgainstOneCSVC', contains=c('CSVC'))

#' Create a One-against-one C-SVC with the given parameters
#'
#' @param num_classes number of distinct classes in the label vector of the problem
#' @param num_dimensions number of dimensions in the attributes table of the problem
#' @param kernel kernel object to use
setMethod('initialize', 'OneAgainstOneCSVC', function(.Object, num_classes, num_dimensions, kernel) {
	if (!inherits(kernel, 'Kernel'))
		stop('`kernel` argument is not a proper kernel object')
	.Object@pointer <- .OneAgainstOneCSVC__new(num_classes, num_dimensions, kernel@pointer)
	.Object
})
