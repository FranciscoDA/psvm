
setClass('LinearKernel', representation(pointer='externalptr'))
setMethod('initialize', 'LinearKernel', function(.Object, ...) {
	.Object@pointer <- .LinearKernel__new(...)
	.Object
})
setMethod('names', 'LinearKernel', function(x) { c('get'); })
setMethod('$', 'LinearKernel', function(x, name) {
	if (name == 'get') {
		function(...) .Call(paste0('_psvm4r_LinearKernel__', name), x@pointer, ...)
	}
})

setClass('PolynomialKernel', representation(pointer='externalptr'))
setMethod('initialize', 'PolynomialKernel', function(.Object, ...) {
	.Object@pointer <- .PolynomialKernel__new(...)
	.Object
})
setMethod('names', 'PolynomialKernel', function(x) { c('get'); })
setMethod('$', 'PolynomialKernel', function(x, name) {
	if (name == 'get') {
		function(...) .Call(paste0('_psvm4r_PolynomialKernel__', name), x@pointer, ...)
	}
})

setClass('RbfKernel', representation(pointer='externalptr'))
setMethod('initialize', 'RbfKernel', function(.Object, ...) {
	.Object@pointer <- .RbfKernel__new(...)
	.Object
})
setMethod('names', 'RbfKernel', function(x) { c('get'); })
setMethod('$', 'RbfKernel', function(x, name) {
	if (name == 'get') {
		function(...) .Call(paste0('_psvm4r_RbfKernel__', name), x@pointer, ...)
	}
})

# convert a multidimensional entity (dataframe or matrix) into a 1d vector
.reorder <- function(attributes) {
	if (!is.null(dim(attributes)))
		as.numeric(t(attributes))
	else
		attributes
}

.makeSVC <- function(SVCT, KT) {
	.class <- paste0(SVCT, KT)
	setClass(.class, representation(pointer='externalptr'))
	setMethod('initialize', .class, function(.Object, num_classes, num_dimensions, kernel) {
		if (KT %in% class(kernel)) {
			.Object@pointer <- .Call(paste0('_psvm4r_', .class, '__new'), num_classes, num_dimensions, kernel@pointer)
			.Object
		}
		else {
			stop(paste('Incorrect kernel type', class(kernel), 'expected', KT))
		}
	})
	setMethod('show', .class, function(object) {
		cat(.class, ' (', object$num_dimensions, ' dimensions, ', object$num_classes, ' classes, ', sum(object$num_sv), ' total SVs)\n', sep='')
	})
	setMethod('print', .class, function(x) {
		show(x)
		invisible(x)
	})
	setMethod('names', .class, function(x) c('train', 'predict', 'num_sv', 'num_classes', 'num_dimensions'))
	setMethod('$', .class, function(x, name) {
		if (name == 'train') {
			function(attributes, labels, C) {
				.Call(paste0('_psvm4r_', .class, '__train'), x@pointer, .reorder(attributes), labels, C);
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
				.Call(paste0('_psvm4r_', .class, '__predict'), x@pointer, attributes)
			}
		}
		else if (name == 'num_sv') {
			.Call(paste0('_psvm4r_', .class, '__num_sv'), x@pointer)
		}
		else if (name == 'num_classes') {
			.Call(paste0('_psvm4r_', .class, '__num_classes'), x@pointer)
		}
		else if (name == 'num_dimensions') {
			.Call(paste0('_psvm4r_', .class, '__num_dimensions'), x@pointer)
		}
	})
}

.makeSVC('OAO', 'LinearKernel')
.makeSVC('OAO', 'PolynomialKernel')
.makeSVC('OAO', 'RbfKernel')
.makeSVC('OAA', 'LinearKernel')
.makeSVC('OAA', 'PolynomialKernel')
.makeSVC('OAA', 'RbfKernel')

