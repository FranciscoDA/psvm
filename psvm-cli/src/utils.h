#pragma once

#include <string>
#include <numeric>

template<typename First, typename Last>
std::string join_str(const std::string& sep, First first, Last last) {
	std::string r = *first;
	while (++first != last)
		r.append(sep).append(*first);
	return r;
}
template<typename Iterable>
std::string join_str(const std::string& sep, const Iterable& iterable) {
	return join_str(sep, begin(iterable), end(iterable));
}
template<typename T>
T normalization_scale(T x, T scale_min, T scale_max, T x_min, T x_max) {
	if (x_min != x_max)
		return (x-x_min)/(x_max-x_min) * (scale_max-scale_min) + scale_min;
	return (scale_min + scale_max) / 2.;
}

template<typename T>
T normalization_standard(T x, T x_mean, T x_stdev) {
	if (x_stdev > 0.)
		return (x - x_mean) / x_stdev;
	return 0.;
}

