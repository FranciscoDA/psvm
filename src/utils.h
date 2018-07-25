
#ifndef _UTILS_H_
#define _UTILS_H_

#include <string>
#include <numeric>

template<typename InputIterator>
string join_str(const string& sep, InputIterator first, InputIterator last) {
	string r = *first;
	while (++first != last)
		r.append(sep).append(*first);
	return r;
}
template<typename Iterable>
string join_str(const string& sep, const Iterable& iterable) {
	return join_str(sep, begin(iterable), end(iterable));
}

template<typename InputIterator>
typename InputIterator::value_type mean(InputIterator first, InputIterator last) {
	auto d = std::distance(first, last);
	if (d != 0)
		return std::accumulate(first, last, 0.) / static_cast<typename InputIterator::value_type>(d);
	return 0.;
}

template<typename InputIterator>
typename InputIterator::value_type stdev(InputIterator first, InputIterator last, typename InputIterator::value_type m) {
	auto d = std::distance(first, last);
	if (d == 0)
		return 0.;
	typename InputIterator::value_type esumsq = 0.;
	for (; first != last; ++first)
		esumsq += (*first-m) * (*first-m) / static_cast<typename InputIterator::value_type>(d);
	return std::sqrt(esumsq);
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

#endif
