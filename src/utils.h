
#ifndef _UTILS_H_
#define _UTILS_H_

#include <string>

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

template<typename Iterator>
class stride_iterator {
public:
	using iterator_category = std::random_access_iterator_tag;
	using value_type = typename Iterator::value_type;
	using difference_type = typename Iterator::difference_type;
	using pointer = typename Iterator::pointer;
	using reference = typename Iterator::reference;

	explicit stride_iterator(Iterator _it, std::size_t _stride) : it(_it), stride(_stride) {
	}
	stride_iterator& operator++() {
		it += stride;
		return *this;
	}
	stride_iterator operator++(int) {
		auto retval = *this;
		it += stride;
		return retval;
	}
	stride_iterator operator+(long j) const {
		auto retval = *this;
		retval.it += j * retval.stride;
		return retval;
	}
	stride_iterator operator-(long j) const {
		auto retval = *this;
		retval.it -= j * retval.stride;
		return retval;
	}
	difference_type operator-(const stride_iterator& other) const {
		return (it - other.it)/stride;
	}
	bool operator==(const stride_iterator& other) const {
		return it - other.it == 0;
	}
	bool operator!=(const stride_iterator& other) const {
		return it - other.it != 0;
	}
	bool operator<(const stride_iterator& other) const {
		return it - other.it < 0;
	}
	bool operator<=(const stride_iterator& other) const {
		return it - other.it <= 0;
	}
	typename Iterator::reference operator*() const {
		return *it;
	}
private:
	Iterator it;
	std::size_t stride;
};

template<typename Iterable>
stride_iterator<typename Iterable::iterator> strided_begin(Iterable& x, std::size_t offset, std::size_t stride) {
	return stride_iterator<typename Iterable::iterator>(x.begin() + offset, stride);
}
template<typename Iterable>
stride_iterator<typename Iterable::iterator> strided_end(Iterable& x, std::size_t offset, std::size_t stride) {
	auto end_offset = ((x.size()+stride-1)/stride * stride + offset) % stride;
	return stride_iterator<typename Iterable::iterator>(x.end() + end_offset, stride);
}

template<typename Iterable>
stride_iterator<typename Iterable::const_iterator> strided_cbegin(const Iterable& x, std::size_t offset, std::size_t stride) {
	return stride_iterator<typename Iterable::const_iterator>(x.cbegin() + offset, stride);
}
template<typename Iterable>
stride_iterator<typename Iterable::const_iterator> strided_cend(const Iterable& x, std::size_t offset, std::size_t stride) {
	auto end_offset = ((x.size()+stride-1)/stride * stride + offset) % stride;
	return stride_iterator<typename Iterable::const_iterator>(x.cend() + end_offset, stride);
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
