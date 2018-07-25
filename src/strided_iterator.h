
#ifndef _STRIDED_ITERATOR_H_
#define _STRIDED_ITERATOR_H_

template<typename Iterator>
class strided_iterator : public Iterator {
public:
	explicit strided_iterator(Iterator _it, typename Iterator::difference_type _stride) : Iterator(_it), stride(_stride) {
	}
	strided_iterator& operator++() {
		Iterator::operator+=(stride);
		return *this;
	}
	strided_iterator operator++(int) {
		strided_iterator retval = *this;
		Iterator::operator+=(stride);
		return retval;
	}
	strided_iterator& operator--() {
		Iterator::operator-=(stride);
		return *this;
	}
	strided_iterator operator--(int) {
		strided_iterator retval = *this;
		Iterator::operator-=(stride);
		return retval;
	}
	strided_iterator& operator+=(typename Iterator::difference_type n) {
		Iterator::operator+=(stride*n);
		return *this;
	}
	strided_iterator operator+(typename Iterator::difference_type n) const {
		strided_iterator retval = *this;
		return retval += n;
	}
	strided_iterator& operator-=(typename Iterator::difference_type n) {
		Iterator::operator-=(stride*n);
		return *this;
	}
	strided_iterator operator-(typename Iterator::difference_type n) const {
		strided_iterator retval = *this;
		return retval -= n;
	}
	typename Iterator::difference_type operator-(const strided_iterator& other) const {
		return (static_cast<const Iterator&>(*this)-static_cast<const Iterator&>(other))/stride;
	}
	typename Iterator::reference operator[](typename Iterator::difference_type n) const {
		return Iterator::operator[](n*stride);
	}
private:
	typename Iterator::difference_type stride;
};

template<typename Iterable>
strided_iterator<typename Iterable::iterator> strided_begin(
	Iterable& x,
	typename Iterable::size_type offset,
	typename Iterable::size_type stride
) {
	return strided_iterator<typename Iterable::iterator>(x.begin() + offset, stride);
}
template<typename Iterable>
strided_iterator<typename Iterable::const_iterator> strided_cbegin(
	const Iterable& x,
	typename Iterable::size_type offset,
	typename Iterable::size_type stride
) {
	return strided_iterator<typename Iterable::const_iterator>(x.cbegin() + offset, stride);
}
/*
 strided_(c)end iterators need to calculate an offset from the true end of
 the iterable. This should be the first past-the-end position that the
 corresponding strided_(c)begin iterator will find.
 This is done so that algorithms that iterate using operator!=
 won't dereference any past-the-end iterator.
*/
template<typename Iterable>
strided_iterator<typename Iterable::iterator> strided_end(
	Iterable& x,
	typename Iterable::size_type offset,
	typename Iterable::size_type stride
) {
	return strided_begin(x, offset, stride) += (x.size() - offset + stride - 1)/stride;
}
template<typename Iterable>
strided_iterator<typename Iterable::const_iterator> strided_cend(
	const Iterable& x,
	typename Iterable::size_type offset,
	typename Iterable::size_type stride
) {
	return strided_cbegin(x, offset, stride) += (x.size() - offset + stride - 1)/stride;
}

template<typename Container>
class strided_range {
public:
	strided_range(Container& _container, typename Container::size_type _offset, typename Container::size_type _stride) : container(_container), offset(_offset), stride(_stride) {
	}
	strided_iterator<typename Container::iterator> begin() { return strided_begin(container, offset, stride); }
	strided_iterator<typename Container::iterator> end() { return strided_end(container, offset, stride); }
private:
	Container& container;
	typename Container::size_type offset;
	typename Container::size_type stride;
};
template<typename Container>
strided_range<Container> strided(Container& container, typename Container::size_type offset, typename Container::size_type stride) {
	return strided_range<Container>(container, offset, stride);
};

#endif
