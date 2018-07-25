
#ifndef _REF_RANGE_H
#define _REF_RANGE_H_

template<typename T, typename... Args>
class ref_range {
private:
	std::array<T*, sizeof...(Args)+1> refs;
public:
	ref_range(T& arg1, Args&... args) : refs{&arg1, &args...} {
	}
	class iterator {
	public:
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using pointer = T*;
		using iterator_category = std::random_access_iterator_tag;

		iterator(T** _base) : base(_base) {}
		T& operator*() { return *(*base); }
		iterator& operator++() { ++base; return *this; }
		bool operator!=(const iterator& other) const { return base != other.base; }
	private:
		T** base;
	};
	auto begin() { return iterator(refs.begin()); }
	auto end() { return iterator(refs.end()); }
};
template<typename T, typename... Args>
ref_range<T, Args...> make_ref_range(T& arg1, Args&... args) {
	return ref_range<T, Args...>(arg1, args...);
}

#endif
