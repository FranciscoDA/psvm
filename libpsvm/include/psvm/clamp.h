#pragma once

#include <algorithm>
#if __cplusplus < 201703L
namespace std {
template <typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
	return v < lo ? lo : hi < v ? hi : v;
}
} // namespace std
#endif
