
#ifndef _NORMALIZATION_H_
#define _NORMALIZATION_H_

template<typename T>
void normalization_standard(
	std::vector<T>& x,
	const std::vector<T>& attribute_mean, const std::vector<T>& attribute_stdev
) {
	const int num_attributes = attribute_mean.size();
	const int num_samples = x.size()/num_attributes;
	for (int j = 0; j < num_attributes; ++j) {
		for (int i = 0; i < num_samples; ++i) {
			T& v = x[i*num_attributes+j];
			v = attribute_stdev[j] == 0. ? 0. : (v - attribute_mean[j]) / attribute_stdev[j];
		}
	}
}

template<typename T>
void normalization_scale(
	std::vector<T>& x,
	const std::vector<T>& attribute_min, const std::vector<T>& attribute_max,
	T a=0., T b=1.
) {
	const int num_attributes = attribute_min.size();
	const int num_samples = x.size() / num_attributes;
	T expected = (a+b)/2.;
	for (int j = 0; j < num_attributes; ++j) {
		T range = attribute_max[j] - attribute_min[j];
		for (int i = 0; i < num_samples; ++i) {
			T& v = x[i*num_attributes+j];
			v = range==0. ? expected : a + (v - attribute_min[j]) * (b - a) / range;
		}
	}
}

#endif
