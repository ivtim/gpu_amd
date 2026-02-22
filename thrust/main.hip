#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>


struct square
{
    __host__ __device__ float operator()(float x) const
    {
        return x * x;
    }
};

int main()
{
    const size_t N = 1'000'000'000;
    const size_t bytes = N * sizeof(float);
    std::cout << bytes << std::endl;

    thrust::device_vector<float> d_vec(N, 2);
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), square());

    float result = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    std::cout << "Sum of squares = " << result << std::endl;

    return 0;
}