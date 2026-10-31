#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}

__global__ void scale_probe(const float* previous, const float* current, float* result,
                            float multiplier, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    // Same expression and approximate exp2 used by the attention rescale.
    // Separate input arrays prevent compile-time cancellation of equal maxima.
    const float delta = previous[i] * multiplier - current[i] * multiplier;
    float scale;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(delta));
    result[i] = scale;
  }
}

int main() {
  constexpr unsigned count = 65536;
  std::vector<float> input(count), result(count);
  for (unsigned i = 0; i < count; ++i) input[i] = (float(i) + .5f) / 256.f - 128.f;
  float *previous, *current, *output;
  check(cudaMalloc(&previous, count * sizeof(float)));
  check(cudaMalloc(&current, count * sizeof(float)));
  check(cudaMalloc(&output, count * sizeof(float)));
  check(cudaMemcpy(previous, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  check(cudaMemcpy(current, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  const float multiplier = 0.0625f * 1.4426950408889634074f;
  scale_probe<<<count / 256, 256>>>(previous, current, output, multiplier, count);
  check(cudaGetLastError());
  check(cudaMemcpy(result.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost));
  unsigned nonunit = 0;
  float max_error = 0;
  for (float scale : result) {
    nonunit += scale != 1.f;
    max_error = std::max(max_error, std::fabs(scale - 1.f));
  }
  std::printf("{\"count\":%u,\"equal_max_count\":%u,\"nonunit_scale_count\":%u,"
              "\"max_abs_scale_error\":%.12g}\n", count, count, nonunit, double(max_error));
  check(cudaFree(previous));
  check(cudaFree(current));
  check(cudaFree(output));
}
