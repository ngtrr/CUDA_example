/*
 * Noise Reduction with cuFFT
 * 
 * 
 */
#include <cuda_runtime.h>
#include <cufft.h>

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

using namespace std;

int main() {

  const size_t N = 44100U; // データ数(44.1kHzサンプリングでの1秒分)

  vector<float> signal(N);
  vector<float> h_in(N);

  // 振幅 ±2 のホワイト・ノイズ
  mt19937 mt;
  uniform_real_distribution<float> rnd(-2.0f, 2.0f);

  // 350,400,450Hzのサイン波にノイズを乗せる
  float omega = 2.0f * 3.1416f / N;
  for ( unsigned int i = 0; i < N; ++i ) {
    signal[i] = 
       sinf(omega * 350.0f * (float)i) * 1.0f +
       sinf(omega * 400.0f * (float)i) * 0.8f +
       sinf(omega * 450.0f * (float)i) * 0.6f ;
    h_in[i]   = signal[i] + rnd(mt);
  }

  // device-memoryの確保(入/出力兼用)
  float* d_real = nullptr;
  cudaMalloc(&d_real, N*sizeof(float));
  float2* d_cplx = reinterpret_cast<float2*>(d_real);

  // フーリエ変換
  cudaMemcpy(d_real, h_in.data(), N*sizeof(float), cudaMemcpyHostToDevice);

  cufftHandle plan_f;
  cufftPlan1d(&plan_f, N, CUFFT_R2C, 1); // Real to Complex (forward)
  cufftExecR2C(plan_f, d_real, d_cplx);

  vector<float2> h_mid(N/2); // スペクトル(フーリエ変換の結果)
  cudaMemcpy(h_mid.data(), d_cplx, N*sizeof(float), cudaMemcpyDeviceToHost);

  // band-pass filter
  // 300Hz以下/500Hz以上の信号をカットする
  cudaMemset(d_cplx     , 0,      300U  * sizeof(float2));
  cudaMemset(d_cplx+500U, 0, (N/2-500U) * sizeof(float2));

  // 逆フーリエ変換
  cufftHandle plan_i;
  cufftPlan1d(&plan_i, N, CUFFT_C2R, 1); // Complex to Real (inverse)
  cufftExecC2R(plan_i, d_cplx, d_real);

  // 結果の出力
  vector<float>  h_out(N);
  cudaMemcpy(h_out.data(), d_real, N*sizeof(float), cudaMemcpyDeviceToHost);

  cout << "signal, noised, processed, spectrum" << endl;
  for ( unsigned int i = 0; i < 500; ++i ) {
    cout << signal[i] << ',' 
         << h_in[i] << ',' 
         << h_out[i]/N << ',' 
         << cuCabsf(h_mid[i]) << endl;
  } 

  cudaFree(d_real);
  cufftDestroy(plan_f);
  cufftDestroy(plan_i);

}
