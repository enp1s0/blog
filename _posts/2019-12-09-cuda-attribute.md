---
layout: post
title:  "cudaGetDevicePropertiesとcudaDeviceGetAttribute"
date:   2019-12-09 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2>CUDAのデバイスのステータスの取得</h2>
<p>
Cooperative Groupsで全スレッド同期が可能なスレッド数の計算やランタイムでのCompute Capabilityの取得などGPU固有の情報を取得することは多々あると思います．<br>
CUDAではデバイスの情報を取得する関数としてcudaGetDevicePropertiesとcudaDeviceGetAttributeが用意されています．<br>
それぞれの関数の違いとしては，cudaGetDevicePropertiesがデバイスの情報すべてを構造体に詰めて返してくれるのに対し，cudaDeviceGetAttributeは欲しい情報をピンポイントで取りに行くところが挙げられます．<br>
（一方の関数でしかとれない情報もあります）<br>
で，NVIDIA Developer Blogに<a href="https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/">こんなエントリー</a>がありました．</a>
要約すると，cudaGetDevicePropertiesで取得される一部の情報は取得に時間がかかるので，この情報が必要ない場合はcudaDeviceGetAttributeでピンポイントで取得したほうがいいよというもの．<br>
遅いものというのはcudaDevAttrClockRate, cudaDevAttrKernelExecTimeout, cudaDevAttrMemoryClockRate, cudaDevAttrSingleToDoublePrecisionPerfRatioだそうです．<br>
ではこれらの情報の取得だけが極端に遅いだけなのかを調べてみましたというのがこの記事です．<br>
</p>
<h2>時間のかかるAttribute Top 10</h2>
<p>
取得したAttributeは<a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49e2f8c2c0bd6fe264f2fc970912e5cd">enum cudaDeviceAttr</a>のもの全てで，廃止されたものや予約されているが空のものも含めて全てです．
<pre class="code-line">$ ./get_attr_time | sort -k 3 -n -r | head -n 10
                        cudaDevAttrMemoryClockRate :  144088.634 [ns]
                              cudaDevAttrClockRate :  143006.136 [ns]
                     cudaDevAttrMaxThreadsPerBlock :  56730.388 [ns]
                      cudaDevAttrKernelExecTimeout :  28354.916 [ns]
              cudaDevAttrHostNativeAtomicSupported :  48.030 [ns]
           cudaDevAttrCooperativeMultiDeviceLaunch :  24.925 [ns]
                      cudaDevAttrCooperativeLaunch :  24.812 [ns]
             cudaDevAttrComputePreemptionSupported :  21.849 [ns]
              cudaDevAttrMaxSurface2DLayeredHeight :  21.642 [ns]
                   cudaDevAttrMaxRegistersPerBlock :  20.199 [ns]
</pre>
やはり挙げられていた4つが極端に遅いようです．<br>
ただ，上位9つは何度プログラムを実行しても入れ替わらず，それ以降は結果が変わります．<br>
やはり極端に遅いわけではないが遅めのものもあるようです．<br>
予約されているだけで使われていなさそうなcudaDevAttrReserved92/3/4は17[ns]くらいなので，20[ns]程度の時間がかかっているAttributeの順番は意味がなさそうです．<br>

<h3>使用したプログラムのソースコード</h3>

{% highlight cuda %}
#include <iostream>
#include <chrono>

#define PRINT_ELAPSED_TIME(attr) std::printf("%50s : % 7.3f [ns]\n", #attr, get_elapsed_time(attr))

auto get_elapsed_time(cudaDeviceAttr attribute) {
  constexpr std::size_t N = 10000;

  int prop;
  const auto start_clock = std::chrono::high_resolution_clock::now();
  for (std::size_t i = 0; i < N; i++) {
    cudaDeviceGetAttribute(&prop, attribute, 0);
  }
  const auto end_clock = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() / static_cast<double>(N);
}

int main() {
  PRINT_ELAPSED_TIME(cudaDevAttrMaxThreadsPerBlock);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxBlockDimX);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxBlockDimY);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxBlockDimZ);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxGridDimX);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxGridDimY);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxGridDimZ);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSharedMemoryPerBlock);
  PRINT_ELAPSED_TIME(cudaDevAttrTotalConstantMemory);
  PRINT_ELAPSED_TIME(cudaDevAttrWarpSize);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxPitch);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxRegistersPerBlock);
  PRINT_ELAPSED_TIME(cudaDevAttrClockRate);
  PRINT_ELAPSED_TIME(cudaDevAttrTextureAlignment);
  PRINT_ELAPSED_TIME(cudaDevAttrGpuOverlap);
  PRINT_ELAPSED_TIME(cudaDevAttrMultiProcessorCount);
  PRINT_ELAPSED_TIME(cudaDevAttrKernelExecTimeout);
  PRINT_ELAPSED_TIME(cudaDevAttrIntegrated);
  PRINT_ELAPSED_TIME(cudaDevAttrCanMapHostMemory);
  PRINT_ELAPSED_TIME(cudaDevAttrComputeMode);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture1DWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DHeight);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture3DWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture3DHeight);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture3DDepth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DLayeredWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DLayeredHeight);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DLayeredLayers);
  PRINT_ELAPSED_TIME(cudaDevAttrSurfaceAlignment);
  PRINT_ELAPSED_TIME(cudaDevAttrConcurrentKernels);
  PRINT_ELAPSED_TIME(cudaDevAttrEccEnabled);
  PRINT_ELAPSED_TIME(cudaDevAttrPciBusId);
  PRINT_ELAPSED_TIME(cudaDevAttrPciDeviceId);
  PRINT_ELAPSED_TIME(cudaDevAttrTccDriver);
  PRINT_ELAPSED_TIME(cudaDevAttrMemoryClockRate);
  PRINT_ELAPSED_TIME(cudaDevAttrGlobalMemoryBusWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrL2CacheSize);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxThreadsPerMultiProcessor);
  PRINT_ELAPSED_TIME(cudaDevAttrAsyncEngineCount);
  PRINT_ELAPSED_TIME(cudaDevAttrUnifiedAddressing);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture1DLayeredWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture1DLayeredLayers);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DGatherWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DGatherHeight);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture3DWidthAlt);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture3DHeightAlt);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture3DDepthAlt);
  PRINT_ELAPSED_TIME(cudaDevAttrPciDomainId);
  PRINT_ELAPSED_TIME(cudaDevAttrTexturePitchAlignment);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTextureCubemapWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTextureCubemapLayeredWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTextureCubemapLayeredLayers);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface1DWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface2DWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface2DHeight);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface3DWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface3DHeight);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface3DDepth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface1DLayeredWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface1DLayeredLayers);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface2DLayeredWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface2DLayeredHeight);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurface2DLayeredLayers);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurfaceCubemapWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurfaceCubemapLayeredWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSurfaceCubemapLayeredLayers);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture1DLinearWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DLinearWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DLinearHeight);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DLinearPitch);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DMipmappedWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture2DMipmappedHeight);
  PRINT_ELAPSED_TIME(cudaDevAttrComputeCapabilityMajor);
  PRINT_ELAPSED_TIME(cudaDevAttrComputeCapabilityMinor);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxTexture1DMipmappedWidth);
  PRINT_ELAPSED_TIME(cudaDevAttrStreamPrioritiesSupported);
  PRINT_ELAPSED_TIME(cudaDevAttrGlobalL1CacheSupported);
  PRINT_ELAPSED_TIME(cudaDevAttrLocalL1CacheSupported);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSharedMemoryPerMultiprocessor);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxRegistersPerMultiprocessor);
  PRINT_ELAPSED_TIME(cudaDevAttrManagedMemory);
  PRINT_ELAPSED_TIME(cudaDevAttrIsMultiGpuBoard);
  PRINT_ELAPSED_TIME(cudaDevAttrMultiGpuBoardGroupID);
  PRINT_ELAPSED_TIME(cudaDevAttrHostNativeAtomicSupported);
  PRINT_ELAPSED_TIME(cudaDevAttrSingleToDoublePrecisionPerfRatio);
  PRINT_ELAPSED_TIME(cudaDevAttrPageableMemoryAccess);
  PRINT_ELAPSED_TIME(cudaDevAttrConcurrentManagedAccess);
  PRINT_ELAPSED_TIME(cudaDevAttrComputePreemptionSupported);
  PRINT_ELAPSED_TIME(cudaDevAttrCanUseHostPointerForRegisteredMem);
  PRINT_ELAPSED_TIME(cudaDevAttrReserved92);
  PRINT_ELAPSED_TIME(cudaDevAttrReserved93);
  PRINT_ELAPSED_TIME(cudaDevAttrReserved94);
  PRINT_ELAPSED_TIME(cudaDevAttrCooperativeLaunch);
  PRINT_ELAPSED_TIME(cudaDevAttrCooperativeMultiDeviceLaunch);
  PRINT_ELAPSED_TIME(cudaDevAttrMaxSharedMemoryPerBlockOptin);
  PRINT_ELAPSED_TIME(cudaDevAttrCanFlushRemoteWrites);
  PRINT_ELAPSED_TIME(cudaDevAttrHostRegisterSupported);
  PRINT_ELAPSED_TIME(cudaDevAttrPageableMemoryAccessUsesHostPageTables);
  PRINT_ELAPSED_TIME(cudaDevAttrDirectManagedMemAccessFromHost);
}
{% endhighlight %}
