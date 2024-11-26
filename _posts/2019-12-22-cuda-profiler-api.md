---
layout: post
title:  "CUDA Profiler Control"
date:   2019-12-22 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2 id="about">CUDA Profiler Controlとは</h2>
<p>
CUDAのRuntime APIやカーネル関数のプロファイルを取りたいときに使うものです．<br>
nvprofのAPI版みたいな気持ちでしょうか<br>
CUDAのすべての呼ばれる関数ではなく，一部の必要な関数のプロファイリングを行いたいときに便利です．<br>
<a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html#group__CUDART__PROFILER">Reference</a>はあるのですが，configファイルの書き方などがあまりネットに転がっていないように思えたのでサンプルを書いておきます．
</p>

<h2 id="sample">Profile Controlサンプル</h2>
<h3>コード</h3>
{% highlight cuda %}
#include <iostream>
#include <cuda_profiler_api.h>

constexpr std::size_t N = 1lu << 13;
constexpr std::size_t block_size = 1lu << 7;

__global__ void kernel(unsigned long* const a) {
  const auto tid = blockIdx.x + blockDim.x * blockIdx.x;

  a[tid] = tid;
}

int main() {
  cudaProfilerInitialize("profile.conf", "profile.csv", cudaCSV);
  unsigned long *da;
  unsigned long *ha;
  cudaMalloc(&da, sizeof(unsigned long) * N);
  cudaMallocHost(&ha, sizeof(unsigned long) * N);

  // カーネル関数のプロファイル結果だけをCSVで出力
  cudaProfilerStart();
  kernel<<<(N + block_size - 1) / block_size, block_size>>>(da);
  cudaProfilerStop();

  cudaMemcpy(ha, da, sizeof(unsigned long) * N, cudaMemcpyDefault);

  cudaFree(da);
  cudaFreeHost(ha);
}
{% endhighlight %}

<h3>profile.conf</h3>
{% highlight cuda %}
active_warps
gridsize3d
threadblocksize
dynsmemperblock
stasmemperblock
regperthread
memtransfersize
memtransferdir
memtransferhostmemtype
streamid
cacheconfigrequested
cacheconfigexecuted
countermodeaggregate
enableonstart 0
active_warps
active_cycles
{% endhighlight %}
<p>
これは<a href="https://github.com/pytorch/pytorch/blob/master/caffe2/contrib/prof/cuda_profile_ops.cc">PyTorchのコード</a>を参考にしました．<br>
表示できる項目一覧は<a href="https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/Compute_Command_Line_Profiler_User_Guide.pdf ">COMPUTE COMMAND LINE PROFILER  User Guide</a>に書かれています．
</p>

<h3>結果</h3>
<p>こうすることで目的のCSVファイルを得ます．
{% highlight csv %}
# CUDA_PROFILE_LOG_VERSION 2.0
# CUDA_DEVICE 0 GeForce RTX 2080
# CUDA_CONTEXT 1
# CUDA_PROFILE_CSV 1
# TIMESTAMPFACTOR 15b136796fe02291
method,gputime,cputime,gridsizeX,gridsizeY,gridsizeZ,threadblocksizeX,threadblocksizeY,threadblocksizeZ,dynsmemperblock,stasmemperblock,regperthread,occupancy,streamid,cacheconfigexecuted,cacheconfigrequested,active_warps,active_cycles,memtransfersize,memtransferdir,memtransferhostmemtype
_Z6kernelPm,3.296,73.355,64,1,1,128,1,1,0,0,16,1.000,1,0,0,0,0
{% endhighlight %}
</p>
