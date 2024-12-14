---
layout: post
title:  "CUDAのstreamにhost関数を流すには"
date:   2021-08-08 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2 id="about">何がしたいか</h2>
<p>
CUDAにはstreamと呼ばれるスケジューリング機能があり、これを用いることでメモリの非同期コピーなどができます。<br>
CUDA 10からは<a href="https://developer.nvidia.com/blog/cuda-graphs/">CUDA Graph</a>と呼ばれるstreamをグラフとして記述し実行する機能もあり、複数のカーネル関数を1つ1つ立ち上げるのと比較してGraphとして立ち上げたほうが高速に立ち上げられるという結果もあります<a href="#a100-wp">[1]</a>。<br>
で、今回はそんなstreamにCPU側の処理を流したい場合にどうすればいいかという話です。
</p>

<h2 id="sample">サンプルコード</h2>
<p>
<span class="code-range">cuLaunchHostFunc</span>関数<a href="#cuLaunchHostFunc">[2]</a>を使うと流すことができます。
</p>
{% highlight cuda %}
#include <cstdio>
#include <cuda.h>

struct data_struct_t {
  int a, b, c;
};

void host_func(void* const data_void_ptr) {
  auto* data_ptr = reinterpret_cast<data_struct_t*>(data_void_ptr);

  data_ptr->a = 10;
  data_ptr->b = 20;
  data_ptr->c = 30;
}

int main() {
  cudaStream_t cuda_stream;
  cudaStreamCreate(&cuda_stream);

  data_struct_t data;

  cuLaunchHostFunc(cuda_stream, &host_func, reinterpret_cast<void*>(&data));

  cudaStreamSynchronize(cuda_stream);

  std::printf("a=%d, b=%d, c=%d\n", data.a, data.b, data.c);
}
{% endhighlight %}
<p>
コンパイル時には
<pre class="code-line">
nvcc main.cu -std=c++11 -arch=sm_70 -lcuda
</pre>
の様に<span class="code-span">libcuda</span>をリンクさせる必要があります。<br>
<span class="code-range">cuLaunchHostFunc</span>関数に渡すhost側の関数の引数は<span class="code-range">void*</span>とし、<span class="code-range">cuLaunchHostFunc</span>関数の第３引数に渡します。<br>
気持ちとしてはWindows APIの<span class="code-range">CreateThread</span>の気持ちです（伝われ）。
</p>
<p>
<span class="code-range">cuLaunchHostFunc</span>関数はExecution Controlの一部ですが、CUDA 10.1でAPIが刷新されており、これによって追加された関数です。<br>
そのためCUDA 10.0以前では使えないのでお気をつけください。
また、制約として、cudaMallocなどのCUDA APIは、引数で渡す関数内で実行できません。
</p>

<h3 id="ref">参考</h3>
<ol>
  <li><a id="a100-wp" href="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf">NVIDIA A100 Tensor Core GPU Architecture</a>, NVIDIA, (p60 : Task Graph Acceleration on NVIDIA Ampere Architecture GPUs)</li>
  <li><a id="cuLaunchHostFunc" href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f">cuLaunchHostFunc</a> - Execution Control, CUDA TOOLKIT DOCUMENTATION</li>
</ol>
