---
layout: post
title:  "CUDAでShared memoryを48KiB以上使うには"
date:   2021-02-03 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2 id="a">何の話か</h2>
<p>
例えばGeForce RTX 3080 (Shared memory/L1 Cache: 128KB)で走らせることを想定した以下のコードがあります。<br>
このコードは64KiB分のShared memoryのデータをGlobal memoryに書き出すだけのコードです。
</p>
{% highlight cuda %}
// 64 KiB
constexpr unsigned shared_memory_size = 64 * 1024;

__global__ void kernel(float* const ptr) {
  __shared__ float smem[shared_memory_size / sizeof(float)];

  const unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= shared_memory_size / sizeof(float)) return;

  ptr[tid] = smem[tid];
}

int main() {
  float *d_array;
  cudaMalloc(&d_array, shared_memory_size);

  kernel<<<shared_memory_size / 256, 256>>>(d_array);
  cudaDeviceSynchronize();
}
{% endhighlight %}
<p>
このコードをnvccでコンパイルすると、アクセスするShared memoryのアドレスは搭載されているShared memoryの大きさを超えていないですが、エラーとなります。
</p>
<pre class="code-line">
ptxas error   : Entry function '_Z6kernelPf' uses too much shared data (0x10000 bytes, 0xc000 max)
</pre>
<p>
<span class="code-range">0xc000 bytes</span>は48KiBです。<br>
ではどう書いたら48KiB以上のSharedメモリを使えるようになるかというのがこの記事です。
</p>
<hr>
<h2 id="b">48KiBを超えるSharedメモリの確保の仕方</h2>
<p>
48KiB以上のShared memoryを確保するために行うことは3つです。
</p>
<ol>
  <li>カーネル関数内でのShared memoryの宣言を修正</li>
  <li><span class="code-range">cudaFuncSetAttribute</span>関数で、立ち上げるカーネル関数が必要とするShared memoryの大きさを設定</li>
  <li>カーネル関数の立ち上げ時に確保するShared memoryのサイズを指定</li>
</ol>
<p>
この3つを行うよう上記のコードを書き換えると以下のようになります。
</p>

{% highlight cuda %}
// 64 KiB
constexpr unsigned shared_memory_size = 64 * 1024;

__global__ void kernel(float* const ptr) {
  // 1.
  extern __shared__ float smem[];

  const unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= shared_memory_size / sizeof(float)) return;

  ptr[tid] = smem[tid];
}

int main() {
  float *d_array;
  cudaMalloc(&d_array, shared_memory_size);

  // 2.
  cudaFuncSetAttribute(&kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);
  // 3.
  kernel<<<shared_memory_size / 256, 256, shared_memory_size>>>(d_array);
  cudaDeviceSynchronize();
}
{% endhighlight %}

<hr>
<h2 id="c">終わりに</h2>
<p>
はじめのコードでは、コンパイルの時点ではどのアーキで実行されるかは判定できないため、サポートしているGPUの最小Shared memoryサイズを上限としてエラーを出しているということですかね。<br>
ptxasでエラーが出ていることからも分かるとおり、ptxからカーネルイメージに落とす際にエラーが出るのですが、cuからptxへの変換はエラーなく行われます。<br>
ですので、<span class="code-range">nvcc -ptx main.error.cu</span>でptxを見てみると、64KiBのShared memoryをとろうとしていることが確認できます。

{% highlight cuda %}
.version 7.2
.target sm_52
.address_size 64

.visible .entry _Z6kernelPf(
  .param .u64 _Z6kernelPf_param_0
)
{
  // iroiro

  // demoted variable
  .shared .align 4 .b8 _ZZ6kernelPfE4smem[65536];

  // iroiro

}
{% endhighlight %}

</p>
