---
layout: post
title:  "Occupancyを可視化する"
date:   2021-08-08 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2 id="a">何の話か</h2>
<p>
CUDAカーネルのプロファイリングをしていると必ず遭遇する言葉「Occupancy」がそもそも何なのか、実際にスレッドが走っている様子を可視化するとどうなるのか、
みたいな話をします。
<h2>Occupancyとは？</h2>
<p>
端的に言えば、あるカーネル関数がStreaming Multiprocessor (SM)の演算器をどれほど使いきれるかという値です。<br>
この値はカーネル関数が使うレジスタ数やSharedメモリ量によって変わってきます。
</p>
<hr>
<p>
CUDAでは1つのThread blockは1つのSMで実行されますが、1つのSMは同時に1つのThread blockしか実行しないわけではありません。<br>
つまり、SMに搭載されているレジスタやSharedメモリの量的に、同時に2つのThread blockを実行可能な場合は2つ実行します（正確には「する可能性があります」）。<br>
しかし、この同時に走らせられるThread blockの数の制約は、レジスタやSharedメモリなどのプログラマブルな資源によるものだけではありません。<br>
そもそも各SMが同時に走らせられる上限のThread数が決まっているのです（注意：Thread block数ではない）。<br>
最近のGPUですと1,536 threadsで、もしThread sizeを256でカーネルを立ち上げた場合は1,536/256=6 thread blockが1 SMで同時に実行可能であることになります。<br>
この1,536という数字は以下のコードで調べられます。
</p>

{% highlight cuda %}
// nvcc threads_per_sm.cu -arch=sm_80 -lcuda
#include <stdio.h>
#include <cuda.h>

int main() {
  cuInit(0);
  CUdevice device;
  cuDeviceGet(&device, 0);

  int num_threads_per_sm;
  cuDeviceGetAttribute(
      &num_threads_per_sm,
      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
      device);

  printf("Num threads per SM = %d\n", num_threads_per_sm);
}
{% endhighlight %}

<p>
では、1 SMが同時に実行できるWarp数を考えると、これは1,536/32=48 warpsとなります。
</p>
<h3>Theoretical occupancyの話</h3>
<p>
ここでもし、1 threadが使う資源が少なく、SMが同時に48 warp分に資源を割り当てられるカーネル関数があったとします。<br>
これが<b>Theoretical occupancy</b> 100%の状態です。<br>
カーネル関数が使う少し資源が増え、24 warp分の資源しか同時に割当できないカーネル関数であればTheoretical occupancy 50%となります。<br>
</p>
<h3>Active occupancyの話</h3>
<p>
Theoretical occupancyはカーネルの立ち上げ時には決定している値です。<br>
一方で、実際に実行してみて1 SM中で同時に48 warp中何Warp走りましたか？という率が<b>Active occupancy</b>です。<br>
</p>

<h2 id="c">Occupancyを時系列で見てみる</h2>
<p>
Occupancyが何となくなにかわかったところで実際に見てみます。<br>
見たいのは、ある時刻に1 SMで同時にいくつのThread blockが走っているかです。<br>
あるThread blockがどのSMで実行されているかは<span class="code-range">%smid</span>レジスタを読めばわかります。<br>
ということで、こちらのコードでThread block IDとSMID、カーネルの実行時刻を取得します。<br>
このコードでは1024GClock (約0.6秒くらい)無をさせるカーネル関数を立ち上げます。<br>
立ち上げる際に1 thread blockあたりのSharedメモリのサイズを決定し、Theoretical occupancyを制御します。
</p>
{% highlight cuda %}
#include <iostream>

constexpr unsigned block_size = 256;
constexpr unsigned grid_size = 1u << 10;
constexpr unsigned wait_clock = 1lu << 30;

__global__ void test_kernel() {
  extern __shared__ unsigned smem[];

  const unsigned long t0 = clock64();
  while(clock64() - t0 < wait_clock){}
  const unsigned long t1 = clock64();

  unsigned smid;
  asm(
      R"({mov.u32 %0, %smid;})":"=r"(smid)
      );
  if (threadIdx.x == 0) {
    printf("%u,%u,%lu,%lu\n", blockIdx.x, smid, t0, t1);
  }
}

void launch(const unsigned smem_size) {
  int num_blocks_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm,
      test_kernel,
      block_size,
      smem_size);
  std::printf(
    "smem_size = %u,num_blocks / SM = %d\n",
    smem_size,
    num_blocks_per_sm);

  cudaFuncSetAttribute(
    &test_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
  test_kernel<<<grid_size, block_size, smem_size>>>();
  cudaDeviceSynchronize();
}

int main() {
  launch(10 * 1024);
  launch(40 * 1024);
}
{% endhighlight %}
<p>
見やすいよう開始時刻をシフトし、SM 0番で実行されているThread blockをグラフで表します。<br>
横軸はカーネルを立ち上げてからの経過Clock、縦軸は実行されているThread block IDです。<br>
Theoretical occupancy = 100%の場合、1 SMあたり6 thread blocks (=1,536/256(thread size))が実行されます。
</p>
<hr>
<p>
Sharedメモリの使用量を少なめに設定し、Theoretical occupancy = 100%(6/6)としたのが下のグラフです。<br>
重なっていて少し見にくいですが、確かに1つのclock区間に6つのThread blockが走っていることがわかります。
</p>

<img src="{{site.baseurl}}/assets/images/occupancy-6.svg">

<p>
▲ SM 0番で動作するThread blockの時系列表示。
</p>
<hr>
<p>
次はSharedメモリの使用量を増やし、Theoretical occupancy = 33.3%(2/6)としました。<br>
確かに1つのclock区間に2つのThread blockが走っていることがわかります。
</p>
<img src="{{site.baseurl}}/assets/images/occupancy-2.svg">
<p>
▲ SM 0番で動作するThread blockの時系列表示。
</p>
<hr>
<p>
この1 clock区間に走っているThread blockの数（の最大値(=1,536/thread_size)に対する割合）こそがOccupancyです。
</p>
<h3>Theoretical occupancyを2倍にできれば理論計算速度も2倍になる？</h3>
<p>
という疑問は当然出てくると思いますが、残念ながら2倍にはなりません。<br>
1 SMが複数のThread blockを実行できるのは、命令の遅延を隠すための機能です。<br>
同時に複数のThread blockが実行されると言っても、演算器の個数はThread数と比較して少ないため、同じタイミングで全Thread blockが同じ命令を実行されるわけではありません。<br>
Threadはタイミングが微妙にずれながら動いています。<br>
（タイミングが揃っていることが保証されているのはsyncされているWarpのみです。）<br>
このズレが演算器を衝突なく利用するための鍵で、Occupancyを高めるということはこのズレたThreadをたくさん生やせるようにするということです。<br>
この思想はCUDAでの命令の遅延隠蔽の根幹となっています。<br>
では、逆にOccupancyが低いと隠蔽を隠せないかと言うと必ずしもそうではなく、Thread数はThread sizeを大きくすることでも増やせるのでこちらで隠蔽していくという手もあります。<br>
遅延隠蔽は他にも使用する回路の異なる命令を同時に動かす命令重ね合わせ（Instruction-level overlap）や、ApmereからはAsynchronous global-to-shared data copyなどがあります。
</p>

<h3>おまけ：SM割当</h3>
<p>
では、あるthread blockがどのSMで実行されるのかには規則があるのでしょうか？<br>
例えば<span class="code-range">blockIdx.x % num_sm</span>のように<span class="code-range">blockIdx.x</span>から簡単に計算可能なものなのでしょうか？<br>
ということで、横軸に<span class="code-range">%smid</span>、縦軸に<span class="code-range">blockIdx.x</span>をとったものが下のグラフたちです。
</p>
<ul>
<li>1 SMあたり6 thread blocksが実行される場合（Theoretical occupancy=100%の場合）<br>
<img src="{{site.baseurl}}/assets/images/sm-sche-6.svg">
<li>1 SMあたり2 thread blocksが実行される場合（Theoretical occupancy=33.3%の場合）<br>
<img src="{{site.baseurl}}/assets/images/sm-sche-2.svg">
</ul>
<p>
RTX 3080を用いており、SMは64基搭載されています。<br>
このためSMIDは64で折り返されます。<br>
1周目の割当時のSMIDは<span class="code-range">blockIdx.x % num_sm</span>で決まり、2周め以降は適当に、という感じに見えますね。
</p>

<h2 id="d">おわり</h2>
<p>
あまり日本語でOccupancyの説明が書かれたWebページが見当たらなかったので書いてみましたが、読んでいただき何か得るものがあったのであれば幸いです。<br>
特に可視化やsmid割当に関しては英語などでも説明されているものを見たことがないので、少しはオリジナリティがあるかなと思います。<br>
SMIDとSIMDはアナグラムになっていてパット見見分けがつかないので、どこかでtypoしていたらごめんなさい。
</p>
