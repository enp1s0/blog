---
layout: post
title:  "CUDAの静的ライブラリを作るには"
date:   2020-12-22 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2 id="a">何の話か</h2>
<p>
CUDA用の静的ライブラリを作る場合のコンパイルオプションについてです．<br>
CUDAではコンパイルを行う時にホスト側のコードとデバイス側のコードを分離し，デバイス側のコードは<span class="code-range">gencode</span>で指定したアーキテクチャの種類だけカーネルイメージを作成します．<br>
この際後方互換性のためにPTXをイメージに埋め込みます．<br>
nvccでCUDAの静的ライブラリを作る時は，これらイメージを再配置可能にする必要があり，そのためのオプションが用意されています．<br>
この記事はこれらのオプションを使って静的ライブラリを作る話です．
</p>

<h2 id="b">作り方</h2>
<p>
ライブラリの定義コード（test.cu）から静的ライブラリ（libtest.a）を作ることが目標です．<br>
今回サンプルとして以下のコードを用います．
</p>

<b>test.cu</b>
{% highlight cuda %}
#include <stdio.h>
#include "test.hpp"

namespace {
__global__ void hello_kernel() {
	printf("[GPU] Hello\n");
}
}

void print_hello() {
	printf("[CPU] Hello\n");
	hello_kernel<<<1, 1>>>();
	cudaDeviceSynchronize();
}
{% endhighlight %}
<b>test.hpp</b>
{% highlight cuda %}
#ifndef __TEST_HPP__
#define __TEST_HPP__

void print_hello();

#endif
{% endhighlight %}

<p>
はじめに定義コード（test.cu）から普通のオブジェクトファイル（test.o）を作ります．<br>
この際，埋め込みたいカーネルイメージの対象だけ<span class="code-range">-gencode arch=compute_XX,code=sm_XX</span>を列挙します．
</p>
<pre class="code-line">
nvcc test.cu -c -o test.o -dc -gencode arch=compute_86,code=sm_86
</pre>
<p>
次に，今作ったオブジェクトファイル内のイメージを再配置可能にします．<br>
これにはnvccの<span class="code-range">-dlink</span>オプションを用います．<br>
これにより先程作ったtest.oからtest.dlink.oを得ます．
</p>
<pre class="code-line">
nvcc -dlink test.o -o test.dlink.o -gencode arch=compute_86,code=sm_86
</pre>
<p>
最後に，作ったtest.oとtest.dlink.oから目的のlibtest.aを作成します．
</p>
<pre class="code-line">
nvcc -o libtest.a -lib test.dlink.o test.o
</pre>
<p>
これだけです．<br>
定義コードがたくさんある場合は，それぞれの.oと.*dlink.oを作って最後の手順に渡すことで一つのライブラリを作れます．
</p>

<h2 id="c">使い方</h2>
<p>
簡単に作ったライブラリを他のプログラムにリンクさせながらビルドするにはnvccを使うのがおすすめです．<br>
例えばこんなコードをコンパイルしてみます．
</p>
<b>main.cpp</b>
{% highlight cuda %}
#include "test.hpp"

int main() {
	print_hello();
}
{% endhighlight %}
<p>
コンパイルプションはこんな感じです．
</p>
<pre class="code-line">
nvcc main.cpp -L[/path/to/libtest.a (w/o libtest.a)] -ltest
</pre>

<p>
静的ライブラリは作りましたが，CUDAの公式のライブラリたちは動的ライブラリなため，このライブラリとは別途リンクさせる必要があります．<br>
nvccであればこれを自動で行ってくれますが，gccなどの他のコンパイラを用いる場合は自分でオプションを付ける必要があります．
</p>

<h2 id="d">おわりに</h2>
<p>
対象のアーキを列挙すればするほどライブラリのファイルサイズが増加し，かつビルド時間も増加します．<br>
不必要なアーキは列挙しないのがおすすめです．
</p>

<h3>参考</h3>
<ul>
  <li><a href="https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-specifying-compilation-phase-device-link">--device-link - CUDA TOOLKIT DOCUMENTATION</a></li>
</ul>
