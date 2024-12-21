---
layout: post
title:  "CUDAの単精度浮動小数点数近似除算命令"
date:   2021-09-19 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2 id='a'>なんの話か</h2>
<p>
CUDAの単精度浮動小数点数（binary32）用の除算命令には近似計算命令があります。<br>
この近似除算命令の精度をちょっと見てみました、という話です。
</p>
<h2 id='b'>除算近似命令</h2>
<p>
近似命令というのは<span class='code-range'>div.approx</span>です。<br>
<a href='https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div'>CUDA Toolkit Documentation 9.7.3.8. Floating Point Instructions: div</a>によると、<span class='code-range'>a / b</span>を<span class='code-range'>a * (1 / b)</span>と計算するとのことです。<br>
で、デフォルトの除算命令より速いとのことです。<br>
逆数計算は<a href='https://qiita.com/k_nitadori/items/cff0b67b1d422a5bcc01'>近似逆数平方根計算</a>などで行うのでしょうか？<br>
ただしこの命令は<span class='code-range'>b</span>が[2^{-126}, 2^{126}]という制約があります。<br>
binary32の指数部の最大値は127なので、指数部が126で仮数部が非零の場合と指数部が127の場合は計算できないということとなります。<br>
で、この制約をスケーリングにより外したものが<span class='code-range'>div.full</span>です（full-rangeのfullです）。<br>
また、これらの近似除算命令では丸めの指定はできません。
</p>
<h2 id='c'>精度評価</h2>
<p>
<span class='code-range'>div.approx</span>、<span class='code-range'>div.full</span>、<span class='code-range'>div.rn</span>（IEEE準拠で丸めにRNを用いる除算命令）の除算計算精度を倍精度で計算した場合と比較したのが下の図です。<br>
倍精度計算との相対誤差を表示しています。
</p>
<img class='img-responsive' src='{{site.baseurl}}/assets/images/cuda-approx-div.svg'>
<p>
やはり近似計算の方が若干精度が悪いことがわかります。<br>
また、<span class='code-range'>div.approx</span>でb_FP32が大きいときにerrorが飛び出ているのは<span class='code-range'>div.approx</span>のbの対応範囲外のためです。<br>
近似計算か否かに関わらず、b_FP32が大きいときは精度が少し悪くなることも確認できます。

<h2 id='d'>評価コード</h2>
{% highlight cuda %}
template <>
__device__ float div<approx_div>(const float a, const float b) {
	float r;
	asm(
			R"(
{
div.approx.f32 %0, %1, %2;
}
)": "=f"(r) : "f"(a), "f"(b)
			);
	return r;
}
{% endhighlight %}
<p>
除算部分は以下のようにinline asmを使って記述しました。<br>
Fullなコードはこちら。<br>
<a href='https://github.com/enp1s0/cuda-div-approx'>enp1s0/cuda-fiv-approx - GitHub</a>
</p>
