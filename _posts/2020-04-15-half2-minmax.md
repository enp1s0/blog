---
layout: post
title:  "CUDA half2のmax/min"
date:   2020-04-15 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2>背景</h2>
<p>
ないんですよね，CUDAのhalf2に対するSIMD max/minが．（追記 2020.10.07：Ampare &amp; CUDA 11から導入されました．）<br>
halfにもないので愚直に書くとバラしてfloatにキャストするなりしてmax/minとって詰め戻すことになるかと．<br>
そこをうまいことbit演算でできるよというのがこの記事です．
</p>

<h2>コード</h2>
{% highlight cuda %}
// max
CUTF_DEVICE_FUNC inline __half2 max(const __half2 a, const __half2 b) {
#if __CUDA_ARCH__ < 800
    const half2 sub = __hsub2(a, b);
    const unsigned sign = (*reinterpret_cast<const unsigned*>(&sub)) & 0x80008000u;
    const unsigned sw = ((sign >> 21) | (sign >> 13)) * 0x11;
    const int res = __byte_perm(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&b), 0x00003210 | sw);
    return *reinterpret_cast<const __half2*>(&res);
#else
    // For Ampere~
    return __hmax2(a, b);
#endif
}
CUTF_DEVICE_FUNC inline __half max(const __half a, const __half b) {
#if __CUDA_ARCH__ < 800
    const half sub = __hsub(a, b);
    const unsigned sign = (*reinterpret_cast<const short*>(&sub)) & 0x8000u;
    const unsigned sw = (sign >> 13) * 0x11;
    const unsigned short res = __byte_perm(*reinterpret_cast<const short*>(&a), *reinterpret_cast<const short*>(&b), 0x00000010 | sw);
    return *reinterpret_cast<const __half*>(&res);
#else
    // For Ampere~
    return __hmax(a, b);
#endif
}
{% endhighlight %}


<p>
肝となるのは__byte_perm関数．<br>
何をしているかというのは「<a href="https://enp1s0.github.io/blog/cuda/2019/12/20/cuda-int.html">CUDAの整数 &amp; bit演算関数 - 天炉48町</a>」を見てもらえればわかると思います．<br>
PTXで見ても分岐なしたったの9演算．<br>
嬉しいですね．<br>
int8なんかも最初の引き算ができれば同じ要領でSIMD max/min関数が作れそうですね．
</p>
