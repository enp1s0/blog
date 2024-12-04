---
layout: post
title:  "AmpereのTensorコアの話"
date:   2020-12-02 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。


<h2 id="a">TF32について</h2>
<p>
Ampereに搭載されているTensorコア Gen 3ではTF32と呼ばれる浮動小数点数を用いることができます．<br>
これは指数部8bit，仮数部10bitの計19bitの浮動小数点数です．<br>
何が32bitだ，とお思いでしょうが，CUDAではfloatとして保持するため，メモリ使用量は32bitです．<br>
要するに，普通に使う分にはメモリの節約には全くなりません．<br>
NVIDIAはこれをcuBLASのGEMM等で使えるようにしており，たまに単精度行列積と謳ってその計算性能を表示しています．<br>
しかし，TF32は仮数部が大きい，すなわち表現範囲可能範囲が広いだけで，仮数部はFP16 (IEEE Binary16)と同じであるため，精度は単精度の足元にも及びません．<br>
FP16ではアンダーフローが起きやすい計算は多少は精度がマシになるかもしれませんが．
</p>
<p>
この記事はそんなTF32を自分のカーネル関数から使う場合の注意点の話です．
</p>

<h2 id="b">TF32のTensorコアでの使い方</h2>
<p>
以前からTensorコアはWMMA APIと呼ばれるAPIで利用可能なわけですが，TF32もこれを用いることで利用可能です．<br>
ただし注意点があります．<br>
NVIDIAの資料でこのようなものを見たことがある方も多いかと思います．<br>
</p>
<div style="max-width: 500px;">
<img class="img-responsive" src="/blog/assets/images/tf32-tc.png">
</div>

出典 : <a href="https://www.nvidia.com/content/dam/en-zz/ja/Solutions/Data-Center/documents/nvidia-ampere-architecture-whitepaper-jp.pdf">NVIDIA A100 Tensor コアGPU アーキテクチャ - NVIDIA</a>
<p>
これを見ると，Tensorコアはメモリ（本当はfragment/レジスタ）からFP32の行列データを読み，Tensorコアへ送ることで使えるようになる気になります．<br>
上述したとおりTF32はメモリ上ではfloatをして保持され，Tensorコアはこの内<b>MSBから19bitのみを</b>読み込みます．<br>
これは丸めとしてはかなりおそまつなものです（RZ）．<br>
少しでも丸めを気にするならば，Tensorコアへ送る前に適切な丸めを<b>自分で</b>行う必要があります．<br>
FP32からFP16への変換（__float2half）はデフォルトでRN（最近接偶数丸め）が使われるため，自分で丸めを行わないと仮数部的にはFP16よりも変換による精度劣化が大きくなります．
</p>
<p>
ではどう変換すればいいかと言いますと，PTXの<span class="code-range">cvt.rna.tf32.f32</span>命令で少しマシな丸めで変換できます．<br>
この命令は零捨一入みたいな丸めを行います．<br>
これはmma.hをincludeしている場合は<span class="code-range">__float_to_tf32</span>関数がこれをwrapしているため，これを用いて以下のように変換できます．
</p>

{% highlight cuda %}
const float fp32 = 1.0f;
const float tf32 = __float_to_tf32(fp32);</pre>
{% endhighlight %}

<p>
PTXでは1命令ですが，少なくともAmpereではSASSレベルで複数のビット演算等に置き換わるので，HWとしては丸め回路を持っていないようです．
</p>

<h2 id="x">おわりに</h2>
<p>
「自分で変換を行ってね」というのはCUDA Toolkit Documentationにかかれています．
</p>
