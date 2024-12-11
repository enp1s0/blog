---
layout: post
title:  "CUDA #pragma unrollについてのtips"
date:   2019-12-25 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2 id="overview">pragma unrollと本記事の内容について</h2>
<p>
CUDAの最適化の一つとしてfor文のunrollがあると思うのですが，単に
{% highlight cuda %}
#pragma unroll
for (unsigned i = 0; i < 32; i++) {
    // iroiro
}
{% endhighlight %}
のように書くだけではなく，いろいろ指示を出せますよという話です．
</p>

<h2 id="iroiro">色々できるよunrolling</h2>
<p>
そもそもCUDAではnvccがunrollしたほうが良さそうと判断した場合には自動でunrollされます．<br>
これはこれで困る場合もあるかもしれません（本当に？）．<br>
こんなときは
{% highlight cuda %}
#pragma unroll 1
for (unsigned i = 0; i < 32; i++) {
    // iroiro
}
{% endhighlight %}
のようにunrollの後に1 を書いておくとunrollされなくなります．<br>
この1は何かと言うと，unrollの展開数です．<br>
つまり
{% highlight cuda %}
#pragma unroll 4
for (unsigned i = 0; i < 32; i++) {
    // iroiro
}
{% endhighlight %}
と書くと4回のみunrollされたものをfor文で8回回します，これを4ではなく1と書くことでunrollを無効化できるわけです．<br>
この様にunrollは展開数を指定することができます．<br>
forの回数が段数で割り切れなかった場合はPTXになる段階でうまい具合に調整されます（余りの文を最初処理し，残りをループ処理します）．
</p>
<p>
このunrollの展開数ですが，こんな感じでコンパイル時に値が判明している定数でも指定することができます．
{% highlight cuda %}
void func ( ... ) {
    // iroiro
#pragma unroll N
    for (unsigned i = 0; i < 32; i++) {
        // iroiro
    }
    // iroiro
}
{% endhighlight %}
なんなら計算もできます．
{% highlight cuda %}
void func ( ... ) {
    // iroiro
#pragma unroll (N + 4)
    for (unsigned i = 0; i < 32; i++) {
        // iroiro
    }
    // iroiro
}
{% endhighlight %}

<h2 id="omake">おまけ</h2>
<p>
pragma unrollはコンパイル時にループ回数が分かっていないループにも少し使えます．<br>
そもそもループ回数がコンパイル時に分かっていない場合も少し展開され，デフォルトでの展開のされ方は上の展開方法の展開数4の場合と似ています．<br>
はじめにループ回数を4で割ったあまりを求め，この回数だけfor内の処理を行います．<br>
あとは展開数4のループを回す感じです．<br>
そこでこのコンパイル時にループ回数が分かっていないfor文に対してpragma unrollをすると，同じ様に展開数を指定できたりします（こちらでは端数処理は最後に行われるようです）．
</p>
<h2 id="ref">参考</h2>
<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pragma-unroll"> CUDA Toolkit Document - Programming guide - B.24. #pragma unroll</a>
