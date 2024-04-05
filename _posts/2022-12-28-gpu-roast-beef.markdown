---
layout: post
title:  "低温調理器具メーカーNVIDIAの実力"
date:   2022-12-28 01:08:41 +0900
categories: etc
---

<h2 id='a'>背景</h2>
<p>
GPU (General Processing Unit) は汎用的な処理装置であり、今まで熱燗やココアを作ってきました。
しかし、<a href='https://www.hakko.co.jp/contest/result_n033.php'>熱コン</a>の報告書のFuture workに書いたとおり、私はローストビーフを食べたいのです。<br>
そう、GPUで低温調理です！
</p>

<h2 ad='b'>加熱器構成</h2>
<p>
例によってGPUはNVIDIA Tesla K20 2枚で、水冷で熱を奪います。<br>
冷却液は高温の状態でタンクに貯め、GPUへの入力の直前にラジエータで冷やします。<br>
冷却液タンクに加熱対象物を入れる設計です。<br>
毎年GPU熱燗をやっているので構築はこなれたものです。<br>
Kepler世代のGPUは最新のNVIDIA Driverではサポートされていないので、古いドライバをインストールします。<br>
このあたりはArchLinuxを使っていればAURに古いのが置いてあるので、yayなどで入れるだけです。
</p>

![]({{site.baseurl}}/assets/images/roast-0.jpg)
![]({{site.baseurl}}/assets/images/roast-1.jpg)

<h2 id='c'>調理</h2>
<p>
ローストビーフのレシピは<a href='https://boniq.jp/recipe/?post_type=recipe&p=4485'>このあたり</a>を参考にしました。<br>
手順としては、
<ol>
    <li>お肉を58度で3時間40分ロースト</li>
    <li>塩を含めて1時間放置</li>
    <li>表面をこんがり焼く</li>
</ol>
と言った感じです。<br>
この内1番のみをGPUを使って行います。<br>
3番も頑張ればできなくないと思いますが、GPUは適正な温度範囲内で動かさないと死ぬので、今回はなしです。<br>
（GPUの売りは計算もできる調理器具であることなので、計算できない状態で使用するのはいかがなものかと）<br>
</p>

<p>
ローストでは、特製の冷却液タンクに袋に入れたお肉を沈めます。<br>
加熱のために用いる計算は私の博士の研究テーマでもある単精度行列積です。<br>
（欲を言えばTensorコアを搭載したGPUでやりたいので、NVIDIAさん、サンプルの提供をお待ちしております）<br>
冷却液の温度はGPUの温度-7度くらいなので、nvidia-smiでGPUの温度が65度くらいで保たれるように、ラジエータのファンの露出を調整します。<br>
GPUへの入力温度は測っていませんが、おそらくTSUBAME 3.0で採用されている温水冷却より高温です。<br>
（HPLは滅ぶべきですが、）大規模化してTop500圏内に入ってGreen500の高順位を目指すのも良いかもしれません。
</p>

![]({{site.baseurl}}/assets/images/roast-2.jpg)
![]({{site.baseurl}}/assets/images/roast-3.jpg)
![]({{site.baseurl}}/assets/images/roast-4.jpg)

<p>
ロースト&放置が終わったお肉がこちら▼。<br>
以前の鮮やかな赤さは抜けました。<br>
これをこんがり焼いていきます。
</p>

![]({{site.baseurl}}/assets/images/roast-5.jpg)

<p>
出来上がったのがこちら▼。<br>
若干ローストし過ぎな気もしますが、初めてのローストビーフにしては美味しくできた気がします。<br>
SGEMMの味がほのかにし、塩加減もいい感じです。
</p>

![]({{site.baseurl}}/assets/images/roast-6.jpg)

<h2 id='d'>おわり</h2>
<p>
最近のNVIDIA GPU はどんどん消費電力が高くなっており、NVIDIAがより一層調理器具としての性能に力を入れていることが伺えます。<br>
もちろん高火力なハードウェアがあることも大切ですが、その火力を十分に引き出すソフトウェア（CUDAカーネル）を書くことも大切です。<br>
将来料理の味は個人のカーネルの最適化能力に依存する時代が来るかもしれませんね、日々精進。
</p>
