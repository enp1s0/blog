---
layout: post
title:  "Nsight Computeでrooflineの図を描く"
date:   2020-12-20 01:08:41 +0900
categories: cuda
---
{% include header.markdown %}

<h2 id='a'>何の話か</h2>
<p>
一言で言えば，Nsight Computeでroofline図<a href='#ref1'>[1]</a>を描く方法について書いています．
</p>
<hr>
<p>
プログラムの最適化を行う上でプロファイリングはとても重要なものです．<br>
プロブラムは計算機上で動作し，その計算機には主要な部品としてプロセッサとメモリがあります．<br>
プロセッサはムーアの法則（もう死んだとも言われていますが）によってトランジスタ数が増加し，発展を遂げてきました．<br>
一方でメモリはこのプロセッサの発展速度と比較して発展が遅いとも言われています．<br>
何が言いたいかと言うと，プロセッサもメモリも有限の性能しかなく，日々変化するこれらの性能比を知らずにプログラムの最適化などできるわけがないということです．
</p>
<p>
このため，プログラムのプロファイリングにおいて知りたいことの一つとして，実行するプロセッサやメモリが決まっている時，<b>そのプログラムがプロセッサの性能とメモリの性能のどちらに律速されているのか</b>，はたまた同期等のそれ以外の処理で律速されているのかという情報があります．<br>
この記事はこの情報を得るための1指標であるrooflineと呼ばれるものをCUDAのプロファイルでどう取得すればいいかについて書きます．
</p>

<h2 id='b'>Nsight Computeでroofline図を描く</h2>
<p>
とても簡単で，CUDA 11からNsight Computeにroofline図を描く機能が追加されています．<a href='#ref2'>[2]</a><br>
もしコマンドライン上でプロファイル結果をファイルに書き出し，これをNsight Computeで開いて見たい場合は，
<pre class='code-line'>
ncu -o report --set full -f ./a.out
</pre>
のように<span class='code-range'>--set full</span>や，<span class='code-range'>--set detailed</span>を付ける必要があるようです．
</p>
<p>
Nsight Computeで見るとこの様になります．<br>
実行したのはcuBLASのsgemm (m = n = k = \(2^{14}\))です．
</p>
![roofline]({{site.baseurl}}/assets/images/roofline.png)
<p>
横軸が理論演算量([Flops])と理論データ転送量([Byte])の比Arithmetic Intensity (AI) [Flops/byte]です．<br>
今回だと，cuBLASのsgemmで呼ばれている関数名から察するに，m=n=128, k=\(2^{14}\)な気がするので，
$$\begin{eqnarray}
\text{AI} = \frac{2mnk}{((2mn + mk + kn) \times \text{sizeof(float)})}\sim 64
\end{eqnarray}$$
くらいとなっているのかと思います．
</p>
<p>
縦軸は実際の計算を行った場合の計算速度で，今回はおおよそ16[TFlop/s]程度だったようです．
</p>
<p>
図中の青い線がハードウェアの理論性能値です．<br>
左側の斜めっている線がメモリのバンド幅の理論性能値，右側の横軸と平行な線がプロセッサの演算性能値となっています．<br>
プロセッサの方は現在SPとDPの2本が引かれているようです．
</p>
<p>
で，プロファイリングとして何を見ればいいかと言うと，図中の青い点です．<br>
これがメモリとプロセッサのどちらの理論性能値の線の真下にいるかで，そのプログラムの律速となりうるのがどちらかが分かります．<br>
今回の場合はプロセッサが律速となりうるようです．<br>
また，律速となりうる理論性能値の線との近さによって，どれほど理論性能に近い効率で計算資源を使えているかを知ることができます．
</p>

<h2 id='c'>おわりに</h2>
<p>
Nsight Computeにこの機能が入るまでは自分でAIを計算する必要がありましたが，これを使うと全自動でやってくれるのでとても楽です．<br>
<span class='code-range'>--set full</span>を付けるとプロファイルがとても遅くはなりますが．
</p>

<h3>参考</h3>
<ul>
  <li id='ref1'>[1] <a href='https://dl.acm.org/doi/10.1145/1498765.1498785'>Roofline: An Insightful Visual Performance Model For Floating-Point Programs And Multicore Architectures</a></li>
  <li id='ref2'>[2] <a href='https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/'>Accelerating HPC Applications with NVIDIA Nsight Compute Roofline Analysis - NVIDIA Developer Blog</a></li>
</ul>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
