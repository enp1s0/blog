---
layout: post
title:  "CUDA 11.2で導入されたcudaMallocAsyncとcudaFreeAsyncについて"
date:   2020-12-18 01:08:41 +0900
categories: etc
---

{% include header.markdown %}
これは<a href='https://adventar.org/calendars/5154'>rioyokotalab Advent Calendar 2020</a>
 18日目の記事です．<h2 id='a'>何の話か</h2><p>CUDA 11.2から<span class='code-range'>cudaMallocAsync</span>，<span class='code-range'>cudaFreeAsync</span>という関数がCUDAに追加されました．<br>
これと同時にmemory poolという概念がCUDAのメモリにも導入されました．<br>
<span class='code-range'>cudaMallocAsync</span>はmemory poolに指定したサイズのメモリ領域がある場合はそれを確保し，ない場合は普通にメモリを取りに行きます．<br>
<span class='code-range'>cudaFreeAsync</span>はmemory poolへメモリを返します．<br>
memory poolにはデフォルトのものとと我々開発者が作成できるものの2種類があります．<br>
この記事はこのAsync関数の性能・挙動について書きます．</p>
<h2 id='b'>評価プログラム</h2><p>評価にはこちらのコードを用いました．<br>
このプログラムでは，頻繁にmalloc/freeを繰り返した場合の処理時間を調べます．<br>
そのために単にメモリを確保し，要素ごとの演算を行うだけのカーネル関数を呼び，終わり次第freeします．<br>
これを4回行います．<br>
この際malloc/freeに<span class='code-range'>cudaMalloc</span>/<span class='code-range'>cudaFree</span>を使う場合は逐次実行，<span class='code-range'>cudaMallocAsync</span>/<span class='code-range'>cudaFreeAsync</span>は4 streamに分けて実行するようにし，それぞれの処理時間を比較します．</p>
<script src="https://gist.github.com/enp1s0/8c1a6fd67ceb811cb5b59bba7cedb9e0.js"></script>
<h3>計算環境</h3><ul><li>NVIDIA GeForce RTX 3080</li><li>CUDA 11.2</li><li>NVIDIA Driver 460.27.04</li></ul><h2 id='d'>結果</h2><table class="table"><thead>  <tr>    <th class="tg-0lax"></th>    <th class="tg-0lax"><span class='code-range'>cudaMallocAsync</span> / <span class='code-range'>cudaFreeAsync</span></th>    <th class="tg-0lax"><span class='code-range'>cudaMalloc</span> / <span class='code-range'>cudaFree</span></th>  </tr></thead><tbody>  <tr>    <td class="tg-0lax">計算時間</td>    <td class="tg-0lax">0.2112 [秒]</td>    <td class="tg-0lax">12.69 [秒]</td>  </tr></tbody></table><p>Asyncの方が圧倒的に速いですね．<br>
この原因は主に<span class='code-range'>cudaFree</span>にあるようです．</p>
<h3>プロファイリング</h3><p>Nsight Systemsで処理のタイムラインを見てみます．</p>

<h4>普通のcudaMalloc/cudaFree</h4>
![]({{site.baseurl}}/assets/images/async-cudafree.png)
<p><span class='code-range'>cudaFree</span>にほとんどの時間が使われているのが見えます．<br>
この時のメモリの使用量の推移をお手製の <a href='https://github.com/enp1s0/gpu_logger'>gpu_logger</a>
 で見てみます．<br>
このプログラムはNVMLを用いて，好きなプログラムを実行しながらGPUで確保されているメモリ量を記録していくことができます．</p>
![]({{site.baseurl}}/assets/images/async-used-memory-sync.svg)
<p>今回の評価プログラムでは1回のmallocあたり4GiBを確保します．<br>
<span class='code-range'>cudaMalloc</span> / <span class='code-range'>cudaFree</span>は逐次実行なため，常に4GiBちょっと取られている状態が見て取れます．<br>
freeの最中でもNVMLからはfree開始時の容量が見えるのですね．<br>
また，単純に考えて1回のfreeに3秒ほどかかっていることが計算できます<span id='tag-a'>[A]</span>．</p>
<h4>Asyncのmalloc/free</h4><p>同様にNsight Systemsでタイムラインを見てみます．</p>
![]({{site.baseurl}}/assets/images/async-cudafreeasync.png)
<p>4つの赤い四角が<span class='code-range'>cudaMallocAsync</span>となっています．<br>
<span class='code-range'>cudaFreeAsync</span>は<span class='code-range'>cudaMallocAsync</span>の隙間にあるのですが，相対的に時間が短くほとんど無視できる時間となりました．<br>
しかし，<a href='#tag-a'>[A]</a>
から4GiBのfreeには3秒ほどかかるはずで，これはどこへ行ったのかが気になります．<br>
実はプロファイル結果には続きがあります．</p>
![]({{site.baseurl}}/assets/images/async-cudafreeasync-2.png)
<p>後ろに空白の処理が6秒ほどあり，およそ8GiB分くらいをfreeできる時間となっています<span id='tag-b'>[B]</span>．</p>
<p>では先ほどと同様にメモリ使用量の推移を見てみます．</p>
![]({{site.baseurl}}/assets/images/async-used-memory-async.svg)
<p>このグラフを見ると，<ol><li>複数のstreamが順にmallocを行いGPUの搭載メモリ量に達するまでmallocして行くが，いずれ確保できなくなる</li><li>あるstreamで処理が終わるとfreeが始まる</li><li>freeにより空き容量が増えると指定量（今回は4GiB）を確保できていなかったstreamが確保しに行く</li></ol>みたいな様子が読み取れるかと思います．<br>
最後0.20秒後に一気にメモリ使用量が0となっていますが，その直前の確保量が8GiB程です．<br>
このfreeに<a href='#tag-b'>[B]</a>
の6秒が使われているのかな，とか思ったり思わなかったりです．</p>
<h2 id='e'>おわりに</h2><p>memory poolと言うからにはNVMLでメモリ使用量を見ると，増えていく一方でプログラムの最後に一気に開放するのかなと思っていましたが，実際はfreeによって途中で減っていく様子が見られました．<br>
どういう実装になっているのでしょうね．<br>
</p>
