---
layout: post
title:  "nvccのコンパイル時の一時ファイルを残す方法"
date:   2020-04-08 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2>何の話か</h2>
<p>
PTXをinline assemblerで書いている場合など，nvcc (ptxas)でアセンブル時に
<pre class="code-line">
ptxas /tmp/tmpxft_00005f3d_00000000-5_main.ptx, line 76; fatal   : Parsing error near '%r10': syntax error
</pre>
のように一時ファイルのptxの行番号などが表示されることがありますが，nvccは終了時に一時ファイルを自動で消すためエラーの追跡ができません．<br>
そこで一時ファイルを消さないようにするにはどうすればいいのかという話．
</p>

<h2>--keepオプション</h2>
<p>
nvccに--keepオプションを渡すと中間のファイルがCurrect directoryに展開され，かつ自動では消されません．
<pre class="code-line">
ptxas main.ptx, line 76; fatal   : Parsing error near '%r10': syntax error
</pre>
するとエラーメッセージもそのディレクトリのものとなりエラーを追えるようになります．<br>
中間ファイルはptxやiiなどいくつかあり，Current directoryが散らかるのですが，これを避けるためには--keep-dirオプションで中間ファイルの置き場を指定してあげるといいようです．
<pre class="code-line">
nvcc main.cu -std=c++14 -arch=sm_75 --keep --keep-dir=tmp
</pre>
</p>
