---
layout: post
title:  "CUDA device関数を別コンパイル単位に書く (ptxas fatal : Unresolved extern function)"
date:   2018-08-26 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

__global__関数と__device__関数を別ファイルに書くなどした場合に
<span>
ptxas fatal   : Unresolved extern function
</span>
というエラーが起きることがある．
<h2>原因</h2>
__device__関数は基本的には__global__関数にインライン展開される.<br>
なので別ファイルなどコンパイル単位が異なる場合はインライン展開できず、このようなエラーが出る.
<h2>解決法</h2>
少なくとも2通りある.
<h3>nvccのコンパイルオプションで解決する方法</h3>
<pre class="code-line"><code>$ nvcc --device-c ...</code></pre>
や
<pre class="code-line"><code>$ nvcc -dc ...</code></pre>
でコンパイルすることでリンクできる.(関数はインライン展開されない)
<h3>__device__関数を__global__関数と同じ翻訳単位に書く方法</h3>
同じファイルやincludeするヘッダファイルに書いておけばいい．
インライン展開される.

<h2>おまけ</h2>
(最近の)__device__関数は再起呼び出しが可能である.<br>
階乗計算関数など簡単な再帰関数ではPTXの段階で再起が削除されループとなるみたい.
