---
layout: post
title:  "Environment Modulesを使う"
date:   2018-06-07 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}


<h2 id="pre">前書き</h2>
CUDAを使っているとgccのバージョンをやたらにあげられないけど、バージョンの高いgccを使いたい時もあったりなかったり.<br>
そんな時に使うgccのバージョンをちょちょいと切り替えできたら嬉しい.<br>
そんな時に使えるのがmodulesと言うもの.<br>
スパコンなんかでたくさん使われているイメージ.
<h2 id="install">インストール</h2>
CentOSだと
<pre class="code-line">
yum install environment-modules
</pre>
で入るはず.
<h2 id="config_path">設定ファイルの置き場</h2>
<pre class="code-line"><code class="bash">echo $MODULEPATH</code></pre>
で表示されるディレクトリをmoduleコマンドは検索する.<br>
root権限がないとかで他の場所に置きたければ
<pre class="code-line"><code class="bash">module use /path/to/modulefiles/dir</code></pre>
で検索ディレクトリを追加できる.


<h2 id="config">設定ファイル</h2>
例: 野良ビルドしたgcc
<pre class="code-line"><code class="prettyprint">#%Module 4.1

conflict gcc     
prepend-path  PATH /usr/local/gcc-5.4.0/bin
prepend-path  LD_LIBRARY_PATH /usr/local/gcc-5.4.0/lib64
prepend-path  CPATH /usr/local/gcc-5.4.0/include</code></pre>
<ol>
<li><br>
<pre class="code-line">#%Module *.*</pre>はじめのおまじない<br>
ないと怒られる</li>
<li><br>
<pre class="code-line">conflict        gcc</pre>
モジュールのグループのようなもの<br>
これが同じモジュールを2つ以上ロードしようとすると怒られる</li>
<li><br>
<pre class="code-line">prepend-path   ENV_VAR val</pre>
環境変数ENV_VARの先頭にvalを追加</li>
<li>(例にはないけど)
<pre class="code-line">setenv ENV_VAR val</pre>
環境変数ENV_VARをvalに設定
</li>
</ol>
<h3 id="dep">依存関係を記述</h3>
<ul>
<li>
<pre class="code-line">prereq hoge</pre>
と書けばhogeがloadされていない場合に怒られる</li>
<li>依存モジュールが勝手にロードされるようにしたければ
<pre class="code-line">load hoge</pre>
と記述しておく.<br>
この場合依存元をunloadするとhogeもunloadされる.<br>
hogeをunloadしたくなければ
<pre class="code-line">always_load hoge</pre>
とすればいい.
</li>
</ul>
<h2 id="use">使い方</h2>
<h3 id="avail">全モジュール一覧を表示</h3>
<pre class="code-line"><code class="bash">module avail</code></pre>
<h3 id="load">モジュールをロード</h3>
<pre class="code-line"><code class="bash">module load module_name</code></pre>
<h3 id="unload">モジュールをアンロード</h3>
<pre class="code-line"><code class="bash">module unload module_name</code></pre>
<h3 id="list">ロードしているモジュールの一覧を表示</h3>
<pre class="code-line"><code class="bash code-line">module list</code></pre>
<h2 id="prob">インストールしたのにmoduleコマンドが無いと言われる場合</h2>
shellの.*rcファイルなどに
<pre class="code-line"><code class="prettyprint">. /etc/profile.d/modules.sh
</code></pre>
などと書いておけばいいかも.
