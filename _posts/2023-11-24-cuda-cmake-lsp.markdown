---
layout: post
title:  "CUDAとcmakeとLanguage server"
date:   2023-11-24 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<h2 id='a'>なんの話か</h2>
<p>
CMakeのCUDA関連のプロジェクトで、LSPを用いたコード補完を行う場合のtips的なものです。<br>
主にVimでCoc-clangdを使っていたときに遭遇した問題と対応を書いていきます。
</p>

<h2 id='b'>cmakeとInclude directoryについて</h2>
<p>
CUDA関連のcmakeのプロジェクトでは、Response fileでinclude directoryが渡されてしまうため、LSP clientによってはこれを認識できず、include directoryが認識されません。結果、<span class='code-range'>#include </span>まで入力しても、欲しいincludeファイルが候補として現れなかったり、手動でheaderファイルを書いても、header内のコンテンツが補完候補として現れなくなります。Responseファイルは、コンパイラのコマンド引数の文字数制限を回避するための仕組みで、これ自体はありがたいものなのですけどね。
</p>
<p>
解決策として、少なくとも2通り：
<ol>
  <li>Responseファイルの使用をやめる</li>
  <li>Responseファイルの中身をcompile_commands.jsonに展開する</li>
</ol>
が考えられます。
</p>
1は、CMakeLists.txtに
{% highlight cmake %}
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
{% endhighlight %}
と書くだけです。
</p>
<p>
2は、生成されるcompile_commands.json中の中にResponseファイルのpathが書かれているので、その内容をcompile_commands.jsonに直書きするだけです。これを簡単に行うスクリプトがこちら。
<script src="https://gist.github.com/enp1s0/60ce82ba469e95782b5b1ace61d9883b.js"></script>
compile_commands.jsonを生成した後、build directory内で
{% highlight bash %}
sh ./inc_options.sh compile_commands.json
{% endhighlight %}
とすればinclude directoryが展開され、上書きされます。
