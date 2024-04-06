---
layout: post
title:  "よく使うSlurmのscontrolコマンド"
date:   2020-12-16 01:08:41 +0900
categories: server
---

{% include header.markdown %}

これは<a href='https://adventar.org/calendars/5154'>rioyokotalab Advent Calendar 2020</a> 16日目の記事です．

<h2 id='a'>何の話か</h2><p>Slurmの<span class='code-range'>scontrol</span>コマンドは，クラスタの追加・設定やジョブの管理など，Slurmを使って複数ノードのジョブスケジューリングをしている場合は頻繁に使います．<br>その中で，自分がよく使うコマンドをいくつか書いていきます．</p><hr>

<h2 id='update'>scontrol update</h2><p><span class='code-range'>sinfo</span>でノードが<span class='code-range'>down</span>と表示されるようになった場合などによく使うコマンドです．<br>
こんな感じで状態を<span class='code-range'>idle</span>にできます．</p>

{% highlight bash %}
scontrol update nodename=node_name state=idle
{% endhighlight %}

いくつかのノードのstateをまとめて更新することもできます．
{% highlight bash %}
scontrol update nodename=node1,node2,node3 state=idle
{% endhighlight %}

<p><span class='code-range'>idle</span>にする場合は上記のコマンドで十分なのですが，逆にdownにしたい場合などは<span class='code-range'>reason</span>を付与する必要があります．</p>

{% highlight bash %}
scontrol update nodename=node_name state=down reason='hoge'
{% endhighlight %}

<p><span class='code-range'>down*</span>のように*が付いている場合は，Slurmのデータベース上でdownなだけではなく，ノードやネットワーク自体にも問題があるため，そちらを直さない限り<span class='code-range'>idle*</span>になったところでジョブは割り当てられません．
</p><hr>

<h2 id='show'>scontrol show</h2>

<p>このコマンドではノードやパーティション，ジョブの状態をみることができます．<br>例えば，jobに割り当てられている計算資源や時間制限を見たければ，
{% highlight bash %}
scontrol show job=job_id
{% endhighlight %}
とすることで見ることができます．</p><p>また，ノードの持つ計算資源やOSの種類，最終起動時間などを見たければ，
{% highlight bash %}
scontrol show node=node_name
{% endhighlight %}
とすることで見ることができます．</p><p>このように<span class='code-range'>ENTRY=ID</span>を指定することで，どの情報を表示させるかを指定するのですが，<span class='code-range'>ENTRY</span>は<span class='code-range'>job</span>や<span class='code-range'>node</span>以外にも多くのものを指定することができます．<br>詳細はSlurmの公式ページをご覧ください．<br><a href='https://slurm.schedmd.com/scontrol.html'>scontrol - Slurm</a></p>
<h2 id='write'>scontrol write batch_script</h2>
<p>
バッチジョブを実行するためにはジョブスクリプトを書きますが，このコマンドで指定したジョブを実行するために使ったジョブスクリプトを出力できます．<br>
使い方はこんな感じです．
{% highlight bash %}
scontrol write batch_script job_id
{% endhighlight %}
実行すると，デフォルトでは<span class='code-range'>slurm-job_id.sh</span>といったファイル名で書き出されます．<br>
標準出力で見たい場合は
{% highlight bash %}
scontrol write batch_script job_id -
{% endhighlight %}
というように<span class='code-range'>-</span>を付けることでそうできます．
</p>

<h2 id='owari'>おわりに</h2>
<p>
個人的には<span class='code-range'>scontrol</span>はユーザとしても管理者としても使いやすいものとなっていると思っていて重宝しています．<br>
ドキュメントもちゃんと整備されていてありがたいです．
</p>
