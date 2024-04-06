---
layout: post
title:  "--dkmsを指定してNVIDIAのドライバをインストールしようとしてコケた話"
date:   2020-03-27 01:08:41 +0900
categories: server
---

{% include header.markdown %}

<h2 id='problem'>何が起きたか</h2>
<p>
NVIDIAのドライバをこのコマンドでインストールしようとした．
{% highlight bash %}
sudo ./NVIDIA-Linux-x86_64-418.87.00.run --silent --dkms
{% endhighlight %}
インストールに失敗したため，NVIDIAのドライバのインストールログである
<pre class='code-line'>
/var/log/nvidia-installer.log
</pre>
を見たところ，dkms buildコマンドでコケている事がわかった．
そこでdkmsコマンド実行時のログファイルである
<pre class='code-line'>
/var/lib/dkms/nvidia/418.87.00/build/make.log
</pre>
を見に行ったところ，下記のようなエラーが出ていた．
</p>

{% highlight text %}
Compiler version check failed:

The major and minor number of the compiler used to
compile the kernel:

gcc version 7.3.0 (Ubuntu 7.3.0-16ubuntu3)

does not match the compiler used here:

cc (Ubuntu 7.4.0-1ubuntu1~18.04) 7.4.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


It is recommended to set the CC environment variable
to the compiler that was used to compile the kernel.

The compiler version check can be disabled by setting
the IGNORE_CC_MISMATCH environment variable to "1".
However, mixing compiler versions between the kernel
and kernel modules can result in subtle bugs that are
difficult to diagnose.

*** Failed CC version check. Bailing out! ***

/var/lib/dkms/nvidia/418.87/build/Kbuild:182: recipe for target 'cc_version_check' failed
make[2]: *** [cc_version_check] Error 1
make[2]: *** Waiting for unfinished jobs....
Makefile:1552: recipe for target '_module_/var/lib/dkms/nvidia/418.87/build' failed
make[1]: *** [_module_/var/lib/dkms/nvidia/418.87/build] Error 2
make[1]: Leaving directory '/usr/src/linux-headers-4.15.0-91-generic'
Makefile:81: recipe for target 'modules' failed
make: *** [modules] Error 2
{% endhighlight %}

<h2 id='env'>環境</h2>
<dl>
  <dt><b>name -a</b></dt><dd>
{% highlight text %}
Linux XXXX 4.15.0-91-generic #92-Ubuntu SMP Fri Feb 28 11:09:48 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux
{% endhighlight %}
</dd>
  <dt><b>lspci | grep VGA</b></dt><dd>
{% highlight text %}
03:00.0 VGA compatible controller: NVIDIA Corporation Device 1e82 (rev a1)
{% endhighlight %}
</dd>
</dl>

<h2 id='sol'>解決</h2>
<p>
Linux kernelのビルドに用いられたgccのバージョンとKernel moduleをビルドするのに使うgccのバージョンが異なるためエラーとなっているらしい．<br>
gccのバージョンを落とすというのも一つの手だが，このバージョンの違いを無視するようにしたほうが穏便に済ませられそう（誰しもが通る道だと思っていますが，昔CentOSでSystemのgccのバージョンを無理矢理上げていろいろ壊してしまい，OSの再インストールをする羽目になったことがあります）．<br>
バージョンの差を無視するには
{% highlight bash %}
sudo ./NVIDIA-Linux-x86_64-418.87.00.run --silent --dkms --no-cc-version-check
{% endhighlight %}
のように--no-cc-version-checkオプションを渡してあげればいいようです．
</p>
