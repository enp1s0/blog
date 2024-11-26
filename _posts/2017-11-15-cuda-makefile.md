---
layout: post
title:  "CUDAのプロジェクトのMakefile"
date:   2017-11-15 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<p>
最近CUDAのテンプレートを整備し直していたのでその備忘録を.
<h2 id="problem">やりたいこと</h2>
コンパイル時間が長引くのが嫌なので分割コンパイル．<br>
オブジェクトファイルがソースコードと同じディレクトリにできるのは嫌なのでobjディレクトリに出力．
<ol>
    <li>cppやcuを別々にobj/~.oにコンパイルする</li>
    <li>obj/~.oをまとめる</li>
</ol>
{% highlight make %}
.SUFFIXES: .o .cpp .cu
.cu.o:
    $(NVCC) ...
.cpp.o:
    $(CXX) ...
{% endhighlight %}
このような書き方ではオブジェクトファイルはできるもののobjディレクトリに作る事はできませんでした...
<h2 id="sol">どう書いたか</h2>
nvccコマンドには--cudaというオプションがあり，<b>cuコードをcppコードに変換する</b>．
これを使えば
<ol>
    <li>nvcc --cudaでcuをcppに変換する</li>
    <li>g++等でcppをobj/~.oにコンパイル</li>
</ol>
のようにしてやりたかったことができた．

<h2 id="ord">普段使っているMakefile</h2>
{% highlight make %}
CXX=g++
CXXFLAGS=-std=c++11
NVCC=nvcc
NVCCFLAGS= -arch=sm_61 -lcurand -lcublas $(CXXFLAGS) --compiler-bindir=$(CXX)
SRCDIR=src
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')
OBJDIR=objs
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))
TARGET=hoge

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) --cuda $&lt; -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CXX) $(CXXFLAGS) $&lt; -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)
{% endhighlight %}
これを使うとSRCDIRディレクトリ内のcpp,cuファイルをobjファイル（OBJDIR下に作られる）にし，TARGETという名の実行ファイルが作られる．
