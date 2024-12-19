---
layout: post
title:  "CUDAでLambda関数"
date:   2019-06-11 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

CUDAではCUDA 8からカーネル関数内でLambda関数が使えます．
で，PTXコードがどうなるのか気になったので調べてみました．

<h3>検証コード</h3>
{% highlight cuda %}
template <class Func>
__device__ void launch(Func func){
  func();
}

__global__ void kernel_0(){
  launch([]__device__(){printf("poi 0\n");});
}

template <class Func>
__global__ void kernel_1(Func func){
  launch(func);
}


int main(int argc, char** argv){
  kernel_1<<<1, 1>>>([]__device__(){printf("poi 1\n");});
  kernel_1<<<1, 1>>>([argc]__device__(){printf("poi 2 %d\n", argc);});
}
{% endhighlight %}
このコードには計3つのカーネル関数があります．
<ol>
    <li>kernel_0<br>カーネル関数内に直にLambda関数を書くパターン</li>
    <li>kernel_1<br>カーネル関数の引数にLambda関数を書くパターン</li>
    <li>kernel_1<br>カーネル関数の引数にホストのデータをキャプチャするLambda関数を書くパターン</li>
</ol>

<h3>PTXコード</h3>
<pre class="code"><code>nvcc lambda.cu --ptx --expt-extended-lambda -arch=sm_75</code></pre>でPTXコードを出力させます．
<pre>
.version 6.4
.target sm_75
.address_size 64

// .globl	_Z8kernel_0v
.extern .func  (.param .b32 func_retval0) vprintf
(
    .param .b64 vprintf_param_0,
    .param .b64 vprintf_param_1
    )
  ;
.global .align 1 .b8 $str[7] = {112, 111, 105, 32, 48, 10, 0};
.global .align 1 .b8 $str1[7] = {112, 111, 105, 32, 49, 10, 0};
.global .align 1 .b8 $str2[10] = {112, 111, 105, 32, 50, 32, 37, 100, 10, 0};

.visible .entry _Z8kernel_0v(

                            )
{
  .reg .b32 	%r<2>;
  .reg .b64 	%rd<4>;

  mov.u64 	%rd1, $str;
  cvta.global.u64 	%rd2, %rd1;
  mov.u64 	%rd3, 0;
  // Callseq Start 0
  {
    .reg .b32 temp_param_reg;
    .param .b64 param0;
    st.param.b64	[param0+0], %rd2;
    .param .b64 param1;
    st.param.b64	[param1+0], %rd3;
    .param .b32 retval0;
    call.uni (retval0),
        vprintf,
        (
            param0,
            param1
        );
    ld.param.b32	%r1, [retval0+0];
  }// Callseq End 0
  ret;
}

// .globl	_Z8kernel_1IZ4mainEUlvE_EvT_
.visible .entry _Z8kernel_1IZ4mainEUlvE_EvT_(
    .param .align 1 .b8 _Z8kernel_1IZ4mainEUlvE_EvT__param_0[1]
    )
{
  .reg .b32 	%r<2>;
  .reg .b64 	%rd<4>;

  mov.u64 	%rd1, $str1;
  cvta.global.u64 	%rd2, %rd1;
  mov.u64 	%rd3, 0;
  // Callseq Start 1
  {
    .reg .b32 temp_param_reg;
    .param .b64 param0;
    st.param.b64	[param0+0], %rd2;
    .param .b64 param1;
    st.param.b64	[param1+0], %rd3;
    .param .b32 retval0;
    call.uni (retval0),
        vprintf,
        (
            param0,
            param1
        );
    ld.param.b32	%r1, [retval0+0];

  }// Callseq End 1
  ret;
}

// .globl	_Z8kernel_1IZ4mainEUlvE0_EvT_
.visible .entry _Z8kernel_1IZ4mainEUlvE0_EvT_(
    .param .align 4 .b8 _Z8kernel_1IZ4mainEUlvE0_EvT__param_0[4]
    )
{
  .local .align 8 .b8 	__local_depot2[8];
  .reg .b64 	%SP;
  .reg .b64 	%SPL;
  .reg .b32 	%r<3>;
  .reg .b64 	%rd<5>;

  mov.u64 	%SPL, __local_depot2;
  cvta.local.u64 	%SP, %SPL;
  ld.param.u32 	%r1, [_Z8kernel_1IZ4mainEUlvE0_EvT__param_0];
  add.u64 	%rd1, %SP, 0;
  add.u64 	%rd2, %SPL, 0;
  st.local.u32 	[%rd2], %r1;
  mov.u64 	%rd3, $str2;
  cvta.global.u64 	%rd4, %rd3;
  // Callseq Start 2
  {
    .reg .b32 temp_param_reg;
    .param .b64 param0;
    st.param.b64	[param0+0], %rd4;
    .param .b64 param1;
    st.param.b64	[param1+0], %rd1;
    .param .b32 retval0;
    call.uni (retval0),
        vprintf,
        (
            param0,
            param1
        );
    ld.param.b32	%r2, [retval0+0];
  }// Callseq End 2
  ret;
}
</pre>

マングリングされた
<ol>
    <li>_Z8kernel_0v</li>
    <li>_Z8kernel_1IZ4mainEUlvE_EvT_</li>
    <li>_Z8kernel_1IZ4mainEUlvE0_EvT_</li>
</ol>
の計3つのカーネル関数が見えます．<br>
すべてinline展開されています．<br>
このコードでのLambda関数の中身はコンパイル時には決定しているものなので普通の__device__関数と扱いは同じなようです．<br>
特筆すべきは3つめのホストのデータをキャプチャするLambda関数でしょうか．<br>
<pre>
.visible .entry _Z8kernel_1IZ4mainEUlvE0_EvT_(
        .param .align 4 .b8 _Z8kernel_1IZ4mainEUlvE0_EvT__param_0[4]
)
...
ld.param.u32    %r1, [_Z8kernel_1IZ4mainEUlvE0_EvT__param_0];
...
st.local.u32    [%rd2], %r1;
...
</pre>
とあるので，カーネル関数の引数で配列としてとしてキャプチャするデータが渡されているようです．<br>

<h3>おまけ 0</h3>
カーネル関数内のprintfですが，出力する文字列は
<pre>.global .align 1 .b8 $str[7] = {112, 111, 105, 32, 48, 10, 0};
.global .align 1 .b8 $str1[7] = {112, 111, 105, 32, 49, 10, 0};
.global .align 1 .b8 $str2[10] = {112, 111, 105, 32, 50, 32, 37, 100, 10, 0};
</pre>
のようにglobalな空間に定義しておいて，%dなどで変数を表示する場合は
<pre>
st.local.u32    [%rd2], %r1;
</pre>
のようにローカルメモリの__local_depot2に配置してvprintfを呼ぶんですね．

<h3>おまけ 1</h3>
C++のstd::functionのCUDA版，nvstd::functionがあるそうです．<br>
詳しくは<a href="https://devblogs.nvidia.com/new-compiler-features-cuda-8/">New Compiler Features in CUDA 8</a>をどうぞ．<br>
余談ですが，NVIDIAはWMMA APIではnvcuda名前空間を使いこちらではnvstd名前空間をつかっているのでちょっぴり統一してほしいです．
