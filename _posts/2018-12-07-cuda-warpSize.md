---
layout: post
title:  "CUDAのwarpSizeについて"
date:   2018-12-07 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2>Warpとは</h2>
<p>NVIDIAのGPUでの並列処理はWarpと呼ばれるスレッド単位で行われます．
Warp shuffleやWMMA API*のようにWarpで協調して計算を行うAPIも提供されており，1 Warpが何Threadsなのかを取得したくなることは多々あります．
今出回っているGPUでは大体32 Threadsなので決め打ちで32というマジックナンバーを使っているプログラムも多々見ますが，CUDAではwarpSizeという変数が提供されており，Warpサイズを取得することができます．</p>

<h2>warpSizeの不満点</h2>
<p>上述したとおりWarpサイズは大体32 Threadsなのですが，warpSizeはconstexprではないため，コンパイル時計算等には使えません．
コンパイル時に計算をしたいなら自分で
{% highlight cuda %}
constexpr std::size_t warp_size = 32;
{% endhighlight %}
みたいなことをすることになります．
これはGPUのアーキに関する値なので，もちろんhost側でも使えません．
32と決め打ちせず，host側でちゃんと取得したければ
{% highlight cuda %}
int device, warp_size;
cudaGetDevice(&device);
cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device);
{% endhighlight %}
</p>
で取れます．

<h2>warpSizeはコンパイルされるとどうなるか</h2>
<p>
例えば以下のようなコードをVolta用にコンパイルしてPTXを見てみます．
{% highlight cuda %}
__global__ void kernel(int* const ptr) {
  *ptr = warpSize;
}
{% endhighlight %}

すると，このようにWARP_SZという値指定子に置き換わります．

<pre>
.visible .entry _Z6kernelPi(
.param .u64 _Z6kernelPi_param_0
)
{
.reg .b32 %r<2>;
.reg .b64 %rd<3>;
ld.param.u64 %rd1, [_Z6kernelPi_param_0];
cvta.to.global.u64 %rd2, %rd1;
mov.u32 %r1, WARP_SZ;
st.global.u32 [%rd2], %r1;
ret;
}
</pre>
ではSASSではどうなるでしょうか．
<pre>
		Function : _Z6kernelPi
	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM70 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM70)"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;  /* 0x00000a00ff017624 */
                                                                            /* 0x000fe400078e00ff */
        /*0010*/              @!PT SHFL.IDX PT, RZ, RZ, RZ, RZ ;            /* 0x000000fffffff389 */
                                                                            /* 0x000fe200000e00ff */
        /*0020*/                   IMAD.MOV.U32 R5, RZ, RZ, 0x20 ;          /* 0x00000020ff057424 */
                                                                            /* 0x000fe200078e00ff */
        /*0030*/                   MOV R2, c[0x0][0x160] ;                  /* 0x0000580000027a02 */
                                                                            /* 0x000fe40000000f00 */
        /*0040*/                   MOV R3, c[0x0][0x164] ;                  /* 0x0000590000037a02 */
                                                                            /* 0x000fd00000000f00 */
        /*0050*/                   STG.E.SYS [R2], R5 ;                     /* 0x0000000502007386 */
                                                                            /* 0x000fe2000010e900 */
        /*0060*/                   EXIT ;                                   /* 0x000000000000794d */
                                                                            /* 0x000fea0003800000 */
        /*0070*/                   BRA 0x70;                                /* 0xfffffff000007947 */
                                                                            /* 0x000fc0000383ffff */
</pre>


0x20( = 32)に置き換わりました．<br>
このようにSASSで初めて即値が入ります．<br>
PTXは前方互換性のためのレイヤーなので，実際にDeviceで実行されるアーキ用のSASSで即値となるのはうなずけます．
</p>
