---
layout: post
title:  "Occupancyを可視化する"
date:   2021-08-08 01:08:41 +0900
categories: cuda
---

{% include header.markdown %}

<a href="https://adventar.org/calendars/10896">消えたCUDA関連の旧ブログ記事を復元するひとり Advent Calendar 2024</a>の記事です。

<h2 id='a'>なんの話か</h2>
<p>
Fortran/OpenACCのループ内サブルーチンで可変長配列を用いた場合、実体はどこに保管されるのかというのを調べてみました。<br>
コンパイル時に配列長が決定していないためレジスタに作られることはないだろうとの予想のもと、カーネル内mallocを呼んでいるのかなーと言う気持ちで調べていきます。<br>
このブログ初のFortranについての記事です。
</p>

<h2>実験コード</h2>
<p>
調査に用いたコードはこちらです。<br>
inline展開されると困るので、2ファイルに分けて書いています。
</p>

{% highlight f90 %}
subroutine sub_routine(a, array_size)
  !$acc routine seq
  real(8) :: a(:)
  integer :: array_size

  real(8), dimension(array_size) :: array
  integer :: i

  do i = 1, array_size
    a(i) = array(i)
  end do
end subroutine
{% endhighlight %}

<p>
（Makefile等はこちら <a href='https://github.com/enp1s0/openacc-subroutine'>enp1s0/openacc-subroutine - GitHub</a>）
</p>
<p>
2ファイル用意したものの、実際にはsub.f90があれば調査できます。<br>
コンパイルしたsub.o内に含まれるPTXをcuobjdumpで覗きます。
</p>

{% highlight f90 %}
!...

$L__tmp0:
.loc	1 1 1
ld.u64 %rd2, [%rd22+56];
ld.u64 %rd3, [%rd22+80];
add.s64 %rd24, %rd2, %rd3;
add.s64 %rd44, %rd24, -1;
ld.u32 %r1, [%rd21];
mul.wide.s32 %rd25, %r1, 8;
{
	.reg .b32 temp_param_reg;
.param .b64 param0;
st.param.b64 [param0+0], %rd25;
.param .b64 retval0;
call.uni (retval0),
malloc,
(
param0
);
ld.param.b64 %rd5, [retval0+0];
}
	setp.ne.s64 %p1, %rd5, 0;
@%p1 bra $L__BB0_4;

!...
{% endhighlight %}

<p>
予想通りmallocが呼ばれていることが確認できます。<br>
つまり可変長配列の実体はDeviceメモリ上です。<br>
では、可変長配列をサイズ決め打ちの固定長にするとどうなるかと言うと、
</p>
<ul>
  <li>サイズが小さいときはレジスタ</li>
  <li>サイズが多いときは同様にカーネル内malloc</li>
</ul>
<p>
となることが確認できます。
</p>

<h2 id='c'>おわり</h2>
<p>
カーネル内mallocというものを知った時誰が使うのだろうと思っていましたが、OpenACC/Fortranの実装で使われていたのですね。
