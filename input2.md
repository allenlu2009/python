---
title: 二次式和正定矩陣 Quadratic Form and Positive Definite Matrix
date: 2023-01-28 23:10:08
categories:
- Language
tags: [Graph, Laplacian]
typora-root-url: ../../allenlu2009.github.io

---



## Main Reference

[二次型與正定矩陣 | 線代啟示錄 (wordpress.com)](https://ccjou.wordpress.com/2009/10/21/二次型與正定矩陣/)http://www-personal.umich.edu/~mmustata/Slides_Lecture13_565.pdf)



## 二次式

二次式大概是數學和物理最重要的形式。不論是求解、極值（最大或最小）、非綫性、簡諧運動等等都和二次式相關。矩陣也不例外。

有了二次式就可以定義正定矩陣，Rayleigh Quotient,  Courant-Fischer Theorem, etc.   二次式和 eigenvalues 直接相關。



令 $A=\left[a_{i j}\right]$ 為一個 $n \times n$ 實矩陣, $\mathbf{x}=\left[\begin{array}{c}x_1 \\ \vdots \\ x_n\end{array}\right]$ 為 $n$ 維實向量, 具有以下形式的實函數稱為 二次型 (quadratic form) :
$$
f(\mathbf{x})=\mathbf{x}^T A \mathbf{x} 。
$$
* 注意：二次型 $\mathbf{x}^T A \mathbf{x}$ 是一個純量。
* 任意二次型 $\mathbf{x}^T A \mathbf{x}$ 都可以轉換為等價的 $\mathbf{x}^T B \mathbf{x}$, 其中 $B$ 是一個**實對稱矩陣**：$B=\frac{1}{2}\left(A+A^T\right)$
  * 利用一點運算技巧改寫矩陣乘法公式可得


$$
\begin{aligned}
\mathbf{x}^T A \mathbf{x} & =\sum_{i=1}^n \sum_{j=1}^n a_{i j} x_i x_j \\
& =\sum_{i=1}^n \sum_{j=1}^n \frac{1}{2}\left(a_{i j}+a_{j i}\right) x_i x_j \\
& =\mathbf{x}^T\left[\frac{1}{2}\left(A+A^T\right)\right] \mathbf{x}
\end{aligned}
$$
* 矩陣 $A$ 與 $B=\frac{1}{2}\left(A+A^T\right)$ 有相等的二次型, 不難驗證 $\frac{1}{2}\left(A+A^T\right)$ 是對稱的。例如,

$$
\left[\begin{array}{ll}
x & y
\end{array}\right]\left[\begin{array}{ll}
5 & 4 \\
2 & 7
\end{array}\right]\left[\begin{array}{l}
x \\
y
\end{array}\right]=5 x^2+6 x y+7 y^2=\left[\begin{array}{ll}
x & y
\end{array}\right]\left[\begin{array}{ll}
5 & 3 \\
3 & 7
\end{array}\right]\left[\begin{array}{l}
x \\
y
\end{array}\right] 。
$$



## 正定矩陣

### 定義

令 $A$為一個 $n\times n$ 階實對稱矩陣。若每一 $n$ 維非零實向量 $\mathbf{x}$ 皆使得 $\mathbf{x}^TA\mathbf{x}>0$，我們稱 $A$ 為正定 (positive definite)；若將上述條件放鬆為 $\mathbf{x}^TA\mathbf{x}\ge 0$，則 $A$ 稱為半正定 (positive semidefinite)。如果 $\mathbf{x}^TA\mathbf{x}$可能是正值也可能是負值，則稱 $A$ 是未定的 (indefinite)。

傳統上，我們習慣將對稱性納入正定矩陣的定義，一方面因為實對稱正定矩陣擁有美好的性質，另一個原因是實對稱正定矩陣的分析就足以應付其他一般的正定矩陣。

* 任意二次型 $\mathbf{x}^T A \mathbf{x}$ 都可以轉換為等價的 $\mathbf{x}^T B \mathbf{x}$, 其中 $B$ 是一個**實對稱矩陣**：$B=\frac{1}{2}\left(A+A^T\right)$
* 如果 $A$ 是正定或半正定矩陣，則實對稱矩陣 $B$ 也是正定或半正定矩陣。
* 如何判斷 $A$ 是正定或半正定矩陣？顯然不可能試所有的 $\mathbf{x}^TA\mathbf{x} > 0$.   
  * 最直接的方法就是看 eigenvalues.  如果所有 eigenvalues 都大於 0, 為正定矩陣。如果所有 eigenvalues 都大於等於 0, 為半正定矩陣。
  * 注意：如果 $A$ 不是對稱矩陣，eigenvalues 有可能是複數。此時判斷 $B = \frac{1}{2}(A+A^T)$  的 eigenvalues.  因爲 $B$ 是對稱矩陣，所有 eigenvalues 一定都是實數。
  * 證明：假設 $A$ 是對稱矩陣，$A = Q D Q^T \to \mathbf{x}^T A \mathbf{x} = \mathbf{x}^T Q D Q^T \mathbf{x} = \mathbf{z}^T  D  \mathbf{z} = \lambda_1 z_1^2 + ... + \lambda_n y_n^2$
    * 如果 $\lambda_k > 0, \, \mathbf{x}^T A \mathbf{x} > 0 \to A $ 是正定矩陣
    * 如果 $\lambda_k \ge 0, \, \mathbf{x}^T A \mathbf{x} \ge 0 \to A $ 是半正定矩陣

### 幾何意義

考慮 ![n=1](https://s0.wp.com/latex.php?latex=n%3D1&bg=ffffff&fg=000000&s=0&c=20201002) 的情況，矩陣 ![A](https://s0.wp.com/latex.php?latex=A&bg=ffffff&fg=000000&s=0&c=20201002) 和向量 ![\mathbf{x}](https://s0.wp.com/latex.php?latex=%5Cmathbf%7Bx%7D&bg=ffffff&fg=000000&s=0&c=20201002) 分別退化為純量 ![a](https://s0.wp.com/latex.php?latex=a&bg=ffffff&fg=000000&s=0&c=20201002) 和 ![x](https://s0.wp.com/latex.php?latex=x&bg=ffffff&fg=000000&s=0&c=20201002)，如果對任意非零 ![x](https://s0.wp.com/latex.php?latex=x&bg=ffffff&fg=000000&s=0&c=20201002) 都有

![xax=ax^2>0](https://s0.wp.com/latex.php?latex=xax%3Dax%5E2%3E0&bg=ffffff&fg=000000&s=0&c=20201002)。

我們說 ![a](https://s0.wp.com/latex.php?latex=a&bg=ffffff&fg=000000&s=0&c=20201002) 是正定的，或簡潔地說 ![a](https://s0.wp.com/latex.php?latex=a&bg=ffffff&fg=000000&s=0&c=20201002) 是正的 (![a>0](https://s0.wp.com/latex.php?latex=a%3E0&bg=ffffff&fg=000000&s=0&c=20201002))，則 ![ax](https://s0.wp.com/latex.php?latex=ax&bg=ffffff&fg=000000&s=0&c=20201002) 與 ![x](https://s0.wp.com/latex.php?latex=x&bg=ffffff&fg=000000&s=0&c=20201002) 有相同的正負號。當 ![n>1](https://s0.wp.com/latex.php?latex=n%3E1&bg=ffffff&fg=000000&s=0&c=20201002) 時，令 ![\theta](https://s0.wp.com/latex.php?latex=%5Ctheta&bg=ffffff&fg=000000&s=0&c=20201002) 為 ![A\mathbf{x}](https://s0.wp.com/latex.php?latex=A%5Cmathbf%7Bx%7D&bg=ffffff&fg=000000&s=0&c=20201002) 與 ![\mathbf{x}](https://s0.wp.com/latex.php?latex=%5Cmathbf%7Bx%7D&bg=ffffff&fg=000000&s=0&c=20201002) 的夾角，此夾角的餘弦為

![\cos\theta=\displaystyle\frac{\mathbf{x}^T(A\mathbf{x})}{\Vert\mathbf{x}\Vert~\Vert A\mathbf{x}\Vert}](https://s0.wp.com/latex.php?latex=%5Ccos%5Ctheta%3D%5Cdisplaystyle%5Cfrac%7B%5Cmathbf%7Bx%7D%5ET%28A%5Cmathbf%7Bx%7D%29%7D%7B%5CVert%5Cmathbf%7Bx%7D%5CVert%7E%5CVert+A%5Cmathbf%7Bx%7D%5CVert%7D&bg=ffffff&fg=000000&s=0&c=20201002)。

上式中，![A\mathbf{x}](https://s0.wp.com/latex.php?latex=A%5Cmathbf%7Bx%7D&bg=ffffff&fg=000000&s=0&c=20201002) 與 ![\mathbf{x}](https://s0.wp.com/latex.php?latex=%5Cmathbf%7Bx%7D&bg=ffffff&fg=000000&s=0&c=20201002) 的內積為正值表示經線性變換後的向量 ![A\mathbf{x}](https://s0.wp.com/latex.php?latex=A%5Cmathbf%7Bx%7D&bg=ffffff&fg=000000&s=0&c=20201002) 與原向量 ![\mathbf{x}](https://s0.wp.com/latex.php?latex=%5Cmathbf%7Bx%7D&bg=ffffff&fg=000000&s=0&c=20201002) 的夾角小於 ![90^{\circ}](https://s0.wp.com/latex.php?latex=90%5E%7B%5Ccirc%7D&bg=ffffff&fg=000000&s=0&c=20201002)。見下圖，![\mathbf{x}](https://s0.wp.com/latex.php?latex=%5Cmathbf%7Bx%7D&bg=ffffff&fg=000000&s=0&c=20201002) 為超平面 ![P](https://s0.wp.com/latex.php?latex=P&bg=ffffff&fg=000000&s=0&c=20201002) 的法向量，正定矩陣 ![A](https://s0.wp.com/latex.php?latex=A&bg=ffffff&fg=000000&s=0&c=20201002) 保證變換後的向量 ![A\mathbf{x}](https://s0.wp.com/latex.php?latex=A%5Cmathbf%7Bx%7D&bg=ffffff&fg=000000&s=0&c=20201002) 與原向量 ![\mathbf{x}](https://s0.wp.com/latex.php?latex=%5Cmathbf%7Bx%7D&bg=ffffff&fg=000000&s=0&c=20201002) 都位於超平面 ![P](https://s0.wp.com/latex.php?latex=P&bg=ffffff&fg=000000&s=0&c=20201002) 的同一側。
