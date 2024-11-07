# Labo 5

`Beelddegradatie`
![](img/Beelddegradatie.png)
- $f(x,y)$: beeld
- $H$: degradatiefunctie
- $n(u,v)$: noise
- $\hat{f}(x,y)$: gedegradeerd beeld
- $g(x,y)$: gereconstrueerd beeld

`Invers filter`
$$ \hat{F}(u,v) = \frac{G(u,v)}{F(u,v)} $$
- $G(u,v) = H(u,v)F(u,v)+N(u,v)$

`Wiener filter`
$$ \hat{F}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2+k}G(u,v) $$
- $|H(f)|^2=H(u,v)H^*(u,v)$: 
- $H^*$: complex toegevoegde van $H$