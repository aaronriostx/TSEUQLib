import pyoti.sparse as oti

z = 1 + oti.e(1, order=1)

t = z+1

y = t**z
y = oti.exp( (z+0.5) * log(t) )
print(y)
