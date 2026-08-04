
def x(a, b = 3):
    if b is None:
        b=3
    return a + b

print(x(3,None))