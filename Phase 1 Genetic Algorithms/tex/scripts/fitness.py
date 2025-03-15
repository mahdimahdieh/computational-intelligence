chromosome = input()
a, b, c, d, e, f, g, h = [int(char) for char in chromosome]
fitness = a + b - c - d + e + f - g - h
print(fitness)