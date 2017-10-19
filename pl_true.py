from logic import *
from utils import *

# define symbols that will be used below
A, B, C, D, E = symbols('A, B, C, D, E')

print("========== Question 1.a ==========")
print("For model m = {A: False, B: True}")
m = {A: False, B: True}
print("Example 1")
s = A | B
result = pl_true_a(s, m)
print("\"s = A | B\" is {}".format(result))
print("Example 2")
s = A | C
result = pl_true_a(s, m)
print("\"s = A | C\" is {}".format(result))
print("Example 3")
s = C | True
result = pl_true_a(s, m)
print("\"s = C | True\" is {}".format(result))


print("========== Question 1.c ==========")

m = {B: False, C: False, D: False, E: False}
s = True | A | B | C | D | E
print("model m = {B: False, C: False, D: False, E: False}, sentence s = True | A | B | C | D | E")
result = pl_true_a_depth(s, m)
print("The result using the original algorithm is {}".format(result))
result = pl_true_d_depth(s, m)
print("The result using the modified algorithm is {}".format(result))


print("========== Question 1.d ==========")
m = {B: True}
print("For model m = {B: True}")
print("It works sometimes.")
print("Example 1")
s = B | A
result = pl_true_d(s, m)
print("\"s = B | A\" is {}".format(result))
print("Example 2")
s = ~B & A
result = pl_true_d(s, m)
print("\"s = ~B & A\" is {}".format(result))
print("Example 3")
s = True | A
result = pl_true_d(s, m)
print("\"s = True | A\" is {}".format(result))

print("It doesn't work sometimes.")
print("Example 1")
s = A | ~A
result = pl_true_d(s, m)
print("\"s = A | ~A\" is {}".format(result))

print("Example 2")
s = A & ~A
result = pl_true_d(s, m)
print("\"s = A & ~A\" is {}".format(result))

print("Example 3")
s = A | ~B | ~A
result = pl_true_d(s, m)
print("\"s = A | ~B | ~A\" is {}".format(result))


print("========== Question 1.e ==========")
print("KB kb = A & B & C, sentence s = E | A & B | D")
kb = A & B & C
s = E | A & B | D
print(tt_entails_a(kb, s))
print(tt_entails_d(kb, s))