'''cubic equation solver'''
import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')

def cubicSolve1(a, b, c, d):
    '''Solve the cubic equation a*x^3 + b*x^2 + c*x + d = 0'''
    if a == 0:
        if b == 0:
            if c == 0:
                if d == 0:
                    return [0]
                else:
                    return []
            else:
                return [-d/c]
        else:
            disc = c**2 - 4*b*d
            if disc < 0:
                return []
            elif disc == 0:
                return [-c/(2*b)]
            else:
                return [(-c + math.sqrt(disc))/(2*b), (-c - math.sqrt(disc))/(2*b)]
    else:
        b /= a
        c /= a
        d /= a
        q = (3*c - b**2)/9
        r = (9*b*c - 27*d - 2*b**3)/54
        disc = q**3 + r**2
        if disc < 0:
            t = math.acos(r/math.sqrt(-q**3))
            return [2*math.sqrt(-q)*math.cos(t/3) - b/3, 2*math.sqrt(-q)*math.cos((t + 2*math.pi)/3) - b/3, 2*math.sqrt(-q)*math.cos((t + 4*math.pi)/3) - b/3]
        elif disc == 0:
            if r == 0:
                return [-2*b/3, -b/3]
            else:
                return [2*math.copysign(math.sqrt(q), r) - b/3, -math.copysign(math.sqrt(q), r) - b/3]
        else:
            s = math.copysign(math.sqrt(disc), r)
            return [math.copysign(math.sqrt(q + s), r) + math.copysign(math.sqrt(q - s), r) - b/3]

def cubicSolve2(a, b, c, d):
    '''Solve the cubic equation a*x^3 + b*x^2 + c*x + d = 0'''
    if a == 0:
        if b == 0:
            if c == 0:
                if d == 0:
                    return [0]
                else:
                    return []
            else:
                return [-d/c]
        else:
            disc = c**2 - 4*b*d
            if disc < 0:
                return []
            elif disc == 0:
                return [-c/(2*b)]
            else:
                return [(-c + math.sqrt(disc))/(2*b), (-c - math.sqrt(disc))/(2*b)]
    else:
        b /= a
        c /= a
        d /= a
        q = (3*c - b**2)/9
        r = (9*b*c - 27*d - 2*b**3)/54
        disc = q**3 + r**2
        if disc < 0:
            t = math.acos(r/math.sqrt(-q**3))
            return [2*math.sqrt(-q)*math.cos(t/3) - b/3, 2*math.sqrt(-q)*math.cos((t + 2*math.pi)/3) - b/3, 2*math.sqrt(-q)*math.cos((t + 4*math.pi)/3) - b/3]
        elif disc == 0:
            if r == 0:
                return [-2*b/3, -b/3]
            else:
                return [2*math.copysign(math.sqrt(q), r) - b/3, -math.copysign(math.sqrt(q), r) - b/3]
        else:
            s = math.copysign(math.sqrt(disc), r)
            return [math.copysign(math.sqrt(q + s), r) + math.copysign(math.sqrt(q - s), r) - b/3]

def cubicSolve3(a, b, c, d):
    '''Solve the cubic equation a*x^3 + b*x^2 + c*x + d = 0'''
    if a == 0:
        if b == 0:
            if c == 0:
                if d == 0:
                    return [0]
                else:
                    return []
            else:
                return [-d/c]
        else:
            disc = c**2 - 4*b*d
            if disc < 0:
                return []
            elif disc == 0:
                return [-c/(2*b)]
            else:
                return [(-c + math.sqrt(disc))/(2*b), (-c - math.sqrt(disc))/(2*b)]
    else:
        b /= a
        c /= a
        d /= a
        q = (3*c - b**2)/9
        r = (9*b*c - 27*d - 2*b**3)/54
        disc = q**3 + r**2
        if disc < 0:
            t = math.acos(r/math.sqrt(-q**3))
            return [2*math.sqrt(-q)*math.cos(t/3) - b/3, 2*math.sqrt(-q)*math.cos((t + 2*math.pi)/3) - b/3, 2*math.sqrt(-q)*math.cos((t + 4*math.pi)/3) - b/3]
        elif disc == 0:
            if r == 0:
                return [-2*b/3, -b/3]
            else:
                return [2*math.copysign(math.sqrt(q), r) - b/3, -math.copysign(math.sqrt(q), r) - b/3]
        else:
            s = math.copysign(math.sqrt(disc), r)
            return [math.copysign(math.sqrt(q + s), r) + math.copysign(math.sqrt(q - s), r) - b/3]

def cubicSolve4(a, b, c, d):
    '''Solve the cubic equation a*x^3 + b*x^2 + c*x + d = 0'''
    if a == 0:
        if b == 0:
            if c == 0:
                if d == 0:
                    return [0]
                else:
                    return []
            else:
                return [-d/c]
        else:
            disc = c**2 - 4*b*d
            if disc < 0:
                return []
            elif disc == 0:
                return [-c/(2*b)]
            else:
                return [(-c + math.sqrt(disc))/(2*b), (-c - math.sqrt(disc))/(2*b)]
    else:
        b /= a
        c /= a
        d /= a
        q = (3*c - b**2)/9
        r = (9*b*c - 27*d - 2*b**3)/54
        disc = q**3 + r**2
        if disc < 0:
            t = math.acos(r/math.sqrt(-q**3))
            return [2*math.sqrt(-q)*math.cos(t/3) - b/3, 2*math.sqrt(-q)*math.cos((t + 2*math.pi)/3) - b/3, 2*math.sqrt(-q)*math.cos((t + 4*math.pi)/3) - b/3]
        elif disc == 0:
            if r == 0:
                return [-2*b/3, -b/3]
            else:
                return [2*math.copysign(math.sqrt(q), r) - b/3, -math.copysign(math.sqrt(q), r) - b/3]
        else:
            s = math.copysign(math.sqrt(disc), r)
            return [math.copysign(math.sqrt(q + s), r) + math.copysign(math.sqrt(q - s), r) - b/3]

def cubicSolve5(a, b, c, d):
    '''Solve the cubic equation a*x^3 + b*x^2 + c*x + d = 0'''
    if a == 0:
        if b == 0:
            if c == 0:
                if d == 0:
                    return [0]
                else:
                    return []
            else:
                return [-d/c]
        else:
            disc = c**2 - 4*b*d
            if disc < 0:
                return []
            elif disc == 0:
                return [-c/(2*b)]
            else:
                return [(-c + math.sqrt(disc))/(2*b), (-c - math.sqrt(disc))/(2*b)]
    else:
        b /= a
        c /= a
        d /= a
        q = (3*c - b**2)/9
        r = (9*b*c - 27*d - 2*b**3)/54
        disc = q**3 + r**2
        if disc < 0:
            t = math.acos(r/math.sqrt(-q**3))
            return [2*math.sqrt(-q)*math.cos(t/3) - b/3, 2*math.sqrt(-q)*math.cos((t + 2*math.pi)/3) - b/3, 2*math.sqrt(-q)*math.cos((t + 4*math.pi)/3) - b/3]
        elif disc == 0:
            if r == 0:
                return [-2*b/3, -b/3]
            else:
                return [2*math.copysign(math.sqrt(q), r) - b/3, -math.copysign(math.sqrt(q), r) - b/3]
        else:
            s = math.copysign(math.sqrt(disc), r)
            return [math.copysign(math.sqrt(q + s), r) + math.copysign(math.sqrt(q - s), r) - b/3]

def main():
    a = 1
    b = 0
    c = -1
    d = 0
    for i in range(1):
        x1 = cubicSolve1(a, b, c, d)
        x2 = cubicSolve2(a, b, c, d)
        x3 = cubicSolve3(a, b, c, d)
        x4 = cubicSolve4(a, b, c, d)
        x5 = cubicSolve5(a, b, c, d)

if __name__ == '__main__':
    main()
