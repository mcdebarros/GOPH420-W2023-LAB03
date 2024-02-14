

xc = 869.565
v1 = 1802.37
v2 = 2792.41



def cross(xc, v1, v2):

    h = ((xc/v1)-(xc/v2)) * ((v1*v2)/(2*(((v2 ** 2)-(v1 ** 2)) ** (1/2))))

    return h

h = cross(xc, v1, v2)

print(h)