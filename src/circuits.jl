module Gates
"""
Minimal definition of circuit components provided for testing convenience. 
QXZoo contains more comprehensive gate set and features
"""

h() = [[1., 1.] [1., -1.]]./sqrt(2)
x() = [[0., 1.] [1., 0.]]
z() = [[1., 0.] [0., -1.]]
I() = [[1., 0.] [0., 1.]]
cx() = [[1., 0., 0., 0.] [0., 1., 0., 0.] [0., 0., 0., 1.] [0., 0., 1., 0.]]
cz() = [[1., 0., 0., 0.] [0., 1., 0., 0.] [0., 0., 1., 0.] [0., 0., 0., -1.]]
swap() = [[1., 0., 0., 0.] [0., 0., 1., 0.] [0., 1., 0., 0.] [0., 0., 0., 1.]]

end