# mandatory exercise 1 in MEK4250
# Stokes equation

import numpy as np
import matplotlib.pyplot as plt
from dolfin import * 
set_log_active(False)

def u_exact(k, l):
    return Expression('sin(k*pi*x[0])*cos(l*pi*x[1])', k=k, l=l)


def solver(k, l, deg,n):

    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, 'Lagrange', deg)

    mf = FacetFunction("size_t", mesh)
    mf.set_all(3)

    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], 0) or near(x[0], 1) and on_boundary)

    class Inlet_Outlet(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[1], 0) or near(x[1], 1) and on_boundary)

    walls = Walls()
    walls.mark(mf, 2)
    noslip = DirichletBC(V, Constant(0), mf, 2)

    inlet_outlet = Inlet_Outlet()
    inlet_outlet.mark(mf, 1)
    # neumann allready implemented (flow through)
    #plot(mf, interactive=True)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression('-sin(k*pi*x[0])*cos(l*pi*x[1])*(pow(k*pi, 2) + pow(l*pi, 2))', k=k, l=l)

    a = -inner(grad(u), grad(v))*dx
    L = f*v*dx

    u_ = Function(V)

    solve(a == L, u_, noslip)

    #plot(u_, interactive=True)
    return u_, V, mesh


def task_a():
    import sympy as sp
    x, y, k, l = sp.symbols('x y k l')
    u = sp.sin(k*sp.pi*x)*sp.cos(l*sp.pi*y)
    integral = u**2
    for p in range(1, 2):
        u_ = u
        for i in range(0, p):
            integrand_x = sp.diff(u_, x)
            integrand_y = sp.diff(u_, y)
            u_ = integrand_x + integrand_y
        integral += u_**2
    print integral
    norm = sp.sqrt(sp.integrate(integral, (x,0,1), (y,0,1)))
    print norm

def task_b():
    for deg in [1, 2]:
        print ''
        print 'Polynomial degree:', deg
        for k in [1, 10, 100]:
            for l in [1, 10, 100]:
                print ''
                print 'k =',k,', l =', l
                print '-'*17
                #l = k
                H1norm = np.zeros(4)
                L2norm = np.zeros(4)
                L2rate = np.zeros(3)
                H1rate = np.zeros(3)
                i = 0
                for n in [8, 16, 32, 64]:                
                    #mesh = UnitSquareMesh(n, n)
                    #V = FunctionSpace(mesh, 'Lagrange', order)
                    un, V, mesh = solver(k, l, deg,n)
                    V2 = FunctionSpace(mesh, 'CG', deg+2)
                    ue = u_exact(k, l)
                    ue = interpolate(ue, V2)
                    #ue = interpolate(Expression('sin(k*pi*x[0])*cos(l*pi*x[1])', k=k, l=l), V)
                    #u = Function(V)
                    H1norm[i] = errornorm(ue, un, 'h1')
                    L2norm[i] = errornorm(ue, un, 'L2')
                    
                    print 'n = %g, L2 errornorm: %g, H1 errornorm: %g' %(n, L2norm[i], H1norm[i])
                    
                    if n > 8:
                        n1 = n/2.0
                        L2rate[i-1] = -(np.log(L2norm[i]/L2norm[i-1])/np.log(float(n)/n1))
                        H1rate[i-1] = -(np.log(H1norm[i]/H1norm[i-1])/np.log(float(n)/n1))
                    i += 1
                print 'Convergence rate:'
                print 'L2: ',L2rate[:]
                print 'H1: ',H1rate[:]

def task_c():
    for normdeg in [1, 2]:
        if normdeg == 1:
            print 'Error norm L2'
        else:
            print 'Error norm H1'
        for deg in [1, 2]:
            for k in [1]:
                for l in [1]:
                    N = [4, 8, 16, 32, 64, 128]
                    a = np.zeros(len(N))
                    y = np.zeros(len(N))
                    for n in range(len(N)):
                        u_num, V, mesh = solver(k, l, deg, N[n])
                        V2 = FunctionSpace(mesh, 'CG', deg+2)
                        u_ex = u_exact(k ,l)
                        u_ex = interpolate(u_ex, V2)
                        #plot(u_ex)
                        #plot(u_num)
                        #interactive()
                        if normdeg == 1:
                            A = errornorm(u_ex, u_num, 'l2')
                            #B = 0.5*((-k*np.pi)**2 + (-l*pi)**2)
                            #B = norm(u_ex, 'h1')
                        else:
                            A = errornorm(u_ex, u_num, 'h1')
                            #B = 0
                            #for i in range(normdeg):
                            #    B += ((k*np.pi)**2 + (l*np.pi)**2)**i
                        y[n] = np.log(A)#/B)
                        a[n] = np.log(mesh.hmin())#np.log(1./(N[n]))

            #plt.plot(a, y, 'o', label='points')

            # solve system Ax = b
            Am = np.zeros([2, 2])
            b = np.zeros(2)
            Am[0, 0] = len(N)
            for i in range(len(N)):            
                Am[0, 1] += a[i]
                Am[1, 0] += a[i]
                Am[1, 1] += a[i]**2
                b[0] += y[i]
                b[1] += a[i]*y[i]
            logC, alfa = np.linalg.solve(Am, b)
            print 'Polynomial order:',deg
            print 'alpha:',alfa, ', C:',np.exp(logC)
            f = lambda x: logC + alfa*x
            x = np.linspace(a[0], a[-1], 101)
            #plt.plot(x, f(x), label='lsq approximation')
            #plt.legend(loc='upper left')
            #plt.grid('on')
            #plt.xlabel('log(h)'), plt.ylabel('log(||u - uh||)')
            #plt.savefig('lsq_normdeg' +str(normdeg)+  '_polydeg' +str(deg)+ '.png')
            #plt.show()



#task_a()
#task_b()   
#task_c()
 


