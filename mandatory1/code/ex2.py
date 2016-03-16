# Mandatory assignment 1
# exercise 2

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
set_log_active(False)


def BC(mesh, V):
    mf = FacetFunction("size_t", mesh)
    mf.set_all(0)
    
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0) and on_boundary
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1) and on_boundary
    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0) and on_boundary
    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1) and on_boundary

    left = Left() 
    left.mark(mf, 1)
    right = Right()
    right.mark(mf, 2)
    bottom = Bottom()
    bottom.mark(mf, 3)
    top = Top()
    top.mark(mf, 4)

    #plot(mf,interactive=True)
    leftBC = DirichletBC(V, Constant(0), mf, 1)
    rightBC = DirichletBC(V, Constant(1), mf, 2)

    BCs = [leftBC, rightBC]
    return BCs


def numerical(mu, n, deg):
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, 'CG', deg)

    BCs = BC(mesh, V)

    u = TrialFunction(V)
    v = TestFunction(V)

    eq = mu*inner(grad(u), grad(v))*dx + u.dx(0)*v*dx

    u_ = Function(V)
    
    solve(lhs(eq) == rhs(eq), u_, BCs)

    return u_, V, mesh
   

def exact(mu):#, n, deg):
    #mesh = UnitSquareMesh(n, n)
    #V = FunctionSpace(mesh, 'CG', deg)
    
    #BCs = BC(mesh, V)

    exact = Expression('(exp(x[0]/mu) - 1) / (exp(1/mu) - 1)', mu=mu)
    #u = interpolate(exact, V)

    return exact


def task_b():
    mu = [1, 0.1, 0.01, 0.002]
    N = [8, 16, 32, 64]
    deg = 1
    error = np.zeros([len(mu),len(N)])
    for m in range(len(mu)):
        rate = np.zeros(len(N)-1)
        for n in range(len(N)):
            un, V, mesh = numerical(mu[m], N[n], deg)
            ue = exact(mu[m])
            V2 = FunctionSpace(mesh, 'CG', deg+2)
            ue = interpolate(ue, V2)
            error[m, n] = errornorm(ue, un)
            if n > 0:
                rate[n-1] = -np.log(error[m,n]/error[m,n-1]) / np.log(N[n]/N[n-1])
            #plotname = 'u_numerical_mu'+str(m)
            #wiz = plot(un, interactive=False)
            #wiz.write_png(plotname)
        print rate
    print error


def task_c():
    for normdeg in [1, 2]:
        if normdeg == 1:
            print 'Error norm L2'
        else:
            print 'Error norm H1'
        for deg in [1, 2]:
            print 'Polynomial order:',deg
            for mu in [1, 0.1, 0.01, 0.002]:
                N = [4, 8, 16, 32, 64, 128]
                a = np.zeros(len(N))
                y = np.zeros(len(N))
                for n in range(len(N)):
                    u_num, V, mesh = numerical(mu, N[n], deg)
                    V2 = FunctionSpace(mesh, 'CG', deg+2)
                    u_ex = exact(mu)
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
                #print 'Polynomial order:',deg
                print 'mu:',mu
                print 'alpha:',alfa, ', C:',np.exp(logC)
                f = lambda x: logC + alfa*x
                x = np.linspace(a[0], a[-1], 101)
                #plt.plot(x, f(x), label='lsq approximation')
                #plt.legend(loc='upper left')
                #plt.grid('on')
                #plt.xlabel('log(h)'), plt.ylabel('log(||u - uh||)')
                #plt.savefig('lsq_normdeg' +str(normdeg)+  '_polydeg' +str(deg)+ '_mu' +str(mu)+ '.png')
                #plt.show()


def SUPG(mu, n, deg):
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, 'CG', deg)
    
    BCs = BC(mesh, V)
    
    u = TrialFunction(V)
    v = TestFunction(V)
    
    beta = 0.5*mesh.hmin()

    v = v + beta*v.dx(0)

    eq = mu*inner(grad(u), grad(v))*dx + u.dx(0)*v*dx

    u_ = Function(V)

    solve(lhs(eq) == rhs(eq), u_, BCs)
    #plot(u_, interactive=True)
    return u_, mesh, V

def task_d():
    # numerical error
    for deg in [1]:  
        print ''
        print 'Polynomial degree:',deg
        print '------------------------'
        for mu in [1.0, 0.1, 0.01, 0.002]:
            print ''
            print 'mu:', mu
            N = [8, 16, 32, 64]
            errorL2 = np.zeros(len(N)) 
            errorH1 = np.zeros(len(N))
            rateL2 = np.zeros(len(N)-1)
            rateH1 = np.zeros(len(N)-1)   
            for n in range(len(N)):
                unum, mesh, V = SUPG(mu, N[n], deg)
                V2 = FunctionSpace(mesh, 'CG', deg+2)
                uex = exact(mu)
                uex = interpolate(uex, V2)
                errorL2[n] = errornorm(uex, unum)
                errorH1[n] = errornorm(uex, unum, 'h1')
                if n == 0:
                    print 'N:',N[n],', L2 error:', errorL2[n], ', H1 error:',errorH1[n]
                #plot(unum, interactive=True)
                elif n > 0:
                    print 'N:',N[n],', L2 error:', errorL2[n], ', H1 error:',errorH1[n]
                    rateL2[n-1] = -np.log(errorL2[n]/errorL2[n-1]) / np.log(N[n]/N[n-1])
                    rateH1[n-1] = -np.log(errorH1[n]/errorH1[n-1]) / np.log(N[n]/N[n-1])
                #plotname = '1D_deg_'+str(deg)+'_mu'+str(n)
                #wiz = plot(unum, interactive=False)
                #wiz.write_png(plotname)
            print 'L2 convergence rate:', rateL2
            print 'H1 convergence rate:', rateH1

#task_b()
#task_c()
#task_d()
#SUPG(0.002, 16, 1)


