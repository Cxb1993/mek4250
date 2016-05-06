from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
set_log_active(False)

N = [2, 4, 8, 16, 32, 64]
u_order = [4, 4, 3, 3]
p_order = [3, 2, 2, 1]

table = open('tables.txt', 'w')
for o in range(len(u_order)):
    table.write('\\begin{table}\n')
    table.write('\\begin{tabular}{|c|c|c|c|c|c|c|}\n')   
    table.write('\\cline{1-7}\n')
    table.write('\\ N & 2 & 4 & 8 & 16 & 32 & 64\\\\\n')
    table.write('\\cline{1-7}\n')
    error_u = np.zeros(len(N))
    error_p = np.zeros(len(N))
    h = np.zeros(len(N))
    print ''
    print 'P%s-P%s' %(u_order[o], p_order[o])
    print '-'*80
    for i in range(len(N)):
        mesh = UnitSquareMesh(N[i], N[i])

        V = VectorFunctionSpace(mesh, 'CG', u_order[o])
        V2 = VectorFunctionSpace(mesh, 'CG', u_order[o]+1)
        Q = FunctionSpace(mesh, 'CG', p_order[o])
        Q2 = FunctionSpace(mesh, 'CG', p_order[o]+1)

        VQ = MixedFunctionSpace([V, Q])

        class Left(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0) and on_boundary

        class velocity_boundaries(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < DOLFIN_EPS or x[1] > 1 - DOLFIN_EPS or x[1] < DOLFIN_EPS

        boundaries = FacetFunction('size_t', mesh)
        boundaries.set_all(0)

        vel_bound = velocity_boundaries()
        vel_bound.mark(boundaries, 1)
        #plot(boundaries)

        u_exact = Expression(('sin(pi*x[1])', 'cos(pi*x[0])'))
        p_exact = Expression('sin(2*pi*x[0])')

        f = Expression(('pi*pi*sin(pi*x[1]) - 2*pi*cos(2*pi*x[0])', 'pi*pi*cos(pi*x[0])'))


        noslip = DirichletBC(VQ.sub(0), u_exact, boundaries, 1)
        bc = [noslip]


        up = TrialFunction(VQ)
        u, p = split(up)
        vq = TestFunction(VQ)
        v, q = split(vq)

        eq1 = inner(grad(u), grad(v))*dx + inner(p, div(v))*dx - inner(f,v)*dx
        eq2 = inner(div(u), q)*dx

        equation = eq1 + eq2

        up_ = Function(VQ)

        solve(lhs(equation) == rhs(equation), up_, bc)

        u_, p_ = up_.split()

        u_exact = interpolate(u_exact, V2)
        p_exact = interpolate(p_exact, Q2)

        error_u[i] = errornorm(u_exact, u_, 'h1')
        error_p[i] = errornorm(p_exact, p_)
        h[i] = 1./N[i]

    table.write('\\ H1 error\t')
    for i in range(len(error_u)):
        if i == len(error_u):
            table.write('& %.4g\\\\\n' %error_u[i])
        else:
            table.write('& %.4g\t' %error_u[i])
    table.write('\\cline{1-7}\n')
    table.write('\\ Convergence rate\t')
    print 'Velocity:'
    for i in range(len(N)):
        convergence = 0
        if i == 0:
            print 'N =',N[i],', H1 error =',error_u[i]
            table.write('& -\t')
        if i > 0:
            convergence = np.log(abs(error_u[i]/error_u[i-1])) / np.log(abs(h[i] / h[i-1]))
            print 'N =',N[i],', H1 error =',error_u[i],', convergence rate =', convergence
            if i == len(N):
                table.write('& %.4g\\\\\n' %convergence)
            else:
                table.write('& %.4g\t' %convergence)
    
    table.write('\\hline \\hline\n')
    table.write('\\ L2 error\t')
    for i in range(len(error_p)):
        if i == len(error_p):
            table.write('& %.4g\\\\\n' %error_p[i])
        else:
            table.write('& %.4g\t' %error_p[i])
    table.write('\\cline{1-7}')
    table.write('\\ Convergence rate\t')
    print ''
    print 'Pressure:'
    for i in range(len(N)):
        convergence = 0
        if i == 0:
            print 'N =',N[i],', L2 error =',error_p[i]
            table.write('& -\t') 
        if i > 0:
            convergence = np.log(abs(error_p[i]/error_p[i-1])) / np.log(abs(h[i] / h[i-1]))
            print 'N =',N[i],', L2 error =',error_p[i],', convergence rate =', convergence
            if i == len(N):
                table.write('& %.4g\\\\\n' %convergence)
            else:
                table.write('& %.4g\t' %convergence)
    table.write('\\cline{1-7}\n')
    table.write('\\end{tabular}\n')
    table.write('\\label{tab:insert label name}\n')
    table.write('\\end{table}')
    table.write('\n')
    plt.figure(1)
    plt.loglog(h, error_u, label='P%s-P%s'%(u_order[o] ,p_order[o]))
    plt.title('H1 error for u')# with P%s-P%s elements' %(u_order[o] ,p_order[o]))
    plt.xlabel('h'), plt.ylabel('error')
    plt.grid('on')
    #plt.savefig('results/velocity_P%s_P%s.png'%(u_order[o] ,p_order[o]))
    plt.legend(loc='lower right')
    plt.figure(2)
    plt.loglog(h, error_p, label='P%s-P%s'%(u_order[o] ,p_order[o]))
    plt.title('L2 error for p')# with P%s-P%s elements' %(u_order[o] ,p_order[o]))
    plt.xlabel('h'), plt.ylabel('error')
    plt.grid('on')
    #plt.savefig('results/pressure_P%s_P%s.png'%(u_order[o] ,p_order[o]))
    plt.legend(loc='lower right')
#plt.xlabel('h'), plt.ylabel('error')
table.close()
plt.show()



