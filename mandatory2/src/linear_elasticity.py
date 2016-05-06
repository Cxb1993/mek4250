from dolfin import *
import numpy as np

set_log_level(ERROR)


mu = Constant(1.0)
#lmbda = Constant(1.0)


def functionspace(N, Lambda, Degree):
    print 'Single functionspace'
    print '--------------------'
    print 'numerical error for:' 
    for deg in Degree:
        table = open('table%s.txt'%deg, 'w')
        table.write('\\begin{figure}\n')
        table.write('\\begin{tabular}{|c|c|c|c|c|c|}\n')   
        table.write('\\cline{2-5} & \multicolumn{4}{|c|}{Numerical error} \\\\\n') 
        table.write('\\cline{1-6}\n')
        table.write('\\diagbox[width=4em]{$\lambda$}{N} & {8} & {16} & {32} & {64} & {Convergence rate} \\\\\n')
        table.write('\\cline{1-6}\n')
        print ''
        print 'Degree of polynomial:',deg
        for lmbda in Lambda:
            table.write('\\multicolumn{1}{|c|}{%g}\t'%lmbda)
            error = np.zeros(len(N))
            h = np.zeros(len(N))
            for n in range(len(N)):
                mesh = UnitSquareMesh(N[n], N[n])

                V = VectorFunctionSpace(mesh, 'CG', deg)
                V2 = VectorFunctionSpace(mesh, 'CG', deg+1)

                f = Expression(('mu*pi*pi* (2*x[1]*sin(pi*x[0]*x[1]) + pi*x[0]*(x[1]*x[1]*cos(pi*x[0]*x[1]) + x[0]*x[0]*cos(pi*x[0]*x[1])))',
                                '-mu*pi*pi* (2*x[0]*sin(pi*x[0]*x[1]) + pi*x[1]*(x[0]*x[0]*cos(pi*x[0]*x[1]) + x[1]*x[1]*cos(pi*x[0]*x[1])))'),mu= mu)

                ue = Expression(('pi*x[0]*cos(pi*x[0]*x[1])', '-pi*x[1]*cos(pi*x[0]*x[1])'))
                ue = interpolate(ue, V2)
                uE = project(ue, V)
                class Boundaries(SubDomain):
                    def inside(self, x, on_boundary):
                        return on_boundary

                boundaries = Boundaries()

                bc = DirichletBC(V, ue, boundaries)

                mf = FacetFunction('size_t', mesh)
                mf.set_all(0)
                boundaries.mark(mf, 2)
                #plot(mf, interactive=True)

                u = TrialFunction(V)
                v = TestFunction(V)

                a = mu*inner(grad(u), grad(v))*dx + lmbda*inner(div(u), div(v))*dx
                L = dot(f, v)*dx

                u_ = Function(V)
                solve(a == L, u_, bc)
                
                error[n] = errornorm(u_, ue)#, norm_type='l2')
                h[n] = 1./N[n]
                table.write(' & {%g}\t'%error[n])
                if n == 0:
                    print 'Lambda=',lmbda,', n=',N[n],', error=',error[n]
                else:
                    convergence_rate = np.log(abs(error[n]/error[n-1]))/np.log(abs(h[n] / h[n-1]))
                    print 'Lambda=',lmbda,', n=',N[n],', error=',error[n],', convergence rate:',convergence_rate
            table.write('& {%g} \\\\\n'%convergence_rate)
            table.write('\\cline{1-6}\n')
            """
            uu = File('results/locking_velocity_numerical.pvd')
            uue = File('results/locking_velocity_exact.pvd')
            uu << u_
            uue << uE
            """
            #plot(u_)
            #plot(uE)
            #interactive(True)
            print ''
        table.write('\\end{tabular}\n')
        table.write('\\caption{insert caption}\n')
        table.write('\\end{figure}\n')
        table.write('\n')
        table.close()



def mixedfunctionspace(N, Lambda, Degree):
    print 'Mixed functionspace'
    print '--------------------'
    print 'numerical error for:'
    for deg in Degree:
        table = open('table%s.txt'%deg, 'w')
        table.write('\\begin{table}\n')
        table.write('\\caption{insert caption}\n')
        table.write('\\begin{tabular}{|c|c|c|c|c|c|}\n')   
        table.write('\\cline{2-5} & \multicolumn{4}{|c|}{Numerical error} \\\\\n') 
        table.write('\\cline{1-6}\n')
        table.write('\\diagbox[width=4em]{$\lambda$}{N} & {8} & {16} & {32} & {64} & {Convergence rate} \\\\\n')
        table.write('\\cline{1-6}\n')
        print ''
        print 'Degree of polynomial:',deg
        for lmbda in Lambda:
            table.write('\\multicolumn{1}{|c|}{%g}\t'%lmbda)
            error = np.zeros(len(N))
            h = np.zeros(len(N))
            for n in range(len(N)):
                mesh = UnitSquareMesh(N[n], N[n])

                V = VectorFunctionSpace(mesh, 'CG', deg)
                V2 = VectorFunctionSpace(mesh, 'CG', deg+1)
                P = FunctionSpace(mesh, 'CG', 1)
                W = MixedFunctionSpace([V, P])

                up = Function(W)
                vq = TestFunction(W)
                u, p = split(up)
                v, q = split(vq)
                
                f = Expression(('mu*pi*pi* (2*x[1]*sin(pi*x[0]*x[1]) + pi*x[0]*(x[1]*x[1]*cos(pi*x[0]*x[1]) + x[0]*x[0]*cos(pi*x[0]*x[1])))',
                                '-mu*pi*pi* (2*x[0]*sin(pi*x[0]*x[1]) + pi*x[1]*(x[0]*x[0]*cos(pi*x[0]*x[1]) + x[1]*x[1]*cos(pi*x[0]*x[1])))'),mu= mu)
                
                uex = Expression(('pi*x[0]*cos(pi*x[0]*x[1])', '-pi*x[1]*cos(pi*x[0]*x[1])'))
                ue = interpolate(uex, V2)

                class Boundaries(SubDomain):
                    def inside(self, x, on_boundary):
                        return on_boundary

                boundaries = Boundaries()

                bc = DirichletBC(W.sub(0), uex, boundaries)

                mf = FacetFunction('size_t', mesh)
                mf.set_all(0)
                boundaries.mark(mf, 2)
                #plot(mf, interactive=True)

                eq1 = mu*inner(grad(u), grad(v))*dx + Constant(lmbda)*inner(p, div(v))*dx - inner(f,v)*dx
                eq2 = 1./Constant(lmbda)*inner(p, q)*dx - inner(div(u), q)*dx

                eq = eq1 - eq2

                solve(eq == 0, up, bc)
                u, p = up.split()

                error[n] = errornorm(u, ue)
                h[n] = 1./N[n]
                table.write(' & {%g}\t'%error[n])
                convergence_rate = 0
                if n == 0:
                    print 'Lambda=',lmbda,', n=',N[n],', error=',error[n]
                else:
                    convergence_rate = np.log(abs(error[n]/error[n-1]))/np.log(abs(h[n] / h[n-1]))
                    print 'Lambda=',lmbda,', n=',N[n],', error=',error[n],', convergence rate:',convergence_rate
            table.write('& {%g} \\\\\n'%convergence_rate)
            table.write('\\cline{1-6}\n')

            #plot(u)
            #plot(ue)
            #interactive(True)

            print ''
        table.write('\\end{tabular}\n')
        table.write('\\end{table}\n')
        table.write('\n')
        table.close()



def mixedfunctionspace2(N, Lambda, deg):
    print 'Mixed functionspace'
    print '--------------------'
    print 'numerical error for:' 
    for lmbda in Lambda:
        error = np.zeros(len(N))
        for n in range(len(N)):
            mesh = UnitSquareMesh(N[n], N[n])
            
            V = VectorFunctionSpace(mesh, 'CG', 1)
            V2 = VectorFunctionSpace(mesh, 'CG', 3)
            P = FunctionSpace(mesh, 'CG', 1)
            W = MixedFunctionSpace([V, P])

            up = TrialFunction(W)
            u, p = split(up)
            vq = TestFunction(W)
            v, q = split(vq)

            f = Expression(('mu*pi*pi* (2*x[1]*sin(pi*x[0]*x[1]) + pi*x[0]*(x[1]*x[1]*cos(pi*x[0]*x[1]) + x[0]*x[0]*cos(pi*x[0]*x[1])))',
                            '-mu*pi*pi* (2*x[0]*sin(pi*x[0]*x[1]) + pi*x[1]*(x[0]*x[0]*cos(pi*x[0]*x[1]) + x[1]*x[1]*cos(pi*x[0]*x[1])))'),mu= mu)
            
            ue = Expression(('pi*x[0]*cos(pi*x[0]*x[1])', '-pi*x[1]*cos(pi*x[0]*x[1])'))
            #u_e = interpolate(ue, V2)
            class Boundaries(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary

            bc = DirichletBC(W.sub(0), ue, Boundaries())

            a1 = mu*inner(grad(u), grad(v))*dx + p*div(v)*dx 
            a2 = p*q*dx - lmbda*div(u)*q*dx
            L = dot(f, v)*dx

            A = a1+a2
            up_ = Function(W)
            solve(A == L, up_, bc)
            u_, p_ = up_.split()
            
            error[n] = errornorm(ue, u_, degree_rise=2)

            if n == 0:
                print 'Lambda=',lmbda,', n=',N[n],', error=',error[n]
            else:
                convergence_rate = np.log(abs(error[n]/error[n-1]))/np.log(abs(1./N[n] / 1./N[n-1]))
                print 'Lambda=',lmbda,', n=',N[n],', error=',error[n],', convergence rate:',convergence_rate          




N = [8, 16, 32, 64]
Lambda = [1, 100, 10000]
Degrees = [1, 2]

#functionspace(N, Lambda, Degrees)
mixedfunctionspace(N, Lambda, Degrees)
#mixedfunctionspace2(N, Lambda)
