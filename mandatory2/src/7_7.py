from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
set_log_active(False)


N = [2, 4, 8, 16, 32, 64]
h = [1./i for i in N]
error = np.zeros(len(N))
u_pol = [4, 4, 3, 3]
p_pol = [3, 2, 2, 1]

for deg in range(len(u_pol)):
    print ''
    print 'P%s-P%s' %(u_pol[deg], p_pol[deg])
    print '-'*80
    for i in range(len(N)):
        mesh = UnitSquareMesh(N[i], N[i])
        V = VectorFunctionSpace(mesh, 'CG', u_pol[deg])
        V2 = VectorFunctionSpace(mesh, 'CG', u_pol[deg]+1)
        Q = FunctionSpace(mesh, 'CG', p_pol[deg])
        Q2 = FunctionSpace(mesh, 'CG', p_pol[deg]+1)

        VQ = MixedFunctionSpace([V, Q])

        up = TrialFunction(VQ)
        u, p = split(up)
        vq = TestFunction(VQ)
        v, q = split(vq)

        u_ex = Expression(('sin(pi*x[1])', 'cos(pi*x[0])'))
        p_ex = Expression('sin(2*pi*x[0])')

        boundaries = FacetFunction('size_t', mesh)
        boundaries.set_all(0)

        class left(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < DOLFIN_EPS

        class right(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] > 1.0 - DOLFIN_EPS

        class bottom(SubDomain):
            def inside(self, x, on_boundary):
                return x[1] < DOLFIN_EPS

        class top(SubDomain):
            def inside(self, x, on_boundary):
                return x[1] > 1.0 - DOLFIN_EPS

        left = left()
        left.mark(boundaries, 1)
        right = right()
        right.mark(boundaries, 2)
        bottom = bottom()
        bottom.mark(boundaries, 3)
        top = top()
        top.mark(boundaries, 4)
        #plot(boundaries, interactive=True)

        left_vel = DirichletBC(VQ.sub(0), u_ex, left)
        bottom_vel = DirichletBC(VQ.sub(0), u_ex, bottom)
        top_vel = DirichletBC(VQ.sub(0), u_ex, top)
        outlet_pressure = DirichletBC(VQ.sub(1), p_ex, right)

        bc = [left_vel, bottom_vel, top_vel, outlet_pressure]

        f = Expression(('2*pi*cos(2*pi*x[0]) - pi*pi*sin(pi*x[1])', '-pi*pi*cos(pi*x[0])'))

        eq1 = inner(grad(u), grad(v))*dx + p*div(v)*dx + inner(f, v)*dx
        eq2 = div(u)*q*dx

        equation = eq1-eq2

        up_ = Function(VQ)
        solve(lhs(equation) == rhs(equation), up_, bc)

        u_, p_ = split(up_)

        # cauchy stress tensor numerical
        tau = -p_*Identity(2) + 0.5*(grad(u_) + grad(u_).T)
        n = FacetNormal(mesh)
        ds = Measure('ds', subdomain_data=boundaries)
        stress = dot(tau, n)

        # cauchy stress tensor exact
        uex = interpolate(u_ex, V2)
        pex = interpolate(p_ex, Q2)
        tau_ex = -pex*Identity(2) + 0.5*(grad(uex) + grad(uex).T)
        stress_ex = dot(tau_ex, n)

        error[i] = assemble((stress_ex[0]-stress[0])**2*ds(1))
        error[i] =  sqrt(error[i])
    
    for i in range(len(N)):
        if i == 0:
            print 'N =',N[i],', error =',error[i]
        else:
            convergence = np.log(error[i]/error[i-1]) / np.log(h[i]/h[i-1])
            print 'N =',N[i],', error =',error[i],', convergence rate =',convergence

    plt.loglog(h, error, label='P%s-P%s'%(u_pol[deg], p_pol[deg]))
plt.xlabel('h'), plt.ylabel('error')
plt.grid('on')
plt.legend(loc='upper left')
plt.show()
