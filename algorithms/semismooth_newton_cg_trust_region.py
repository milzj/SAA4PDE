"""
This file is a modification of

https://github.com/funsim/moola/blob/master/moola/algorithms/newton_cg_trust_region.py

funsim/moola is licensed under the GNU Lesser General Public License v3.0

Changes made to https://github.com/funsim/moola/blob/master/moola/algorithms/newton_cg_trust_region.py
(commit: https://github.com/funsim/moola/commit/601b3d01fde893a2286d199a425cea38fcbd69b9)

    main loop and step computation:
    -------------------------------
    - gradient computation was replaced by normal_map.primal() computation
    - Hessian computation was replaced by derivative_normal_map computation

    main loop:
    ----------
    - The optimization variable mk was added. mk is the actural iterate. Upon calling
      normal_map, the prox of mk is stored in mk.
    - The computation of the predicted reduction was modified (no use of objxkpk) to avoid a substraction
      when evaluating the predicted reduction.
    - Moved res = ... and self.update ... (ll. 150-151) prior to while loop
    - Added a heuristic to accept step close to a stationary point based on conditioning of
      computing the actural reduction and the decrease of the residual norm at the next iterate
    - In addition to the norm of the residual, the objective function value is recorded.
    - We use a merit function based on that used in Ouyang and Milzarek (2021), as we observed
    that the step returned by the Steihaug-CG method may not decrease the objective function

    step computation:
    -----------------
    - z_old + tau*d was replaced by z_old.axpy(tau,d)

    tau_get:
    --------
    - The default option verify was chanced to False.

    header:
    -------
    - sqrt of numpy is used
    - moola tools are loaded.

    References:
    -----------

    W. Ouyang and A. Milzarek, A trust region-type normal map-based semismooth
    Newton method for nonsmooth nonconvex composite optimization, (2021), https:
    //arxiv.org/abs/2106.09340

"""

from numpy import sqrt
from moola.algorithms.optimisation_algorithm import *
from moola.algorithms.bfgs import LinearOperator, dual_to_primal
from moola.adaptors import *

class TrustRegionSemismoothNewtonCG(OptimisationAlgorithm):
    ''' 
    An implementation of the trust region NewtonCG method 
    described in Wright 2006, section 7. 
    '''
    def __init__(self, problem, initial_point = None, options={}):
        '''
        Initialises the trust region Newton CG method. The valid options are:
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200. 
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.

          '''

        # Set the default options values
        self.problem = problem
        self.set_options(options)

        self.data = {'control'   : initial_point,
                     'iteration' : 0}

        # validate method specific options
        assert 0 <= self.options['eta'] < 1./4
        assert self.options['tr_Dmax'] > 0
        assert 0 < self.options['tr_D0'] < self.options['tr_Dmax']
    
    def __str__(self):
        s = "Trust region Semismooth Newton CG method.\n"
        s += "-"*30 + "\n"
        s += "Maximum iterations:\t %i\n" % self.options['maxiter']
        return s

    # set default parameters
    
    @classmethod
    def default_options(cls):
        # this is defined as a function to prevent defaults from being changed at runtime.
        default = OptimisationAlgorithm.default_options()
        default.update(
            # generic parameters:
            {"gtol"                   : 1e-4,
             "maxiter"                :  200,
             "display"                :    2,
             "callback"               : None,
             "record"                 : ("grad_norm"), 

             # method specific parameters:
             "tr_Dmax"                :  1e5, # overall bound on the step lengths
             "tr_D0"                  :    1, # current bound on the step length
             "eta"                    : 1./8,
             "correction_step"        : False,
             "restrict"               : False
             })
        return default

    def get_tau(self, obj, z, d, D, verify=False):
        """ Function to find tau such that p = pj + tau.dj, and ||p|| = D. """

        
        dot_z_d = z.inner(d)
        len_d_sqrd = d.inner(d)
        len_z_sqrd = z.inner(z)

        t = sqrt(dot_z_d**2 - len_d_sqrd * (len_z_sqrd - D**2))

        taup = (- dot_z_d + t) / len_d_sqrd
        taum = (- dot_z_d - t) / len_d_sqrd

        if verify:
            eps = 1e-8
            if abs((taup*d+z).norm()-D)/D > eps or abs((z+taum*d).norm()-D)/D > eps:
                raise ArithmeticError("Tau could not be computed accurately due to numerical errors.")

        return taup, taum

    def compute_pk_cg_steihaug(self, obj, x, m, D):
        ''' Solves min_pk fk + grad fk(p) + 0.5*p^T H p using the CG Steighaug method '''  
        z = x.copy()
       	z.zero()
        r = obj.normal_map(x,m).primal()
        d = -r
        Td = d.copy()

        if self.options["restrict"] == True:
            obj.restrict(r)
            obj.restrict(d)
            obj.restrict(Td)

        rtr = r.inner(r)
        rnorm = rtr**0.5
        eps = min(0.5, sqrt(rnorm))*rnorm  # Stopping criteria for CG

        if rnorm <= eps:
            print("CG solver converged")
            return z, False

        cg_iter = 0
        while True:
            print("CG iteration %s" % cg_iter)
            cg_iter += 1

            if self.options["restrict"] == True:
                obj.restrict(Td)
            Hd = obj.derivative_normal_map(x,m)(d)
            curv = Hd.apply(Td)

            # Curvatur test 
            if curv <= 0.0:
                print("curv <= 0.0, curv = {}".format(curv))
                taup, taum = self.get_tau(obj, z, d, D)
                pp = z.copy()
                pp.axpy(taup,d)
                pm = z.copy()
                pm.axpy(taum,d)

                ppHpp = obj.derivative_normal_map(x,m)(pp).apply(pp)
                pmHpm = obj.derivative_normal_map(x,m)(pm).apply(pm)
                if r.inner(pp)+.5*ppHpp <= r.inner(pp)+.5*pmHpm:
                    return pp, True
                else:
                    return pm, True

            alpha = rtr / curv
            z_old = z.copy()
            z.axpy(alpha, d)
            znorm = z.norm()
            print("|z_%i| = %f" % (cg_iter, znorm))

            # Trust region boundary test
            if znorm >= D:
                print("|z| >= Delta")
                tau = self.get_tau(obj, z_old, d, D)[0]
                assert tau >= 0
                z_old.axpy(tau,d)
                return z_old, True

            r.axpy(alpha, Hd.primal())
            rtr, rtr_old = r.inner(r), rtr
            rnorm = rtr**0.5

            # CG convergence test
            if rnorm < eps:
                print("CG solver converged")
                return z, False

            beta = rtr / rtr_old
            d = -r + beta*d
            Td.assign(d)

    def solve(self):
        ''' Solves the optimisation problem with the trust region Newton-CG method. 
         '''
        print(self)
            
        print("Doing another iteration")
        obj = self.problem.obj
        i = 0
        Dk = self.options['tr_D0']
        xk = self.data['control']
        mk = xk.copy()
        obj(xk)
        #mk = obj.derivative(xk).primal()
        #mk.scale(-1/obj.alpha)
        res = obj.normal_map(xk,mk)
        self.update({'objective': obj(xk), 'grad_norm' : res.primal_norm()})

        resxkpk = res.copy()
        mknew = mk.copy()
        xknew = xk.copy()
        cond_tol = 1e6

        while True:

            if self.check_convergence() != 0:
                break

            self.display(self.iter_status, 2)

            # Compute trust region point
            pk, is_cauchy_point = self.compute_pk_cg_steihaug(obj, xk, mk, Dk)

            if self.options["correction_step"] == True:
                rk = res.primal() + obj.derivative_normal_map(xk,mk)(pk).primal()
                pk.axpy(-1.0/obj.alpha, rk)
                pknorm = pk.norm()
                pk.scale(min(1.0, Dk/pknorm/obj.alpha))

            # Evaluate trust region performance
            objxk = obj(xk)
            res = obj.normal_map(xk,mk)
            mkxkpk = res.apply(pk) + 0.5 * obj.derivative_normal_map(xk,mk)(pk).apply(pk)
            objxkpk = obj(xk + pk)

            mknew.assign(mk)
            mknew.axpy(1.0, pk)
            resxkpk = obj.normal_map(xknew,mknew)

            tau = .5
            Hxk = objxk + .5*tau*res.primal_norm()/obj.alpha
            Hxkpk = objxkpk + .5*tau*resxkpk.primal_norm()/obj.alpha
            #ared = objxk-objxkpk + .5*tau*res.primal_norm()**2/obj.alpha**2 - .5*tau*resxkpk.primal_norm()**2/obj.alpha**2
            ared = Hxk-Hxkpk
            rhok = ared / (- mkxkpk)
            print("rhok={}".format(rhok))
            print("objxk-objxkpk={}".format(objxk-objxkpk))
            print("ared={}".format(ared))
            cond = (abs(objxk)+abs(objxkpk))/abs(objxk-objxkpk)
            condared = (abs(Hxk)+abs(Hxkpk))/abs(Hxk-Hxkpk)
            cond = condared
            print("cond={}".format(cond))

            heuristic = cond > cond_tol
            if heuristic:
                mknew.assign(mk)
                mknew.axpy(1.0, pk)
                resxkpk = obj.normal_map(xknew,mknew)

            # Update rhok if new step yields smaller gradient and condition number of ared is high
            heuristic = resxkpk.primal_norm() < res.primal_norm() and heuristic

            if heuristic:
                print("cond={}".format(cond))
                print("objxk-objxkpk={}".format(objxk-objxkpk))
                rhok = 2./4

            if rhok < self.options['eta']:
                Dk *= 1./4
                print("Decreasing trust region radius to %f." % Dk)

            elif rhok > 3./4 and is_cauchy_point:
                Dk = min(2*Dk, self.options['tr_Dmax'])
                Dk = max(Dk, 1e-2)
                print("Increasing trust region radius to %f." % Dk)

            if rhok > self.options['eta']:
                mk.axpy(1., pk)
                Dk = max(Dk, 1e-2)

                if heuristic:
                    res.assign(resxkpk)
                    xk.assign(xknew)
                else:
                    res = obj.normal_map(xk,mk)

                print("Trust region step accepted.")
            else:
                print("Rejecting step. Reason: trust region step did not reduce objective.")

            i += 1

            # store current iteration variables
            self.update({'iteration' : i,
                         'control'   : xk,
                         'grad_norm' : res.primal_norm(),
                         'objective': obj(xk)
                        })
        self.display(self.convergence_status, 1)
        self.display(self.iter_status, 1)
        return self.data

