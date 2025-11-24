import quimb as qu
import quimb.tensor as qtn
import math
import cotengra as ctg
import numpy as np
import itertools
import re
import jax
import jax.numpy as jnp
import autoray as ar
from quimb.utils import oset
from tqdm import tqdm
import gen_loop_tn as lg_tn

from quimb.tensor.belief_propagation.bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    create_lazy_community_edge_map,
)



ar.register_function("torch", "stop_gradient", lambda x: x.detach())
ar.register_function("jax",   "stop_gradient", jax.lax.stop_gradient)
def stop_grad(x):
    return ar.do("stop_gradient", x)



def tn_normalize_wrt_massages(tn, bp , site_tags, opt="auto-hq"):
    bp.normalize_messages()
    
    res = lg_tn.bp_excited_loops(tn, bp, [[]], chi=10, contract_=False, prgbar=False, opt=opt)
    tn_l = res["tn_l"]
    tags = list(site_tags)
    tn_ = tn_l[0]
    norm_l = []
    for tag in tags:
        tn_local_ = tn_.select(tag, which="all")
        tn_local = tn.select(tag, which="all")
    
        norm_ = tn_local_.contract(all)
        norm_l.append(   ar.do("log10", norm_)    )
        tn_local /= norm_ 
        #tn_local.draw()


    tn.exponent += sum(norm_l)
    local_tn = { str_: tn[str_] for str_ in site_tags}
    bp.local_tn = local_tn


def run_bp(tn, site_tags, normalize=True, progbar=False, 
           max_iterations=520, tol=1.e-7, opt="auto-hq",
           damping=0.01, update='parallel', normalize_="L2phased", 
           local_convergence=True,
           inplace = False, print_=False, diis=True,
          ):
    
    tn = tn if inplace else tn.copy()
    

    if print_:
        print(f"bp runs with the {update} update ~ iter = {max_iterations}")

    
    bp = L1BP(tn, optimize=opt, site_tags=site_tags, damping=damping, 
              normalize=normalize_, 
              update=update,
              local_convergence=local_convergence)
    bp.run(tol=tol, max_iterations=max_iterations, progbar=progbar, diis=diis)

    mantissa, norm_exponent = bp.contract(strip_exponent=True)
    est_norm = mantissa * 10**norm_exponent

    if print_:
        print("messages normalized and projects are being cal")
    
    bp.normalize_messages()
    bp.cal_projects()
    
    if normalize:
        if print_:
            print("tn is being normalized w.r.t bp massages")
        tn_normalize_wrt_massages(tn, bp, site_tags, opt=opt)

    res = {"bp":bp, "norm": est_norm, "tn":tn}
    return res


# build-up a projector into the excited subspace of the bp massages:   I - |tmi><tmj|  
def projector(tmi, tmj):
    
    tmi_ = tmi.data
    tmj_ = tmj.data

    # infer data
    dtype = ar.get_dtype_name(tmi_)
    backend = ar.infer_backend(tmi_)
    try:
        device = tmi_.device
    except:
        device = "cpu"
    

    #store original shape
    shape_ =  ar.do("shape", tmi_)
    
    # P = |mi> <mj|
    mi_vector = ar.do("reshape", tmi_, (-1,))
    mj_vector = ar.do("reshape", tmj_, (-1,))
    Pr_ij = ar.do("outer", mi_vector, mj_vector)
    
    # Create the identity matrix of the same shape as the projector
    I = ar.do("eye", Pr_ij.shape[0], dtype=dtype)  
    I = ar.do('asarray', I, like=backend, device=device)
    
    # Subtract the projector from the identity matrix: I - |mi> <mj|
    Pr_excited = I - Pr_ij
    
    #put it back to original shape
    Pr_excited = ar.do("reshape", Pr_excited, shape_+shape_)


    return Pr_excited


def bp_info_rho(cor):
    
    
    leftinds_rho = []
    rightinds_rho = []
    reg_tags = []
    reg_reindex = {}
    for cor_ in cor:
        x, y = cor_ 
        reg_tags.append(f"I{x},{y}")
        reg_reindex |= {f"k{x},{y}":f"b{x},{y}"}
        leftinds_rho.append(f"k{x},{y}")
        rightinds_rho.append(f"b{x},{y}")
    inds_rho = leftinds_rho + rightinds_rho
    
    res_cor = {"inds_rho":inds_rho, "reg_reindex":reg_reindex, "reg_tags":reg_tags, "leftinds_rho":leftinds_rho}
    res_cor |= {"rightinds_rho":rightinds_rho}
    return res_cor


class L1BP(BeliefPropagationCommon):
    """Lazy 1-norm belief propagation. BP is run between groups of tensors
    defined by ``site_tags``. The message updates are lazy contractions.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to run BP on.
    site_tags : sequence of str, optional
        The tags identifying the sites in ``tn``, each tag forms a region,
        which should not overlap. If the tensor network is structured, then
        these are inferred automatically.
    damping : float or callable, optional
        The damping factor to apply to messages. This simply mixes some part
        of the old message into the new one, with the final message being
        ``damping * old + (1 - damping) * new``. This makes convergence more
        reliable but slower.
    update : {'sequential', 'parallel'}, optional
        Whether to update messages sequentially (newly computed messages are
        immediately used for other updates in the same iteration round) or in
        parallel (all messages are comptued using messages from the previous
        round only). Sequential generally helps convergence but parallel can
        possibly converge to differnt solutions.
    normalize : {'L1', 'L2', 'L2phased', 'Linf', callable}, optional
        How to normalize messages after each update. If None choose
        automatically. If a callable, it should take a message and return the
        normalized message. If a string, it should be one of 'L1', 'L2',
        'L2phased', 'Linf' for the corresponding norms. 'L2phased' is like 'L2'
        but also normalizes the phase of the message, by default used for
        complex dtypes.
    distance : {'L1', 'L2', 'L2phased', 'Linf', 'cosine', callable}, optional
        How to compute the distance between messages to check for convergence.
        If None choose automatically. If a callable, it should take two
        messages and return the distance. If a string, it should be one of
        'L1', 'L2', 'L2phased', 'Linf', or 'cosine' for the corresponding
        norms. 'L2phased' is like 'L2' but also normalizes the phases of the
        messages, by default used for complex dtypes if phased normalization is
        not already being used.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    contract_every : int, optional
        If not None, 'contract' (via BP) the tensor network every
        ``contract_every`` iterations. The resulting values are stored in
        ``zvals`` at corresponding points ``zval_its``.
    inplace : bool, optional
        Whether to perform any operations inplace on the input tensor network.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.
    """

    def __init__(
        self,
        tn,
        site_tags=None,
        *,
        damping=0.0,
        update="sequential",
        normalize=None,
        distance=None,
        local_convergence=True,
        optimize="auto-hq",
        message_init_function=None,
        contract_every=None,
        gauges = {},
        inplace=False,
        **contract_opts,
    ):
        super().__init__(
            tn,
            damping=damping,
            update=update,
            normalize=normalize,
            distance=distance,
            contract_every=contract_every,
            inplace=inplace,
        )

        self.local_convergence = local_convergence
        self.optimize = optimize
        self.contract_opts = contract_opts
        self.projects = {}
        self.gauges = gauges
        self.tn = tn.copy()

        if site_tags is None:
            self.site_tags = tuple(tn.site_tags)
        else:
            self.site_tags = tuple(site_tags)

        (
            self.edges,
            self.neighbors,
            self.local_tns,
            self.touch_map,
        ) = create_lazy_community_edge_map(tn, site_tags)
        self.touched = oset()

        # for each meta bond create initial messages
        self.messages = {}
        for pair, bix in self.edges.items():
            # compute leftwards and rightwards messages
            for i, j in (sorted(pair), sorted(pair, reverse=True)):
                tn_i = self.local_tns[i]
                # initial message just sums over dangling bonds

                if message_init_function is None:
                    tm = tn_i.contract(
                        all,
                        output_inds=bix,
                        optimize=self.optimize,
                        drop_tags=True,
                        **self.contract_opts,
                    )
                    # normalize
                    tm.modify(apply=self._normalize_fn)
                    tm.modify(apply=stop_grad)
                else:
                    shape = tuple(tn_i.ind_size(ix) for ix in bix)
                    tm = qtn.Tensor(
                        data=message_init_function(shape),
                        inds=bix,
                    )
                    tm.modify(apply=stop_grad)
                    
                self.messages[i, j] = tm

        # compute the contractions
        self.contraction_tns = {}
        for pair, bix in self.edges.items():
            # for each meta bond compute left and right contractions
            for i, j in (sorted(pair), sorted(pair, reverse=True)):
                tn_i = self.local_tns[i].copy()
                # attach incoming messages to dangling bonds
                tks = [
                    self.messages[k, i] for k in self.neighbors[i] if k != j
                ]
                # virtual so we can modify messages tensors inplace
                tn_i_to_j = qtn.TensorNetwork((tn_i, *tks), virtual=True)
                self.contraction_tns[i, j] = tn_i_to_j


    def run_vidal(self, iter_=20, tol=1.e-7, progbar=True, 
                  method_ = "can", #su
                  equalize_norms=False, 
                  print_=False, test_=False,
                  normalize=True, to_backend=None):

            expoent_ = self.tn.exponent
            
            if print_:
                print("\033[34m tn is being modified to --> Gammas and Lambda (new messages) \033[0m")

    
            if method_ == "su":
                self.tn.gauge_all_simple_(iter_, tol=tol, gauges=self.gauges, 
                                          progbar=progbar, damping=self.damping, 
                                          equalize_norms=equalize_norms,
                                         )
                if to_backend:
                    self.gauges = {key: to_backend(values) for key, values in self.gauges.items()}                        
            
            if method_ == "can":
                self.tn.gauge_all_canonize(max_iterations=iter_, absorb='both', 
                                           gauges=self.gauges, equalize_norms=equalize_norms, 
                                           inplace=True, **{"cutoff":0, "cutoff_mode":'rsum2', "method":'svd'},
                                          )
                if to_backend:
                    self.gauges = {key: to_backend(values) for key, values in self.gauges.items()}                        
    
    
            self.tn.exponent += expoent_

            
        
            if test_:
                tn_ = self.tn.copy()
                outer, inner = tn_.gauge_simple_insert(self.gauges)
                val_ = tn_.contract(all, optimize=self.optimize)*10**tn_.exponent
                print(f"\033[33m original tn estimation {val_} \033[0m", 
                      
                     )
        
            (    _,
                _,
                self.local_tns,
                _,
            ) = create_lazy_community_edge_map(self.tn, self.site_tags)

            if print_:
                norm_0 = norm_l1(self.tn, self.gauges, self.site_tags, self.optimize)            
                print(f"\033[35m vidal norm estimation = {norm_0} \033[0m")

        
        
            for key, message_ in self.messages.items():
                inds = list(message_.inds)
                g = self.gauges[inds[0]]
                #print(g.shape, message_.data.shape)
                message_.modify(data=g*1.)
            
        
            if print_:
                print("\033[1;35m The messages are normalized, and the projections are calculated \033[0m")

            
            self.normalize_messages()
            self.cal_projects_gauge()


            if normalize:
                if print_:
                    print("\033[31m The tn is normalized w.r.t the bp messages \033[0m")
                res = bp_excited_loops(self.tn, self, [[]], chi=10, contract_=False, prgbar=False)
                tn_ = res["tn_l"][0]
                norm_l = []
                for tag in self.site_tags:
                    tn_local_ = tn_.select(tag, which="all")
                    tn_local = self.tn.select(tag, which="all")
                
                    norm_ = tn_local_.contract(all)
                    norm_l.append(np.log10(complex(norm_)))
                    tn_local /= norm_ 
                    #tn_local.draw()
            
            
                self.tn.exponent += sum(norm_l)
                local_tn = { str_: self.tn[str_] for str_ in self.site_tags}
                self.local_tn = local_tn
    
    
    
    def iterate(self, tol=5e-6):
        if (not self.local_convergence) or (not self.touched):
            # assume if asked to iterate that we want to check all messages
            self.touched.update(
                pair for edge in self.edges for pair in (edge, edge[::-1])
            )

        ncheck = len(self.touched)
        nconv = 0
        max_mdiff = -1.0
        new_touched = oset()

        def _compute_m(key):
            i, j = key
            bix = self.edges[(i, j) if i < j else (j, i)]
            tn_i_to_j = self.contraction_tns[i, j]
            tm_new = tn_i_to_j.contract(
                all,
                output_inds=bix,
                optimize=self.optimize,
                **self.contract_opts,
            )
            return self._normalize_fn(tm_new.data)

        def _update_m(key, data):
            nonlocal nconv, max_mdiff

            tm = self.messages[key]

            # pre-damp distance
            mdiff = self._distance_fn(data, tm.data)

            if self.damping:
                data = self._damping_fn(data, tm.data)

            # # post-damp distance
            # mdiff = self._distance_fn(data, tm.data)

            if mdiff > tol:
                # mark touching messages for update
                new_touched.update(self.touch_map[key])
            else:
                nconv += 1

            max_mdiff = max(max_mdiff, mdiff)
            tm.modify(data=data)

        if self.update == "parallel":
            new_data = {}
            # compute all new messages
            while self.touched:
                key = self.touched.pop()
                new_data[key] = _compute_m(key)
            # insert all new messages
            for key, data in new_data.items():
                _update_m(key, data)

        elif self.update == "sequential":
            # compute each new message and immediately re-insert it
            while self.touched:
                key = self.touched.pop()
                data = _compute_m(key)
                _update_m(key, data)

        self.touched = new_touched
        return {
            "nconv": nconv,
            "ncheck": ncheck,
            "max_mdiff": max_mdiff,
        }

    def contract(self, strip_exponent=False, check_zero=True):
        zvals = []
        for site, tn_ic in self.local_tns.items():
            if site in self.neighbors:
                tval = qtn.tensor_contract(
                    *tn_ic,
                    *(self.messages[k, site] for k in self.neighbors[site]),
                    optimize=self.optimize,
                    **self.contract_opts,
                )
            else:
                # site exists but has no neighbors
                tval = tn_ic.contract(
                    all,
                    output_inds=(),
                    optimize=self.optimize,
                    **self.contract_opts,
                )
            zvals.append((tval, 1))

        for i, j in self.edges:
            mval = qtn.tensor_contract(
                self.messages[i, j],
                self.messages[j, i],
                optimize=self.optimize,
                **self.contract_opts,
            )
            # power / counting factor is -1 for messages
            zvals.append((mval, -1))

        return combine_local_contractions(
            zvals,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
        )

    def normalize_message_pairs(self):
        """Normalize all messages such that for each bond `<m_i|m_j> = 1` and
        `<m_i|m_i> = <m_j|m_j>` (but in general != 1).
        """
        for i, j in self.edges:
            tmi = self.messages[i, j]
            tmj = self.messages[j, i]
            nij = abs(tmi @ tmj) ** 0.5
            nii = (tmi @ tmi) ** 0.25
            njj = (tmj @ tmj) ** 0.25
            tmi /= nij * nii / njj
            tmj /= nij * njj / nii



    def normalize_messages(self):
        """Normalize all messages such that for each bond `<m_i|m_j> = 1` and
        `<m_i|m_i> = <m_j|m_j>` (but in general != 1).
        """
        for i, j in self.edges:
            tmi = self.messages[i, j]
            tmi.add_tag(j)
            tmj = self.messages[j, i]
            tmj.add_tag(i)
            
            nij = (tmi @ tmj)**0.5
            nii = (tmi @ tmi)**0.25
            njj = (tmj @ tmj)**0.25


            tmi /= (nij * nii / njj)
            tmj /= (nij * njj / nii)

    
    def cal_projects(self):
        pr_excited = {}
        for pair, bix in self.edges.items():
                pair_ = sorted(pair) 
                i, j = pair_
                tmi = self.messages[j, i]
                tmj = self.messages[i, j]
                
                #left indicies are similar to bp massages (the bond connect i and j)
                left_inds = list(tmi.inds)

                
                right_inds = list(tmj.inds)
                tmj = tmj.transpose(*right_inds)
                #right indicies are chosen randomly
                right_inds = [ qtn.rand_uuid() for indx in right_inds]
                
                tmi = tmi.transpose(*left_inds)
                
                
            
                # make projector into excited bp manifold
                p_excited_ij = projector(tmi, tmj)
            
                #Store tensors
                pr_excited[i, j] =  qtn.Tensor(p_excited_ij, left_inds=left_inds, inds=left_inds+right_inds, tags=["proj"])
        
        
        
        self.projects = pr_excited
        
    def cal_projects_gauge(self):
        pr_excited = {}
        for pair, bix in self.edges.items():
                pair_ = sorted(pair) 
                i, j = pair_
                tmi = self.messages[j, i].copy()
                tmj = self.messages[i, j].copy()
                bix, = bix
                lambda_ = self.gauges[bix] * 1.
                #left indicies are similar to bp massages (the bond connect i and j)
                left_inds = list(tmi.inds)

                
                right_inds = list(tmj.inds)
                tmj = tmj.transpose(*right_inds)
                #right indicies are chosen randomly
                right_inds = [ qtn.rand_uuid() for indx in right_inds]
                
                tmi = tmi.transpose(*left_inds)
                
                
            
                # make projector into excited bp manifold
                p_excited_ij = projector_gauge(tmi, tmj, lambda_)
            
                #Store tensors
                pr_excited[i, j] =  qtn.Tensor(p_excited_ij, left_inds=left_inds, inds=left_inds+right_inds, tags=["proj"])
        
        
        
        self.projects = pr_excited




