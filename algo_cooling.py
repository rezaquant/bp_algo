import quimb as qu
import numpy as np
import quimb.tensor as qtn
import os, sys
import itertools
import functools
import torch
from tqdm import tqdm
from time import sleep
import cotengra as ctg
import jax
import jax.numpy as jnp
import autoray as ar
import time
import numpy as np
import re

import logging
logger = logging.getLogger(__name__)


def backend_torch(device = "cpu", dtype = torch.float64, requires_grad=False):
    
    def to_backend(x, device=device, dtype=dtype, requires_grad=requires_grad):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
    
    return to_backend

def backend_numpy(dtype=np.float64):
    
    def to_backend(x, dtype=dtype):
        return np.array(x, dtype=dtype)
    
    return to_backend


def backend_jax(dtype=jnp.float64, device=jax.devices("cpu")[0]):
    # device = jax.devices("cpu")[0]
    # dtype=jnp.float64
    # def to_backend(x, device=device, dtype=dtype):
    #     return jax.device_put(jnp.array(x, dtype=dtype), device)

    
    def to_backend(x, dtype=dtype, device=device):
        arr = jax.device_put(jnp.array(x, dtype=dtype), device)
        return arr

    return to_backend


def opt_(progbar=True, max_time="rate:1e8"):
    opt = ctg.ReusableHyperOptimizer(
        max_repeats=2**8,
        max_time=max_time,
        parallel=True,  # optimize in parallel
        optlib="cmaes",  # an efficient parallel meta-optimizer
        hash_method="b",  # most generous cache hits
        directory="cash",  # cache paths to disk
        progbar=progbar,  # show live progress
    )
    return opt

def energy_global(MPO_origin, mps_a, opt="auto-hq"):

    mps_a_ = mps_a.copy()
    mps_a_.normalize()
    p_h=mps_a_.H 
    p_h.reindex_(  { f"k{i}":f"b{i}" for i in range(mps_a.L)} )
    MPO_t = MPO_origin *1.0
 
    E_dmrg = (p_h | MPO_t | mps_a_).contract(all,optimize=opt)
    return E_dmrg 



def gate_1d(tn, where, G, ind_id="k{}", site_tags="I{}",
            cutoff=1.e-12, contract='split-gate', 
            inplace=False):

    """
    Apply a 1D gate to a tensor network at one or two sites.

    Args:
        tn:      Tensor network (quimb/qtn TensorNetwork).
        where:   Iterable of site indices; length 1 (single-qubit) or 2 (two-qubit).
        G:       Gate tensor (or matrix).
        ind_id: Format string for site indices (e.g., "k{}" -> "k3").
        site_tags: Format string for site tags   (e.g., "I{}" -> "I3").
        cutoff:  SVD cutoff (used for split contraction paths).
        contract: Contraction mode (e.g., "split-gate") or bool for single-qubit.
        inplace: Modify tn in place if True; otherwise return a new TN.

    Returns:
        TensorNetwork with the gate applied and site tags added.
    """
    
    if len(where)==2:
        x, y = where
        tn = qtn.tensor_network_gate_inds(tn, G, [ind_id.format(x), ind_id.format(y)], contract=contract, inplace=inplace,
                                **{"cutoff":cutoff}
                                    )


        # for s in (x, y):
        #     ind = ind_id.format(s)
        #     tids = tn.ind_map.get(ind)
        #     if tids:
        #         tid = next(iter(tids))
        #         tn.tensor_map[tid].add_tag(site_tags.format(s))

        
        # adding site tags
        t = [ tn.tensor_map[i] for i in tn.ind_map[ind_id.format(x)] ][0]
        t.add_tag(site_tags.format(x))
        t = [ tn.tensor_map[i] for i in tn.ind_map[ind_id.format(y)] ][0]
        t.add_tag(site_tags.format(y))

    if len(where)==1:
        x, = where
        tn = qtn.tensor_network_gate_inds(tn, G, [ind_id.format(x)], contract=True, inplace=inplace)

    return tn




def internal_inds(psi):
    open_inds = psi.outer_inds()
    innre_inds = []
    for t in psi:
        t_list = list(t.inds)
        for j in t_list :
            if j not in open_inds:
                innre_inds.append(j)
    return innre_inds

# def fidel_mps(psi, psi_fix, opt):
#     val_0 = abs((psi.H & psi).contract(all, optimize=opt) )
#     val_1 = abs((psi.H & psi_fix).contract(all, optimize=opt))
#     val_ = abs((psi_fix.H & psi_fix).contract(all, optimize=opt))
#     val_1 = val_1 ** 2
    
#     return  val_1 / (val_0 * val_) 


def fidel_mps_normalized(psi, psi_fix, opt,  cur_orthog=None):
    tn = psi.H & psi_fix
    val_1 = abs(tn.contract(all, optimize=opt))
    return  val_1 ** 2

def gate_info(circ, map_1d):
    
    gates = circ.gates

    gate_info = {}
    for count, gate in enumerate(gates):    
        where = gate.qubits
        
        if len(where) == 2:
            x, y = where
            where_ = (map_1d[x], map_1d[y]) 
            gate_info[ (where_, count)] = gate.array.reshape(2,2,2,2)
        if len(where) == 1:
            x,  = where
            where_ = (map_1d[x],) 
            gate_info[(where_, count)] = gate.array.reshape(2,2)

    return gate_info

def rand_uni(n, to_backend=None, dtype=torch.complex64, requires_grad_=True, device="cpu", seed=2 ):
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Step 1: Make a complex-valued random matrix
    A = torch.randn(n, n, dtype=dtype, device=device)
    
    # Step 2: QR decomposition to get unitary matrix
    Q, R = torch.linalg.qr(A)
    
    # Step 3: Normalize to ensure det(Q) = 1 (optional for SU(n))
    # Adjust sign to ensure unitarity (optional for full unitarity)
    d = torch.diagonal(R)
    Q = Q * (d / torch.abs(d)).conj()

    # Step 4: Enable gradient tracking
    if to_backend:
        Q = to_backend(Q)
        Q = Q.clone().detach().requires_grad_(requires_grad_)
        return Q
    else:
        Q = Q.clone().detach().requires_grad_(requires_grad_)
        return Q

def canonize_mps(p, where, cur_orthog):
    xmin, xmax = sorted(where)
    p.canonize([xmin, xmax], cur_orthog=cur_orthog, 
               #info=info_c
              )
    # update cur_orthog in place (preserving reference)
    cur_orthog[:] = [xmin, xmax]


def fidel_mps(psi, psi_fix):

    opt = opt_(progbar=False)
    val_0 = abs((psi.H & psi).contract(all, optimize=opt) )
    val_1 = abs((psi.H & psi_fix).contract(all, optimize=opt))
    val_ = abs((psi_fix.H & psi_fix).contract(all, optimize=opt))

    val_1 = val_1 ** 2
    f = complex(val_1 / (val_0 * val_) ).real
    return  f




class FIT:
    """
    Fidelity Fitting for tensor networks.

    Parameters
    ----------
    tn : TensorNetwork
        Target tensor network to fit.
    p0 : TensorNetwork, optional
        Initial MPS (starting state). Must support `.copy()` and `.canonize()`.
    cutoffs : float, optional
        Numerical cutoff for truncation (default: 1e-9).
    backend : str or None, optional
        Backend specification for tensor operations.
    n_iter : int, optional
        Number of optimization iterations (default: 4).
    verbose : bool, optional
        If True, logs fidelity at each iteration.
    re_tag : bool, default=True
        If True, (re)tag the target TN for environment construction.
    """

    def __init__(self, tn, p=None, cutoffs=1.e-10, backend=None, 
                 site_tag_id="I{}", opt = "auto-hq", range_int=[],
                 re_tag=False, info={}, warning=True):

        if not isinstance(p, (qtn.MatrixProductState, qtn.MatrixProductOperator)):
            if warning:
                logger.warning("No initial MPS `p` provided. FIT requires an initial state for fitting.")        
        
        self.L = len(p.tensor_map.keys())
        
        self.p = p.copy() if p is not None else None
        if site_tag_id:
            self.p.view_as_(qtn.MatrixProductState, L = self.L, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
        
        
        
        
        self.site_tag_id = site_tag_id
        self.tn = tn.copy()
        self.opt = opt
        self.cutoffs = cutoffs
        self.backend = backend
        self.loss = []
        self.loss_ = []
        self.info = info
        self.range_int = range_int

        # Reindex tensor network with random UUIDs for internal indices
        self.tn.reindex_( {idx: qtn.rand_uuid() for idx in self.tn.inner_inds()} )



        
        if set(self.tn.outer_inds()) != set(self.p.outer_inds()):
            if warning:
                logger.warning("tn & p contains different inds ")        

        
        # re_new tags of tn to be used for effective envs:
        if re_tag:
            self._re_tag()


    def visual(self, figsize=(14, 14), layout="neato", show_tags=False, tags_=[], show_inds=False):
        # Visualize network with MPS
        tags = [  self.site_tag_id.format(i)  for i in range(self.L)] + tags_
        return (self.tn & self.p).draw(tags, legend=False, show_inds=show_inds,
                                 show_tags=show_tags, figsize=figsize, node_outline_darkness=0.1, 
                                       node_outline_size=None, highlight_inds_color="darkred",
                                      edge_scale=2.0, layout=layout,refine_layout="auto",
                                      highlight_inds=self.p.outer_inds(),
                                      )

    
    # -------------------------
    # Tagging methods
    # -------------------------
    def _deep_tag(self):
        """
        Propagates tags through the tensor network to ensure every tensor
        receives at least one site tag. Useful for layered TNs.
        """
        tn = self.tn
        count = 1

        while count >= 1:
            tags = tn.tags
            count = 0
            for tag in tags:
                tids = tn.tag_map[tag]
                neighbors = qtn.oset()
                for tid in tids:
                    t = tn.tensor_map[tid]
                    for ix in t.inds:
                        neighbors |= tn.ind_map[ix]
                for tid in neighbors:
                    t = tn.tensor_map[tid]
                    if not t.tags:
                        t.add_tag(tag)
                        count += 1

    def _re_tag(self):
        
        # drop tags
        tn = self.tn
        tn.drop_tags()

        # get outer inds and all tags
        p = self.p
        site_tags = [ self.site_tag_id.format(i) for i in range(p.L)   ]
        inds = list(p.outer_inds())
        

        # smart tagging for the first layer: meaning each tensor in tn is connected directly to p's tensors
        for site_tag in site_tags:
            indx = [i for i in p[site_tag].inds if i in inds][0]
            
            t = [tn.tensor_map[tid] for tid in tn.ind_map[indx]][0]
            
            if not t.tags:
                t.add_tag(site_tag)
                


        if len(tn.tensor_map.keys()) != len(tn.tags):
            if warning:
                logger.warning("Missing tags in the tensor network — it’s probably a layered TN.") 
            self._deep_tag()

            
    def run(self, n_iter=6, verbose=True):
        
        """Run the fitting process."""
        if self.p is None:
            raise ValueError("Initial state `p0` must be provided.")

        psi = self.p
        L = self.L        
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        for iteration in range(n_iter):            
            for site in range(L):
                
                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1

                # Canonicalize psi at the current site
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)

                
                psi_h = psi.H.select([site_tag_id.format(site)], "!any")
                tn_ = psi_h | self.tn


                # Contract and normalize
                f = tn_.contract(all, optimize=opt)
                f = f.transpose(*psi[site].inds)

                norm_f = (f.H & f).contract(all) ** 0.5
                self.loss_.append( complex(norm_f).real )
                
                # Update tensor data
                psi[site].modify(data=f.data)

            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))

    def _build_env_right(self, psi, env_right):
        """
        Build right environments env_right["I{i}"] for i in 0..L-1.
        env_right[i] corresponds to contraction of site i and everything to the right (inclusive).
        """
        L = self.L
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        # iterate from rightmost to leftmost
        for i in reversed(range(L)):
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            
            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block 

                
            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
            else:
                # tie to previously computed right environment
                t |= env_right[site_tag_id.format(i+1)]
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)




    def _right_range(self, psi, env_right, start, stop):
        """
        Build right environments env_right["I{i}"] for i in 0..L-1.
        env_right[i] corresponds to contraction of site i and everything to the right (inclusive).
        """
        L = self.L
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        # iterate from rightmost to leftmost
        # for i in reversed(range(L)):
        for count, i in enumerate(range(stop, start, -1)):
            
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            # Is there any tensor in tn to be included in env
            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block 

                
            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
            else:
                
                if count==0:
                    indx = psi.bond(stop+1, stop)
                    indx_ = self.tn.bond(stop+1, stop)

                    
                # tie to previously computed right environment
                if env_right[site_tag_id.format(i+1)] is not None:
                    t |= env_right[site_tag_id.format(i+1)]
                    env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
                else:
                    t = t.reindex( {indx:indx_} ) 
                    env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)

    def _left_range(self, psi, site, count, env_left):
        """Update left environment incrementally for current site."""

        # get tensor at stie from p
        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block 
            
        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
        else:
            if count - 1 == 0:
                indx = psi.bond(site-1, site)
                indx_ = self.tn.bond(site-1, site)
                t = t.copy()
                t = t.reindex( {indx:indx_} )
                env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
            else:
                t |= env_left[site_tag_id.format(site-1)]
                env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)



    def _update_env_left(self, psi, site: int, env_left):
        """Update left environment incrementally for current site."""
        
        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block 
            
        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
        else:
            t |= env_left[site_tag_id.format(site-1)]
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)

    
    def run_eff(self, n_iter=6, verbose=True):

        """Run the eefective fitting process"""
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L
        opt = self.opt

        #info_c = self.info_c
        #range_int = self.range_int 


        
        env_left = { site_tag_id.format(i):None   for i in range(psi.L)}
        env_right = { site_tag_id.format(i):None   for i in range(psi.L)}

        
        for iteration in range(n_iter):    
            
            for site in range(L):


                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1
                # Canonicalize psi at the current site

                
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)


                self._build_env_right(psi, env_right) if site == 0 else self._update_env_left(psi, site-1, env_left)
                
                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None
                
                if site == 0:
                    if tn_site:
                        tn =  tn_site | env_right[site_tag_id.format(site+1)]
                    else:
                        tn = env_right[site_tag_id.format(site+1)]
                    
                if site > 0 and site < L-1:
                    if tn_site:
                        tn =  tn_site  |  env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
                    else:
                        tn =  env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
            
                if site == L-1:
                    if tn_site:
                        tn =  tn_site | env_left[site_tag_id.format(site-1)]
                    else:
                        tn =  env_left[site_tag_id.format(site-1)]

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=opt).transpose(*psi[site_tag_id.format(site)].inds)
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)
                

                norm_f = (f.H & f).contract(all) ** 0.5
                self.loss_.append( complex(norm_f).real )

                # Contract and normalize
                # Update tensor data
                psi[site].modify(data=f.data)


            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))



    def run_gate(self, n_iter=6, verbose=True):

        """Run the eefective fitting process"""
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L
        opt = self.opt

        start, stop = self.range_int 
        
        env_left = { site_tag_id.format(i):None   for i in range(psi.L)}
        env_right = { site_tag_id.format(i):None   for i in range(psi.L)}


        for iteration in range(n_iter):    
            
            for i in range(stop, start, -1):
                psi.right_canonize_site(i, bra=None)

            for count_, site in enumerate(range(start, stop+1)):
 
                
                self._right_range(psi, env_right, start, stop) if count_ == 0 else self._left_range(psi, site-1, count_, env_left)

                
                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None

                if site == 0:
                    if tn_site:
                        tn =  tn_site | env_right[site_tag_id.format(site+1)]
                    else:
                        tn = env_right[site_tag_id.format(site+1)]


                
                if site > 0 and site < L-1:

                    # Boundary consistency: the left and right indices must match between tn and p
                    if count_ == 0:
                        indx = psi.bond(start-1, start)
                        indx_ = self.tn.bond(start-1, start)
                        tn_site = tn_site.reindex({indx_:indx})
                    if count_ == stop  - start:
                        indx = psi.bond(stop+1, stop)
                        indx_ = self.tn.bond(stop+1, stop)
                        tn_site = tn_site.reindex({indx_:indx})
                        
                    
                    if tn_site:
                        if env_right[site_tag_id.format(site+1)] is not None and env_left[site_tag_id.format(site-1)] is not None:
                            tn =  tn_site  |  env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
                        elif env_left[site_tag_id.format(site-1)] is not None:
                            tn =  tn_site | env_left[site_tag_id.format(site-1)]
                        elif env_right[site_tag_id.format(site+1)] is not None:
                            tn =  tn_site | env_right[site_tag_id.format(site+1)]
                        else:
                            tn =  tn_site 
                         
                    else:
                        tn = env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
            
                if site == L-1:
                    if tn_site:
                        tn =  tn_site | env_left[site_tag_id.format(site-1)]
                    else:
                        tn =  env_left[site_tag_id.format(site-1)]

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=opt).transpose(*psi[site_tag_id.format(site)].inds)
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)


                norm_f = (f.H & f).contract(all) ** 0.5
                # norm_f = ar.do("norm", f.data)
                
                self.loss_.append( complex(norm_f).real )
                
                # Contract and normalize
                # Update tensor data
                psi[site].modify(data=f.data)

                if site < stop:
                    psi.left_canonize_site(site, bra=None)


            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))




            


class Trck_boundary:

    def __init__(self, tn, opt="auto-hq", chi=4, cutoffs=1.e-10, to_backend=None, to_backend_=None, Lx=4, Ly=4):


        self.tn = tn
        self.opt = opt
        self.chi = chi
        self.to_backend = to_backend
        self.to_backend_ = to_backend_
        self.Lx = Lx
        self.Ly = Ly
        self.cutoffs = cutoffs

        self.mps_b = self._init_left(site_tag_id = "X{}", cut_tag_id = "Y{}")
        self.mps_b |= self._init_right(site_tag_id = "X{}", cut_tag_id = "Y{}")
        #self.mps_b |= self._init_right(site_tag_id = "Y{}", cut_tag_id = "X{}")
        #self.mps_b |= self._init_left(site_tag_id = "Y{}", cut_tag_id = "X{}")

        self.mpo_b = self._init_left_(site_tag_id = "X{}", cut_tag_id = "Y{}")
        #self.mpo_b |= self._init_left_(site_tag_id = "Y{}", cut_tag_id = "X{}")
        #self.mpo_b |= self._init_right_(site_tag_id = "Y{}", cut_tag_id = "X{}")
        self.mpo_b |= self._init_right_(site_tag_id = "X{}", cut_tag_id = "Y{}")



    
    def _init_left(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
        
        p_b_left = {}
        
        for count in range(self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
            
            if count == 0:
                mps = tn
            else:
                mps = tn | p_b_left[cut_tag_id.format(count-1) + "_l"]
        
            
            outer_inds = mps.outer_inds()
            L_mps = len(outer_inds)
        
            regex = re.compile("^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$")
            site_tags = [t for t in mps.tags if regex.match(t)]
            numbers = [int(regex.match(t).group(1)) for t in mps.tags if regex.match(t)]
            
            # print(site_tags, numbers)
        
            
            inds_ = []
            inds_size = {}
            for tag in site_tags:
                tn_select = mps.select(tag)
                inds_local = []
                for j_ in tn_select.outer_inds():
                    if j_ in outer_inds:
                        inds_local.append(j_)
                        #print( mps.ind_size(j_) )
                        inds_size |= {j_:mps.ind_size(j_)}
                inds_.append(inds_local)
        
        
            inds_k = {}
            count_ = 0
            for inds in inds_:
                for indx in inds:
                    inds_k |= {f"k{count_}":indx}  
                    count_ += 1
        
        
            # create the nodes, by default just the scalar 1.0
            tensors = [qtn.Tensor() for _ in range(L_mps)]
            
            for i_ in range(L_mps):
                if i_ < (L_mps-1):
                    # add the physical indices, each of size 2
                    tensors[i_].new_ind(f'k{i_}', size=inds_size[inds_k[f"k{i_}"]])
                    tensors[i_].add_tag(f"I{i_}")
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)
                if i_ == L_mps-1:
                    # add the physical indices, each of size 2
                    tensors[i_].new_ind(f'k{i_}', size=inds_size[inds_k[f"k{i_}"]])
                    tensors[i_].add_tag(f"I{i_}")
        
                    
            p = qtn.TensorNetwork(tensors)
            p.reindex_(inds_k)
            p.view_as_(qtn.MatrixProductState, L = L_mps, site_tag_id="I{}", site_ind_id=None, cyclic=False)
            p.apply_to_arrays(self.to_backend_)
            p.randomize( seed=20, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi, cutoff=self.cutoffs)
            p.normalize()
            p_b_left[cut_tag_id.format(count) + "_l"] = p    
    
    
        return p_b_left



    def _init_right(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
    
        p_b_right = {}
        
        # iterate from Ly-1 down to 0 (inclusive)
        for count in range(0, self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(self.Ly - 1 - count), "any").copy()
        
            # first iteration (top row) -> initialize; otherwise attach previously built block
            if count == 0:
                mps = tn
            else:
                # when going downward, previously-built block is at count+1
                
                mps = tn | p_b_right[cut_tag_id.format(count - 1) + "_r"]
        
            outer_inds = mps.outer_inds()
            L_mps = len(outer_inds)
        
            # Build regex to capture the integer inside site_tag_id, e.g. "X(\d+)"
            regex = re.compile("^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$")
        
            regex = re.compile("^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$")
            site_tags = [t for t in mps.tags if regex.match(t)]
            numbers = [int(regex.match(t).group(1)) for t in mps.tags if regex.match(t)]
        
            # Build index lists and sizes for the selected site tags
            inds_ = []
            inds_size = {}
            for tag in site_tags:
                tn_select = mps.select(tag)
                inds_local = []
                for j_ in tn_select.outer_inds():
                    if j_ in outer_inds:
                        inds_local.append(j_)
                        inds_size[j_] = mps.ind_size(j_)
                inds_.append(inds_local)
        
            # flatten inds -> new names k0, k1, ... (keeps order consistent with sorted site_tags)
            inds_k = {}
            count_ = 0
            for inds in inds_:
                for indx in inds:
                    inds_k[f"k{count_}"] = indx
                    count_ += 1
        
            # create tensors (one per outer index)
            tensors = [qtn.Tensor() for _ in range(L_mps)]
        
            for i_ in range(L_mps):
                # create physical index k{i_} with recorded size
                tensors[i_].new_ind(f'k{i_}', size=inds_size[inds_k[f"k{i_}"]])
                tensors[i_].add_tag(f"I{i_}")
                # create left-to-right bonds: bond i_ <-> i_+1 (same as before)
                if i_ < (L_mps - 1):
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)
        
            p = qtn.TensorNetwork(tensors)
            p.reindex_(inds_k)
            p.view_as_(qtn.MatrixProductState, L=L_mps, site_tag_id="I{}", site_ind_id=None, cyclic=False)
            p.apply_to_arrays(self.to_backend_)
            p.randomize(seed=20, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi, cutoff=self.cutoffs)
        
            # store the block at the current cut tag
            p.normalize()
            p_b_right[cut_tag_id.format(count) + "_r"] = p
    
    
        return p_b_right
    
    def _init_left_(self, site_tag_id = "X{}", cut_tag_id = "Y{}", chi_1=2, chi_2=2):
        
    
        mps_b = {}
        
        
        for count in range(self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
            tn.compress_all(inplace=True, **{"max_bond":chi_1, "canonize_distance": 2, "cutoff":1.e-12})
        
            
            if count == 0:
                mps = tn
            else:
                mps_ = mps_b[cut_tag_id.format(count-1) + "_l"].copy()
                mps_.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
                mps = tn | mps_
                mps.drop_tags(cut_tag_id.format(count-1))
        
        
        
            for i in range(self.Lx): 
                mps.contract_tags_(
                                    site_tag_id.format(i), optimize=self.opt)
            
            mps.fuse_multibonds_()
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
        
            mps.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
            
            mps.apply_to_arrays(self.to_backend_)
        
            mps.expand_bond_dimension(self.chi, 
                                      rand_strength=0.01
                                     )
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
            mps.normalize()
            mps.apply_to_arrays(self.to_backend)
        
            # mps.draw([f"X{i}" for i in range(Lx)], show_inds="bond-size", show_tags=False)
        
            mps_b[cut_tag_id.format(count) + "_l"] = mps
        return mps_b

    
    
    def _init_right_(self, site_tag_id = "X{}", cut_tag_id = "Y{}", chi_1=2, chi_2=2):
        mps_b = {}
        
        
        for count in range(self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(self.Ly - 1 - count), "any")
            tn = tn.copy()
            tn.compress_all(inplace=True, **{"max_bond":chi_1, "canonize_distance": 4, "cutoff":1.e-12})
        
            
            if count == 0:
                mps = tn
            else:
                mps_ = mps_b[cut_tag_id.format(count-1) + "_r"].copy()
                mps_.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
                mps = tn | mps_
                mps.drop_tags(cut_tag_id.format(self.Ly-1-count+1))
        
        
        
            for i in range(self.Lx): 
                mps.contract_tags_(
                                    site_tag_id.format(i), optimize=self.opt)
            
            mps.fuse_multibonds_()
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
        
            mps.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
            
            mps.apply_to_arrays(self.to_backend_)
        
            mps.expand_bond_dimension(self.chi, 
                                      rand_strength=0.01
                                     )
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)

            mps.normalize()
            mps.apply_to_arrays(self.to_backend)
        
            # mps.draw([f"X{i}" for i in range(Lx)], show_inds="bond-size", show_tags=False)
        
            mps_b[cut_tag_id.format(count) + "_r"] = mps
        return mps_b
