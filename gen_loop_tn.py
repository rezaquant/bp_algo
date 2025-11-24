from itertools import chain
import quimb as qu
import quimb.tensor as qtn
from tqdm import tqdm
import numpy as np
import autoray as ar


def bp_excited_loops(tn_, bp_, loop_l, progbar=False, chi=None, max_repeats_=2**6, parallel=True, 
                     contract_=True, pari_exclusive=[], tree_gauge_distance=4, f_max = 15 , peak_max=34,
                    prgbar=True, opt="auto-hq", simplify_=False, 
                    ):
    
    is_subset = False

    res = {}
    excit_ = []
    tn_l = []
    inds_excit = []
    inds_gs = []
    tags_excit = []
    # put all regions in below list
    pair_gs_l = []
    pair_excited_l = []
    
    for idx, loop in enumerate(loop_l):
        loop = [tuple(sorted(pair)) for pair in loop]
        pair_excited = loop
        
        if pair_excited:
            elem0, elem1 = pair_excited[0]
            if not isinstance(elem0, str):
                format_string = "I" + ",".join(["{}"] * len(elem0))
                pair_excited = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_excited]
           
        all_edges = [tuple(sorted(i)) for i in bp_.edges.keys()]
        pair_gs = list(set(all_edges) - set(pair_excited))
        if pari_exclusive:
            
            if not set(pair_excited):
                # if pair_excited = empty that means "gs bp"  that we want to include
                is_subset = False
            else:
                # if pair_excited is in pari_exclusive we want to ignore to not include "double counting"
                is_subset = set(pair_excited).issubset(set(pari_exclusive))
            
            
            #remove the exclusive pairs from the both regions 
            pair_excited = list(set(pair_excited)-set(pari_exclusive))
            pair_gs = list(set(pair_gs)-set(pari_exclusive))

        if not is_subset:
            pair_gs_l.append(pair_gs)
            pair_excited_l.append(pair_excited)





    # eliminate the pairs that are common 

    # Helper function to normalize a sublist
    def normalize_sublist(sublist):
        return tuple(sorted(tuple(sorted(tup)) for tup in sublist))
    
    # Helper function to normalize a pair of sublists
    def normalize_pair(sublist1, sublist2):
        return (normalize_sublist(sublist1), normalize_sublist(sublist2))
    
    # Track unique pairs with a set
    unique_pairs = set()
    unique_list1 = []
    unique_list2 = []
    
    # Process each pair of sublists
    for sublist1, sublist2 in zip(pair_gs_l, pair_excited_l):
        normalized_pair = normalize_pair(sublist1, sublist2)
        # Add to unique lists only if the normalized pair hasn't been seen
        if normalized_pair not in unique_pairs:
            unique_pairs.add(normalized_pair)
            unique_list1.append(sublist1)
            unique_list2.append(sublist2)

    
    #print("redunction-->", len(pair_excited_l), len(unique_list1))
    pair_gs_l = unique_list1
    pair_excited_l = unique_list2


    for pair in pair_excited_l:
        pair = list(chain.from_iterable(pair))
        tags_excit.append(pair)


    flops_l = []
    peak_l = []
    with tqdm(total=len(pair_excited_l),  desc="bp:", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:
    
    
        for count in range(len(pair_excited_l)):
            pair_excited = pair_excited_l[count]
            pair_gs = pair_gs_l[count]
            inds_local = []
            inds_local_ = []
    
            if pair_excited:
                for pair in pair_excited:
                    inds = bp_.messages[pair].inds
                    for indx in inds:
                        inds_local_.append(indx)
            inds_local_ = list(set(inds_local_))
            
            
            if pair_gs:
                for pair in pair_gs:
                    inds = bp_.messages[pair].inds
                    for indx in inds:
                        inds_local.append(indx)
            inds_local = list(set(inds_local))       
             
            inds_excit.append(inds_local_)
            inds_gs.append(inds_local)
            
            tn_appro = tn_bp_excitations(tn_, bp_, pair_excited, pair_gs, loop=[], opt=opt) 
            
    
            
            
            
            tn_l.append(tn_appro.copy())

            if simplify_:       
             tn_appro.full_simplify_(seq='R', split_method='svd', inplace=True)
            flops, peak = (1,1)

            
            if contract_:
                excited = tn_appro.contract(all, optimize=opt)
                excit_.append(excited)
        
            pbar.set_postfix({"flops": np.log10(flops), 
                              "peak": peak,
                              })
            pbar.refresh()
            pbar.update(1)

    res |= { "tn_l":tn_l, "excited":excit_,  "peak":peak_l, "flops":flops_l, "tags_excit":tags_excit, "inds_excit":inds_excit, "inds_gs":inds_gs }
    return res

    

def tn_bp_excitations(tn, bp, pair_excited=[], pair_gs=[], loop =[], opt="auto-hq"):
    
    # copy the tensor network                 
    tn_appro = tn.copy()
    
    # get projectors into excited states of bp
    pr_excited = bp.projects

    # get redundancy in the tag pairs:
    pair_excit_tags = list(set(pair_excited))
    pair_gs_tags = list(set(pair_gs))

    # if format is not str 
    if pair_excit_tags:
        elem0, elem1 = pair_excit_tags[0]
        format_string = "I" + ",".join(["{}"] * len(elem0))

        if not isinstance(elem0, str):
            pair_excit_tags = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_excit_tags]

    if pair_gs_tags:
        elem0, elem1 = pair_gs_tags[0]
        format_string = "I" + ",".join(["{}"] * len(elem0))

        if not isinstance(elem0, str):
            pair_gs_tags = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_gs_tags]


    
    if set(pair_excit_tags).intersection(pair_gs_tags):
        print("warnning: common pair in bp gs and excited state")
        print("warnning: decide about gs or excitation" )

    
    
    for pair in pair_excit_tags:
        
        pair_ = sorted(pair)
        #print("excited", pair_)
        i, j = pair_
        
        pr_excited_tn = pr_excited[tuple(pair_)].copy()
    
        # drop indicies in the right tensor and attach those indicies to pr_excited_tn
        tn_j = tn_appro.select(j)
        #tn_j = bp.local_tns[j].copy()


        
        left_inds_ = pr_excited_tn.left_inds
        inds_ = pr_excited_tn.inds
        right_inds_ = [i for i in inds_ if i not in left_inds_]

        map_inds = { left_inds_[count]:right_inds_[count]      for count in range(len(right_inds_)) }
        tn_j.reindex_(map_inds)

    
        tn_appro |= pr_excited_tn
    


    
    #print(tn_appro.draw("Me"))
    for pair in pair_gs_tags:
        pair_ = sorted(pair)
        #print("excited", pair_)
        i, j = pair_
        #select tensors with tag i and j and their common indicies
        tn_i = tn_appro.select(i)
        #tn_i = bp.local_tns[i]
        tn_j = tn_appro.select(j)
        #tn_j = bp.local_tns[j]

        
        # Get massages going to tensors with tag i (from j) and tensors with tag j (from i)
        mij = bp.messages[j,i].copy()
        mji = bp.messages[i,j].copy()
        inds_ij = list(mij.inds)
        
        #drop the bond between tensor_i and tensor_j and add massages m_ij to tensor_i and massage m_ji to tensor j
        map_inds_i = { i:qtn.rand_uuid() for i in inds_ij }
        map_inds_j = { i:qtn.rand_uuid() for i in inds_ij }
        
        mij.reindex_(map_inds_i)
        tn_i.reindex_(map_inds_i)
        
        mji.reindex_(map_inds_j)
        tn_j.reindex_(map_inds_j)
    
        # add some tags to massages
        mij.add_tag(["in", "Mg"])
        mji.add_tag(["out", "Mg"])
        
        tn_appro |=  (mij | mji)

    #print(tn_appro.contract(all, optimize=opt))
    return tn_appro




def bp_excited_loops_(tn_, bp_, loop_l, progbar=False, chi=None, max_repeats_=2**6, parallel=True, 
                     contract_=True, pari_exclusive=[], tree_gauge_distance=4, f_max=15 , peak_max=34,
                      prgbar=False, opt="auto-hq", inplace=False, simplify=False,
                    ):
    

    pari_exclusive_flat = pari_exclusive

    tn_ = tn_ if inplace else tn_.copy()
    
    is_subset = False
    res = {}
    excit_ = []
    tn_l = []
    inds_excit = []
    inds_gs = []
    
    # put all regions in the list below
    pair_gs_l = []
    pair_excited_l = []
    
    for idx, loop in enumerate(loop_l):
        loop = [tuple(sorted(pair)) for pair in loop]
        pair_excited = loop
        pair_excited_flat = list(chain.from_iterable(pair_excited))
        
        if pair_excited:
            elem0, elem1 = pair_excited[0]
            if not isinstance(elem0, str):
                format_string = "I" + ",".join(["{}"] * len(elem0))
                pair_excited = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_excited]
           
        
        
        # Ground-state massage pairs should not overlap with excited-state massage pairs.
        

        if pair_excited:
            all_edges = [tuple(sorted(i)) for i in bp_.edges.keys()]
            pair_gs = list(set(all_edges) - set(pair_excited))
            # Ground-state messages are valid only when there is a common area defined by excited-state messages
            pair_excited_flat = list(chain.from_iterable(pair_excited))
            pair_gs = [tup for tup in pair_gs if any(item in pair_excited_flat for item in tup)]
            # make sure there is no overlap with excited-state massage pairs: double-check
            pair_gs = list(set(pair_gs) - set(pair_excited))
        
        else:
            # Notice: if there is no pair_excited, include all gs-bp massages in the TN to fom ground-state TN == 1
            all_edges = [tuple(sorted(i)) for i in bp_.edges.keys()]
            pair_gs = list(set(all_edges))      

    
        if pari_exclusive:
            if not set(pair_excited):
                # if pair_excited = empty it means "gs bp"  that we want to include: proceed 
                is_subset = False
            else:
                # if pair_excited is in pari_exclusive we want to ignore to not include "double counting": break for loop
                is_subset = set(pair_excited).issubset(set(pari_exclusive))
            
    

            
            if pair_excited:
                #remove the exclusive pairs from both regions 
                pair_excited = list(set(pair_excited)-set(pari_exclusive))
                pair_excited_flat = list(chain.from_iterable(pair_excited))
    
               
                # Ground-state messages are valid only when there is a common area defined by excited/exclusive messages
                pair_flat = pari_exclusive_flat + pair_excited_flat
                pair_gs = list(set(all_edges) - set(pair_excited))
                pair_gs = [tup for tup in pair_gs if any(item in pair_flat for item in tup)]
                # make sure there is no common gs pair in both pair_excited or pari_exclusive
                pair_gs = list(set(pair_gs) - set(pair_excited))
                pair_gs = list(set(pair_gs) - set(pari_exclusive))
            else:
                # include all gs-bp massages, except the ones in pari_exclusive
                pair_gs = list(set(all_edges) - set(pair_excited))
                pair_gs = list(set(pair_gs) - set(pari_exclusive))


        # add excited/gs pairs into the list
        if not is_subset:
            pair_gs_l.append(pair_gs)
            pair_excited_l.append(pair_excited)







    # eliminate the pairs that are common 

    # Helper function to normalize a sublist
    def normalize_sublist(sublist):
        return tuple(sorted(tuple(sorted(tup)) for tup in sublist))
    
    # Helper function to normalize a pair of sublists
    def normalize_pair(sublist1, sublist2):
        return (normalize_sublist(sublist1), normalize_sublist(sublist2))
    
    # Track unique pairs with a set
    unique_pairs = set()
    unique_list1 = []
    unique_list2 = []
    
    # Process each pair of sublists
    for sublist1, sublist2 in zip(pair_gs_l, pair_excited_l):
        normalized_pair = normalize_pair(sublist1, sublist2)
        # Add to unique lists only if the normalized pair hasn't been seen
        if normalized_pair not in unique_pairs:
            unique_pairs.add(normalized_pair)
            unique_list1.append(sublist1)
            unique_list2.append(sublist2)

    
    #print("reduction-->", len(pair_excited_l), len(unique_list1))
    
    
    pair_gs_l = unique_list1
    pair_excited_l = unique_list2

    tags_excit = []
    
    # that is an important function that includes site tags that are defined loop:
    for count, pair in enumerate(pair_excited_l):
        
        if pair:
            pair = list(chain.from_iterable(pair))
            tags_excit.append(pair+pari_exclusive_flat+["proj"])
        
        # In the rare case of no excitation loop, I want to include all tags
        else:
            pair = pair_gs_l[count]
            pair = list(chain.from_iterable(pair))
            if pari_exclusive:
                tags_excit.append(pair+pari_exclusive_flat)
            else:
                tags_excit.append(pair)



    flops_l = []
    peak_l = []
    with tqdm(total=len(pair_excited_l),  desc="sum-loop", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:
    
    
        for count in range(len(pair_excited_l)):
            pair_excited = pair_excited_l[count]
            pair_gs = pair_gs_l[count]
            inds_local = []
            inds_local_ = []
    
            if pair_excited:
                for pair in pair_excited:
                    inds = bp_.messages[pair].inds
                    for indx in inds:
                        inds_local_.append(indx)
            inds_local_ = list(set(inds_local_))
            
            
            if pair_gs:
                for pair in pair_gs:
                    inds = bp_.messages[pair].inds
                    for indx in inds:
                        inds_local.append(indx)
            inds_local = list(set(inds_local))       
             
            inds_excit.append(inds_local_)
            inds_gs.append(inds_local)
            
            tn_appro = tn_bp_excitations_(tn_, bp_, pair_excited, pair_gs, loop=[], opt=opt) 
            

            if tags_excit[count]:
                exponent = tn_appro.exponent
                tn_appro = tn_appro.select(tags_excit[count], which="any")
                tn_appro.exponent = exponent
                
            tn_l.append(tn_appro.copy())
            
            if simplify:            
                tn_appro.full_simplify_(seq='R', split_method='svd', inplace=True)
            
            
            flops, peak = (1,0)
            if contract_:
                tree = tn_appro.contraction_tree(opt)
                flops = tree.contraction_cost()
                peak = tree.peak_size(log=2)
                if flops<1:
                    flops = 1
                flops = np.log10(flops)
            
                excited = tn_appro.contract(all, optimize=opt)
                excited = excited
                excit_.append(excited)
                flops_l.append(flops)
                peak_l.append(peak)
        
            pbar.set_postfix({"flops": flops, 
                              "peak": peak,
                              })
            pbar.refresh()
            pbar.update(1)

    
    
    res |= { "tn_l":tn_l, "excited":excit_,  "peak":peak_l, "flops":flops_l, "tags_excit":tags_excit, "inds_excit":inds_excit, "inds_gs":inds_gs }
    
    
    return res





def tn_bp_excitations_(tn, bp, pair_excited=[], pair_gs=[], loop =[], opt="auto-hq"):
    
    # copy the tensor network                 
    tn_appro = tn.copy()
    
    # get projectors into excited states of bp
    pr_excited = bp.projects

    # get redundancy in the tag pairs:
    pair_excit_tags = list(set(pair_excited))
    pair_gs_tags = list(set(pair_gs))

    # if format is not str 
    if pair_excit_tags:
        elem0, elem1 = pair_excit_tags[0]
        format_string = "I" + ",".join(["{}"] * len(elem0))

        if not isinstance(elem0, str):
            pair_excit_tags = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_excit_tags]

    if pair_gs_tags:
        elem0, elem1 = pair_gs_tags[0]
        format_string = "I" + ",".join(["{}"] * len(elem0))

        if not isinstance(elem0, str):
            pair_gs_tags = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_gs_tags]


    
    if set(pair_excit_tags).intersection(pair_gs_tags):
        print("warnning: common pair in bp gs and excited state")
        print("warnning: decide about gs or excitation" )


    for pair in pair_excit_tags:
        
        pair_ = sorted(pair)
        #print("excited", pair_)
        i, j = pair_
        
        pr_excited_tn = pr_excited[tuple(pair_)].copy()
    
        # drop indicies in the right tensor and attach those indicies to pr_excited_tn
        tn_j = tn_appro.select(j)
        left_inds_ = pr_excited_tn.left_inds
        inds_ = pr_excited_tn.inds
        right_inds_ = [i for i in inds_ if i not in left_inds_]

        map_inds = { left_inds_[count]:right_inds_[count]      for count in range(len(right_inds_)) }
        tn_j.reindex_(map_inds)


        tn_appro |= pr_excited_tn



    #print(tn_appro.draw("Me"))
    for pair in pair_gs_tags:
        pair_ = sorted(pair)
        #print("excited", pair_)
        i, j = pair_
        #select tensors with tag i and j and their common indicies
        tn_i = tn_appro.select(i)
        #tn_i = bp.local_tns[i]

        tn_j = tn_appro.select(j)

        # Get massages going to tensors with tag i (from j) and tensors with tag j (from i)
        mij = bp.messages[j,i].copy()
        mji = bp.messages[i,j].copy()
        inds_ij = list(mij.inds)

        #drop the bond between tensor_i and tensor_j and add massages m_ij to tensor_i and massage m_ji to tensor j
        map_inds_i = { i:qtn.rand_uuid() for i in inds_ij }
        map_inds_j = { i:qtn.rand_uuid() for i in inds_ij }

        mij.reindex_(map_inds_i)
        tn_i.reindex_(map_inds_i)

        mji.reindex_(map_inds_j)
        tn_j.reindex_(map_inds_j)

        # add some tags to massages
        mij.add_tag(["in", "Mg"])
        mji.add_tag(["out", "Mg"])

        tn_appro |= ( mij | mji )

    #print(tn_appro.contract(all, optimize=opt))
    
    return tn_appro

def env_rho(info_pass, tree_gauge_distance=4, f_max = 14 , peak_max=34, prgbar=True, simplify=False,
            draw_=False, opt="auto-hq"):

    tn_l = info_pass["tn_l"]
    reg_reindex = info_pass["reg_reindex"]
    reg_tags = info_pass["reg_tags"]
    inds_rho = info_pass["inds_rho"]

    leftinds_rho = info_pass["leftinds_rho"]
    rightinds_rho = info_pass["rightinds_rho"]
    
    rho_l = []
    rhodata_l = []
    flops_l = []
    peak_l = []
    tn_rho_l = []



    for count, tn_ in enumerate(tn_l):
        
        tn_ = tn_.copy()
    
        if reg_reindex:
            tn_bra = tn_.select(["BRA"], which="any")
            tn_bra.reindex_(reg_reindex)
        if draw_:
            print("--------------------------")
            tn_.draw(["proj"]+["Mg"], 
                     edge_alpha=1.0, edge_scale=1.0, 
                fix=info_pass["fix"], node_outline_darkness=0.20, node_outline_size=1.0,     
                edge_color='gray', highlight_inds_color="darkred",
                     show_tags=False, legend=False, node_scale=1.2, figsize=(4,4))
            print("--------------------------")

        if simplify:
            tn_.full_simplify_(seq='R', split_method='svd', inplace=True)
    
   
        rho = tn_.contract(optimize=opt, output_inds=inds_rho)


        rho_l.append(rho)
        
            

    return rho_l
    


def rho_bp_excited_loops_(tn, bp, pass_rho, edges_f, obs_tensor, opt="auto-hq", draw_=False, simplify=False,
                          pari_exclusive=None, progbar=False):
    

    res = bp_excited_loops_(tn, bp, edges_f, contract_=False, pari_exclusive=pari_exclusive, simplify=simplify, progbar=progbar)
    pass_rho |= {"tn_l": res["tn_l"],"inds_excit": res["inds_excit"],"tags_excit": res["tags_excit"]}
    rho_l = env_rho(pass_rho, draw_= draw_, opt=opt, prgbar=progbar, simplify=simplify)
    
    obs_ = 0
    norm_ = 0
    for rho in rho_l:
        val = (rho | obs_tensor).contract(all, optimize=opt)
        val_ = rho.trace(pass_rho['leftinds_rho'], pass_rho['rightinds_rho'])
        norm_ += val_
        obs_ += val
    
    return obs_, norm_
