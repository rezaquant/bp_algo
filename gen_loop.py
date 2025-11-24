from itertools import combinations, chain
from collections import Counter

def _canonical_edge(e):
    # ensure each edge is in sorted order and hashable
    return tuple(sorted(e))

def _canonical_combo(combo):
    # canonical tuple of canonical edges (sorted for stable hashing)
    return tuple(sorted((_canonical_edge(e) for e in combo)))

def filter_loopy_combin(combos, sites=None):
    """Keep combos that are 'loops':
       - If no singletons (all counts ∈ {0,2,4,...}), keep it.
       - If singletons exist, keep only if all singletons ⊆ sites.
    """
    sites_set = set(sites or [])
    keep = []

    for combo in combos:
        # Flatten once and count
        counts = Counter(chain.from_iterable(combo))
        singles = {x for x, c in counts.items() if c == 1}

        if not singles:
            # no dangling sites -> valid loop
            keep.append(combo)
        else:
            # dangling sites allowed only if all lie within sites
            if sites_set and singles.issubset(sites_set):
                keep.append(combo)

    # canonicalize (sorted edges, then sorted combo) for stable representation
    return [ _canonical_combo(c) for c in keep ]


def combine_elements(edges, length, sites=None, loopy_filter=True):
    """Generate combinations of edges of given length; optionally filter to 'loops'."""
    edges = [_canonical_edge(e) for e in edges]
    combos = combinations(edges, length)

    if loopy_filter:
        combos = filter_loopy_combin(combos, sites=sites)
    else:
        # still canonicalize for consistency
        combos = [ _canonical_combo(c) for c in combos ]

    # Special case: length == 0 -> one empty combo
    if length == 0:
        return [tuple()]

    return combos


def loop_gen_local(bp, circum=None, tags_cluster=None, sites=None, circum_g=None,
                   tn_flat=None, site_tags=None, intersect=True,
                   print_=False, loopy_filter=True, str_="R"):
    """Generate local/global loop candidates and dedupe."""
    tags_cluster = set(tags_cluster or [])
    site_tags = site_tags or []
    sites = set(sites or [])

    # all edges in tn, canonicalize pairs
    edges = [ _canonical_edge(e) for e in bp.edges.keys() ]

    # optional: restrict to a cluster (both endpoints inside cluster)
    if tags_cluster:
        edges = [e for e in edges if (e[0] in tags_cluster and e[1] in tags_cluster)]

    if print_:
        print("max: number of edges:", len(edges))

    # Local loops
    loops_local = []
    if circum is not None:
        loops_local = combine_elements(edges, circum, sites=sites, loopy_filter=loopy_filter)


    # Deduplicate (works because we canonicalize combos)
    loops = set(loops_local) 
    return list(loops)



def smart_tag_cluster(tn, tag_, fix ={}, max_distance=1, fillin=1, site_tags=None, draw_=False):

    tn_local = tn.select_local(
            tag_,
            which='any',
            max_distance=1,
            fillin=1,
            virtual=True,
            include=None,
            exclude=None,
        )
    inds = tn_local.inner_inds()
    tags = [tag for tag in tn_local.tags if tag in site_tags]
    
    if draw_:
        tn.draw(
            tag_ + tags,
            fix=fix,
            node_hatch={tag: '////' for tag in tag_},
            node_shape={tag: "H" for tag in tag_},
            legend=False,
            node_scale=1.0,
            highlight_inds=inds,
            show_tags=False,
            figsize=(8, 8),
        )

    return tags