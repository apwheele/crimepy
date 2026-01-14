import networkx as nx
from crimepy.domset import domSet_WheSub

def test_domset_whesub_chain():
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
    dom = domSet_WheSub(G,['A', 'D'])
    dom.sort()
    assert len(list(dom)) == 2
    assert dom == ['A', 'D']