'''
Functions to find the dominant
set in a graph

see https://crimede-coder.com/graphs/network
for peer review references and web-app
'''

import networkx as nx
import itertools
import ipycytoscape

# Function to turn dataframe into how you want it formatted
# for adding nodes

def add_nodes(G,data,idv='Id'):
    lv = list(data.set_index(idv).to_dict(orient='index').items())
    G.add_nodes_from(lv)


# default colors/shapes
# for the called in graph

dlab = {'Called In': {'node_color': 'red',
                      'node_shape': 's',
                      'node_size': 400,
                       'edgecolors': 'k'},
        'Reached': {'node_color': 'pink',
                      'node_shape': 's',
                      'node_size': 350,
                      'edgecolors': 'k'},
        'Left Over': {'node_color': '#286090',
                      'node_shape': 'o',
                      'node_size': 300,
                      'edgecolors': 'k'}}


# Function to color based on called in
def color(G,called_in,pos,lab=dlab,ax=None):
    # Figure out the reached
    reached = []
    for c in called_in:
        reached += list(G[c].keys())
    reached = list(set(reached) - set(called_in))
    leftover = set(G.nodes) - set(reached) - set(called_in)
    nx.draw_networkx_nodes(G,pos,nodelist=called_in,**dlab['Called In'],label='Called In',ax=ax)
    nx.draw_networkx_nodes(G,pos,nodelist=reached,**dlab['Reached'],label='Reached',ax=ax)
    nx.draw_networkx_nodes(G,pos,nodelist=leftover,**dlab['Left Over'],label='Left Over',ax=ax)
    nx.draw_networkx_edges(G, pos)

#Function needs to distinguish among ties by decreases in set
#First one with the max new set wins
def DegTieBreak(G,neighSet,nbunch):
    maxN = -1
    for i in nbunch:
        neigh_cur = set(G[i].keys())
        dif = (neigh_cur | set([i]) ) - neighSet
        te = len(dif)
        if te > maxN:
            myL = [i,neigh_cur,dif]
            maxN = te
    return myL


def MaxDegSub(G,onlyLook):
    vals = nx.degree_centrality(G).items()
    #strip items that are not in onlyLook
    valsSub = [i for i in vals if i[0] in onlyLook]
    valsOnly = [i[1] for i in valsSub]
    max_val = max(valsOnly)
    all_max = [i[0] for i in valsSub if i[1] == max_val]  
    return all_max

#function with a restricted set of matches (eg return only those under supervision)
#only need to update subfunctions MaxDeg, just supply onlyLook with a list
#returns in the best order again
def domSet_WheSub(G,onlyLook,total=None):
    uG = G.copy() #make a deepcopy of the orig graph to update for the algorithm
    domSet = []      #list to place dominant set
    neighSet = set([])    #list of neighbors to dominating set   
    if not total:
        loop_num = len(onlyLook) #total is the set maximum number of loops for graph
    else:                        #default as many nodes in graph               
        loop_num = total         #can also set a lower limit though
    for i in range(loop_num):     
        nodes_sel = MaxDegSub(G=uG,onlyLook=onlyLook)   #select nodes from updated graph with max degree centrality
        #chooses among degree ties with the maximum set of new neighbors
        if len(nodes_sel) > 1:
            temp = DegTieBreak(G=uG,neighSet=neighSet,nbunch=nodes_sel)
            node_sel = temp[0]
            neigh_cur = temp[1]
            newR = temp[2]
        else:
            node_sel = nodes_sel[0]
            neigh_cur = set(uG[node_sel].keys()) #neighbors of the current node
            newR = neigh_cur - neighSet          #new neighbors added in
        domSet.append(node_sel) #append that node to domSet list
        #break loop if dominant set found, else decrement counter
        if nx.is_dominating_set(G,domSet):
            break
        #should only bother to do this junk if dominant set has not been found!
        uG.remove_node(node_sel)  #remove node from updated graph
        #now this part does two loops to remove the edges between reached nodes
        #one for all pairwise combinations of the new reached nodes, the second
        #for the product of all new reached nodes compared to prior reached nodes
        #new nodes that have been reached
        for i in itertools.combinations(newR,2):
            if uG.has_edge(*i):
                uG.remove_edge(*i)       
        #product of new nodes and old neighbor set
        #this loop could be pretty costly in big networks, consider dropping
        #should not make much of a difference
        for j in itertools.product(newR,neighSet):
            if uG.has_edge(*j):
                uG.remove_edge(*j)
        #now update the neighSet to include newnodes, but strip nodes that are in the dom set
        #since they are pruned from graph, all of their edges are gone as well
        neighSet = (newR | neighSet) - set(domSet)
    return domSet


#alternate where pruning nodes that have the most remainder in the graph instead of choosing by maximum degree
def domSet_Whe2(G):
    uG = G.copy() #make a deepcopy of the orig graph to update for the algorithm
    domSet = []      #list to place dominant set
    neighSet = set([])    #list of neighbors to dominating set   
    fullNodes = set(nx.nodes(G))
    while nx.is_dominating_set(G,domSet) == False:    #?can also set a max number of loops for large graphs?
        rem_nodes = fullNodes - neighSet - set(domSet)
        temp = DegTieBreak(G=uG,neighSet=neighSet,nbunch=rem_nodes)
        node_sel = temp[0]
        neigh_cur = temp[1]
        newR = temp[2]   
        domSet.append(node_sel) #append that node to domSet list
        uG.remove_node(node_sel)  #remove node from updated graph
        #now update the neighSet to include newnodes, pruning edges not necessary
        neighSet = neigh_cur | neighSet
    return domSet


call_in_style = [
   {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '10px',
            'width': '30px',
            'height': '30px',
            'border-width': '2px',
            'border-color': 'black'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'width': 2,
            'line-color': '#9CA8B3'
        }
    },
    {
        'selector': 'node[category="called-in"]',
        'style': {
            'shape': 'square',
            'background-color': 'red',
            'width': '40px',
            'height': '40px'
        }
    },
    {
        'selector': 'node[category="reached"]',
        'style': {
            'shape': 'square',
            'background-color': 'pink',
            'width': '35px',
            'height': '35px',
        }
    },
    {
        'selector': 'node[category="leftover"]',
        'style': {
            'background-color': 'lightblue',
            'width': '20px',
            'height': '20px',
        }
    }
]


# This is for ipycytoscape
def color_cytoscape(G,called_in,styles=call_in_style,layout='cose'):
    # Figure out the reached
    reached = []
    for c in called_in:
        reached += list(G[c].keys())
    reached = list(set(reached) - set(called_in))
    leftover = set(G.nodes) - set(reached) - set(called_in)
    # remake a new graph
    G2 = G.copy()
    # set attributes for those in each list
    for n in G2.nodes():
        if n in called_in:
            G2.nodes[n]['category'] = 'called-in'
        elif n in reached:
            G2.nodes[n]['category'] = 'reached'
        elif n in leftover:
            G2.nodes[n]['category'] = 'leftover'
    cytoscape_widget = ipycytoscape.CytoscapeWidget()
    cytoscape_widget.graph.add_graph_from_networkx(G2)
    cytoscape_widget.set_style(call_in_style)
    cytoscape_widget.set_layout(name=layout)
    return cytoscape_widget
