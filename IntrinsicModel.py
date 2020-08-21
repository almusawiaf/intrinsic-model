import networkx as nx

def start_simulation():
    
    for i in range(5):
        net = datasets[i]
        G = nx.read_adjlist(spath + net, create_using = nx.DiGraph(), nodetype = int)
        new_approach(G, spath + "\Results" + net)
    
def new_approach(G, savein):
    print(info(G))
    xfile = savein
    info(G)
    simulation = "delta \t N \t E  \t cc \t bc \t density \t pr \t clc \t hc \t katz \n"
    
    network = nx.DiGraph(G)
    N , E  , cc1  , bc1  , density1  ,  pr1 ,  clc1  , hc1  , katz1 = centralities(network)
    NN = N
    MM = E
    simulation = simulation  + "{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {}".format(-0.1, N/NN, E/MM, cc1, bc1, density1, pr1, clc1, hc1, katz1) + '\n'
    for i in range(0, 100 , 5):
        delta = i / 100
        d = compact_d(network, delta)
        r = 0
        while len(d)>0:
            network.remove_nodes_from(d)
            d = compact_d(network, delta)
            r = r + 1

        network.remove_nodes_from(d)   
        N , E  , cc  , bc  , density  ,  pr ,  clc  , hc  , katz = centralities(network)
        simulation = simulation +  "{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {}".format(delta, N/NN, E/MM, cc/cc1, bc/bc1, density/density1,  pr/pr1, clc/clc1, hc/hc1, katz/katz1) + '\n'
        print(delta)
    writetofile(xfile, simulation)

def centralities(network):
    N = len(network.nodes)
    E = len(network.edges)
    cc = nx.average_clustering(network)
    bc = av(nx.betweenness_centrality(network))
    density = nx.density(network)
    #print( av(nx.eigenvector_centrality(network.to_undirected())))
    pr = av(nx.pagerank(network))
    clc = av(nx.closeness_centrality(network))
    hc = av(nx.harmonic_centrality(network))
    katz = av(nx.katz_centrality(network))
    
    return N , E  , cc  , bc  , density  , pr ,  clc  , hc  , katz

def sampling(network, k):
    import random
    sampled_nodes = random.sample(network.nodes, k)
    sampled_graph = network.subgraph(sampled_nodes)
    return sampled_graph

def writetofile(filename, text):
    file = open("{}".format(filename), "w") 
    file.write(text) 
    file.close() 

def average(L):
    if len(L)>0:
        return sum(L)/len(L)
    else:
        return 0
def av(D):
    s = sum(D.values())
    return s/len(D)

def compact_d(network, delta):
    Nodes = []
    deleted = []
    for i in network.nodes:
        Nodes.append(i)
    #print("Network: ", Nodes)
    for i in range(len(Nodes)-1):
        if Nodes[i] not in deleted:
            for j in range(i+1, len(Nodes),1):
                if Nodes[j] not in deleted:
                    N_InA = list(network.predecessors(Nodes[i]))
                    N_OutA = list(network.successors(Nodes[i]))
                    N_InB = list(network.predecessors(Nodes[j]))
                    N_OutB = list(network.successors(Nodes[j]))
                    if jaccard(Nodes[i], Nodes[j], N_InA, N_OutA, N_InB, N_OutB)>= 1-delta:
                        deleted.append(Nodes[j])
    return deleted
    


    
def jaccard(a,b, Ain, Aout, Bin, Bout):
    """ Return similarity between A, B in directed network"""
    if a in Bin: Bin.remove(a)
    if a in Bout: Bout.remove(a)
    if b in Ain: Ain.remove(b)
    if b in Aout: Aout.remove(b)    
    lin1 = len(set(Ain).intersection(set(Bin)))
    lin2 = len(set(Ain).union(set(Bin)))
    lout1 = len(set(Aout).intersection(set(Bout)))
    lout2 = len(set(Aout).union(set(Bout)))
    in_result=0
    out_result=0
    total=0
    if lin2!=0: 
        in_result = lin1/lin2*1.0
    if lout2!=0:
        out_result = lout1/lout2*1.0        
    if (lin2 + lout2) !=0: 
        total = (lin1 + lout1)/(lin2 + lout2)*1.0
    return total


def simpson(a,b, Ain, Aout, Bin, Bout):
    """ Return similarity between A, B in directed network"""
    if a in Bin: Bin.remove(a)
    if a in Bout: Bout.remove(a)
    if b in Ain: Ain.remove(b)
    if b in Aout: Aout.remove(b)    
    lin1 = len(set(Ain).intersection(set(Bin)))
    lin2 = len(set(Ain).union(set(Bin)))
    lout1 = len(set(Aout).intersection(set(Bout)))
    lout2 = len(set(Aout).union(set(Bout)))
    in_result=0
    out_result=0
    total=0
    if lin2!=0: 
        in_result = lin1/lin2*1.0
    if lout2!=0:
        out_result = lout1/lout2*1.0        
    if (lin2 + lout2) !=0: 
        total = (lin1 + lout1)/(lin2 + lout2)*1.0
    return total    

def info(G):
    N , E  , cc  , bc  , density  ,  pr ,  clc  , hc  , katz = centralities(G)
    sim1 = "{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {}".format( N, E, cc, bc, density,  pr, clc, hc, katz)
    d = compact_d(G, 0)
    G.remove_nodes_from(d)
    N , E  , cc  , bc  , density  ,  pr ,  clc  , hc  , katz = centralities(G)
    sim2 = "{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {}".format( N, E, cc, bc, density,  pr, clc, hc, katz)
    return (sim1, sim2)

path = 'D:\Documents\Research Projects\Complex Networks Researches\Compaction Model\Intrinsic Model with approximating\Python Codes\Dataset\Static'
sample = path + '\my example network\sample.txt'

spath = "D:\Documents\Research Projects\Complex Networks Researches\Compaction Model\Intrinsic Model with approximating\Python Codes\\new Dataset"
# =============================================================================
# spath + "\WI8\interolog.txt",
#             spath + "\WI8\lit.txt",
#             spath + "\WI8\scaffold.txt",
#             spath + "\WI8\wi8.txt",
#             spath + "\WI8\wi2004.txt",
#             spath + "\WI8\wi2007.txt",
#            spath + "\Yeast Interactome\CCSB-Y2H.txt",
#            spath + "\Yeast Interactome\Collins.txt",
#            spath + "\Yeast Interactome\Ito_core.txt",
#            spath + "\Yeast Interactome\\Uetz_screen.txt",
#            spath + "\Yeast Interactome\Y2H_union.txt",
# =============================================================================
datasets = ["\\ba_1k_2k\\ba_1k_2k.txt",
            "\\ba_1k_40k\\ba_1k_40k.txt",
            "\er_graph_1k_4k\er_graph_1k_4k.txt",
            "\er_graph_1k_6k\er_graph_1k_6k.txt",
               
            '\Ecoli\Ecoli.txt',
            '\\rt_assad\\rt_assad.txt',
            "\\bio-CE-LC\\bio-CE-LC.txt",
            "\\bio-yeast\\bio-yeast.txt",
            "\ca-CSphd\ca-CSphd.mtx",
            "\ca-GrQc\ca-GrQc.mtx",
            "\mammalia-voles-kcs-trapping\mammalia-voles-kcs-trapping.txt",
            "\socfb-Reed98\socfb-Reed98.mtx",
            "\socfb-Simmons81\socfb-Simmons81.mtx",
            "\\bio-CE-PG\\bio-CE-PG.txt",
            "\socfb-Haverford76\socfb-Haverford76.mtx",
            "\\bio-CE-CX\\bio-CE-CX.txt"]

for i in datasets:
    G = nx.read_adjlist(spath + i, create_using = nx.DiGraph(), nodetype = int)
    print(i, info(G))





def intrinsic(network, delta):
    """ undirected, comprehensively"""
    import numpy as np
    deletedNodes=np.zeros((len(network), 1), dtype=bool)
    Nodes = []
    Neighbors = [] 
    DN = [] # DeletedNodes

    network.remove_edges_from(network.selfloop_edges())
    for i in network.nodes:
        #new line for giving affiliation
        Nodes.append(i)
        Neighbors.append(neighbor(i, network))

    checked = [[0 for x in range(len(network.nodes))] for y in range(len(network.nodes))]
    for i in Nodes:
        indexi = Nodes.index(i)
        if not deletedNodes[indexi]:
            s= Neighbors[indexi]
            L = len(s)
            if L > 1:
                #u->A,    v-> B
                for u in range(0,L-1,1):
                    for v in range (u+1, L,1):
                        indexA = Nodes.index(s[u])
                        indexB = Nodes.index(s[v])
                        if (deletedNodes[indexA]==0) and (deletedNodes[indexB]==0) and ((checked[indexA][indexB]==0) or checked[indexB][indexA]==0) :
                            #Checking the similarity between node u, v by comparing their neighbors
                            dsim = sim(s[u], s[v], Neighbors[indexA], Neighbors[indexB])
                            checked[indexA][indexB] = dsim
                            checked[indexB][indexA] = dsim                            
                            if dsim >= delta:
                                deletedNodes[indexA] = 1 #deletedNodes.append(s[u])
                                DN.append(s[u])
    return (DN)

def intrinsic2group(network, delta):
    """ undirected, comprehensively"""
    Nodes = []
    Neighbors = [] 
    deletedNodes = []
    groups = []

    network.remove_edges_from(network.selfloop_edges())
    for i in network.nodes:
        #new line for giving affiliation
        Nodes.append(i)
        Neighbors.append(neighbor(i, network))

    for i in Nodes:
        indexi = Nodes.index(i)
        if not deletedNodes[indexi]:
            s= Neighbors[indexi]
            L = len(s)
            if L > 1:
                #u->A,    v-> B
                for u in range(0,L-1,1):
                    for v in range (u+1, L,1):
                        indexA = Nodes.index(s[u])
                        indexB = Nodes.index(s[v])
                        if (deletedNodes[indexA]==0) and (deletedNodes[indexB]==0)  :
                            #Checking the similarity between node u, v by comparing their neighbors
                            dsim = sim(s[u], s[v], Neighbors[indexA], Neighbors[indexB])
                            if dsim >= delta:
                                groups.append((s[v],s[u]))
                                deletedNodes[indexA] = 1 #deletedNodes.append(s[u])
    return groups
        

def dintrinsic(network, delta):
    """ directed, comprehensively"""
    print("Our network of type ", type(network))
    import numpy as np
    n = len(network)
    similarity_matrix = np.zeros(n*n).reshape(n,n)
    print(similarity_matrix)
    deletedNodes=np.zeros((len(network), 1), dtype=bool)
    Nodes = []
    d=0
    DN = []
    Neighbors_In = []
    Neighbors_Out = []
    for i in network.nodes:
        Nodes.append(i)
        Neighbors_In.append(list(network.predecessors(i)))
        Neighbors_Out.append(list(network.successors(i)))
    checked = [[0 for x in range(len(network))] for y in range(len(network))]
    for i in Nodes:
        m=0
        d=0
        indexi = Nodes.index(i)
        if not deletedNodes[indexi]:
            s = Neighbors_In[indexi] + Neighbors_Out[indexi]
            L = len(s)
            if L > 0:
                for u in range(0,L-1,1):
                    for v in range (u+1, L,1):
                        m+=1
                        indexA = Nodes.index(s[u])
                        indexB = Nodes.index(s[v])
                        if (deletedNodes[indexA]==0) and (deletedNodes[indexB]==0) and ((checked[indexA][indexB]==0) or checked[indexB][indexA]==0) :
                            ds = dsim(s[u],s[v],Neighbors_In[indexA], Neighbors_Out[indexA], Neighbors_In[indexB], Neighbors_Out[indexB])
                            checked[indexA][indexB] = ds
                            checked[indexB][indexA] = ds
                            if ds >= delta :
                                print(indexA, indexB, ds)
                                DN.append(s[u])
                                deletedNodes[indexA] = 1 
                                d+=1
                                similarity_matrix[indexA][indexB] = ds
    print(similarity_matrix)
    return (DN)

    

def plot_degree_distribution(G):
    import matplotlib.pyplot as plt
    dd = {}
    dd = G.degree()
    plt.hist(list(dd.values()), histtype="step")
    plt.xlabel("Degree $K$")
    plt.ylabel("$P(K)$")
    plt.title("Degree distribution")
    plt.savefig("hist1.pdf")

def dsim(a,b, Ain, Aout, Bin, Bout):
    """ Return similarity between A, B in directed network"""
    if a in Bin: Bin.remove(a)
    if a in Bout: Bout.remove(a)
    if b in Ain: Ain.remove(b)
    if b in Aout: Aout.remove(b)
    
    lin1 = len(set(Ain).intersection(set(Bin)))
    lin2 = len(set(Ain).union(set(Bin)))
    
    lout1 = len(set(Aout).intersection(set(Bout)))
    lout2 = len(set(Aout).union(set(Bout)))
    
        # they do not share ins
    if lin1 == 0 and lout2 != 0: 
        result =  lout1/lout2*1.0
        # they do not share outs    
    if lout1 ==0 and lin2 != 0: 
        result =  lin1/lin2*1.0
        
    if lin1>0 and lout1>0 and lin2+lout2 >0:
        result =  lin1+lout1/(lin2+lout2)*1.0
    else:
        result = 0
    print(a,b, ":", Ain, Aout, Bin, Bout,":", result)
    return result

def percent(per, whole):
    return round((per * whole) / 100.0)

def percentage(old, new):
    if old != 0: 
        return 100-(new*100/old)
    else:
        return 0

def neighbor(n, Network):
    NN = []
    for nbr in Network[n]:
        if Network.edges(n,nbr):
            NN.append(nbr)
    return NN

def sim(a,b, A, B):
    """Return similarity between A, B 
    in undirected network
    return 1 if similar
    return 0 if not similar"""
    if a in B: B.remove(a)
    if b in A: A.remove(b)
    l1 = len(set(A).intersection(set(B)))
    l2 = len(set(A).union(set(B)))
    
    if l2 != 0 :
        return l1/(l2*1.0)
    else:
        return 0
    
def showDD(G):
    import collections
    import matplotlib.pyplot as plt
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')
    
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

# =============================================================================
# # pos = nx.spring_layout(G)
# # nx.draw_networkx(G, pos)
# # plt.show()
# # # 
# 
# 
# #ecc = nx.eccentricity(network)
# #diameter = nx.diameter(network)
# #robustness = len(max(nx.connected_component_subgraphs(network), key=len))/N
# """ directed, comprehensively
# b = nx.adjacency_matrix(network)
# matrix = b.todense()
# print(matrix)
# pos = nx.spring_layout(network)
# for _ in range(20):
#     nx.draw_networkx(network, pos)
# plt.show()"""
# 
# =============================================================================
