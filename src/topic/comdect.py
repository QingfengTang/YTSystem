from igraph import Graph

class CommunityDetection:
    def __init__(self, min_nodes=10):
        self.min_nodes = min_nodes

    def detect(self, graph):
        raise NotImplementedError

    def comids_to_names(self, graph, coms):
        com_names = []
        for com in coms:
            nameset = frozenset(graph.vs.select(com)['names'])
            com_names.append(nameset)
        return com_names

class LabelCommunityDetection(CommunityDetection):
    def detect(self, graph):
        h = graph.community_fastgreedy(weights='weight')
        vc = h.as_clustering()
        coms = [c for c in vc if len(c)>=self.min_nodes]
        return coms
#     def detect(self,graph):
#         vc = graph.community_label_propagation(weights='weight')
#         coms = [c for c in vc if len(c)>=self.min_nodes]
#         return coms


def get_comdect(algor='label', min_nodes=20):
    if algor == 'label':
        return LabelCommunityDetection(min_nodes)
    else: 
        return NotImplementedError('unknown community detection algorithm')



