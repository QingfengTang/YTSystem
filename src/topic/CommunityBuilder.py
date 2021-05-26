import itertools,math
import igraph
from src.topic.comgroup import Community, CommunityGroup, Document
from src.topic import comdect


class CommunityBuilder:
    def __init__(self, pageloader, min_docs, logfile=None):
        self.pageloader = pageloader
        self.logfile = logfile
        self.min_docs = min_docs
        self._run_time = 0.0

    def build(self, max_depth=5, min_doc_num=15,min_nodes=20):
        (twn,docs) = self.load_title_document()
        if len(docs) < self.min_docs:
            return None
        return self.build_from_data(twn,docs,max_depth,min_doc_num,min_nodes)
  
    def build_from_data(self,twn,docs,max_depth, min_doc_num,min_nodes):
        cowordnet = self.build_global_cowordnet(docs)
        com_dect = comdect.LabelCommunityDetection(min_nodes)
        group = CommunityGroup(cowordnet,docs,com_dect)
        depth = 0
        rootcom = Community(twn,group,0)
        group.add_community(rootcom)
        while depth<=max_depth:
            print ('Iteration')
            acoms = group.active_coms()
            if not acoms: break
            for c in acoms:
                children = c.make_children()
                if children:
                    group.remove_community(c)
                    for ch in children:
                        group.add_community(ch)  
            acoms = group.active_coms()
            if not acoms: break
            uncdocs = group.unclassified_docs()
            Community.map_docs_to_coms(uncdocs, acoms)
            group.remove_null_community(min_doc_num)
            depth += 1
            for c in acoms:
                c.rebuild_wordnet()
        self.merge_communities(group, 0.5)
        return group
   
    def output_keywords(self,group):
        s = ''
        for com in group:
            s += str(com.get_doc_num())+ ": "+ ' '.join(com.top_keywords()) + '\r\n'
        return s

    def merge_communities(self, group, merge_freshold=0.5):
        n = len(group)
        dset = DisjoinSet(n)
        coms = list(iter(group))
        for i in range(0,n-1):
            for j in range(i+1,n):
                sim = Community.similarity(coms[i],coms[j])
                if sim > merge_freshold:
                    dset.union(i,j)
        
        clusters = dset.sets(min_size=2)
        for c in clusters:
            group.merge_communities([coms[i] for i in c])

    def load_title_document(self,min_coocur=2, min_weight=1e-3):
        docs = list()
        codict, dfdict = dict(),dict() # wordpair:co_num
        for r in self.pageloader.loadpage():
            pid, title_list, content_dict = r
            docs.append(Document(pid, content_dict,title_list))
            
            for wp in itertools.combinations(title_list,2):
                if wp[0] > wp[1]: wp = (wp[1], wp[0])
                codict[wp] = (codict[wp]+1) if wp in codict else 1
            for w in title_list:
                dfdict[w] = (dfdict[w]+1) if w in dfdict else 1    

        elist = list()
        for wp,co in codict.items():
            if co >= min_coocur:
                weight = co/math.sqrt(dfdict[wp[0]] * dfdict[wp[1]])
                if weight > min_weight:
                    elist.append((wp[0],wp[1],weight))
        title_wordnet = igraph.Graph.TupleList(edges=elist, weights=True) 
        return(title_wordnet, docs)


    def build_global_cowordnet(self, docs, min_coocur=3):
        dociter = Community.wordpair_weight(docs, min_coocur,0)
        def dict2list(docs):
            for w1,w2,co,weight in docs:
                yield (w1, w2, co)

        return igraph.Graph.TupleList(edges=dict2list(dociter), weights=True)
    

    def outputCommunities(self, com_group, filename):
        f = open(filename, 'w')
        f.write(str(com_group))
        f.close()

class DisjoinSet:
    def __init__(self,n):
        # self.parent = range(0,n)
        # 下一行原代码如上，find()函数报错了，可能是这里的原因，因此修改代码
        self.parent = list(range(0,n))
        
    def find(self, i):
        root = i
        elems = []
        if root != self.parent[root]:
            root = self.parent[root]
            elems.append(root)
        for e in elems:
            self.parent[e] = root
        return root

    def union(self, i,j):
        ri = self.find(i)
        rj = self.find(j)
        if ri != rj:
            self.parent[i] = rj

    def sets(self, min_size=1):
        clusters = [[] for i in range(0,len(self.parent))]
        for i in range(0,len(self.parent)):
            root = self.find(i)
            clusters[root].append(i)
        return [li for li in clusters if len(li)>=min_size]









