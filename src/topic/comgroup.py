import math,itertools
import heapq
from igraph import Graph


class Community:
    def __init__(self, wordnet, comgroup, doc_num=0):
        self._wordnet = wordnet
        self.group = comgroup
        self.comid = self.group.make_comid()
        self.active = True
        self.doc_num = doc_num
        self.newdoc_num = 0
        self.evaluate_word_weight()

    def evaluate_word_weight(self, method='strength'):
        if method == 'pagerank': # fast
            self._wordnet.vs['weight'] = self._wordnet.pagerank(directed=False,weights='weight')
        elif method == 'strength': #very fast and accurate
            vs = range(0,self._wordnet.vcount())
            self._wordnet.vs['weight'] = self._wordnet.strength(vs,weights='weight')
            total = sum(self._wordnet.vs['weight'])
            for v in self._wordnet.vs: v['weight'] /= total
        elif method =='eigenvector': #slower
            self._wordnet.vs['weight'] = self._wordnet.eigenvector_centrality(directed=False,weights='weight', scale=False)
            total = sum(self._wordnet.vs['weight'])
            for v in self._wordnet.vs: v['weight'] /= total
        elif method == 'betweenness': # too slow
            self._wordnet.vs['weight'] = self._wordnet.betweenness(directed=False,weights='weight')
            total = sum(self._wordnet.vs['weight'])
            for v in self._wordnet.vs: v['weight'] /= total
        elif method == 'authority': # as engenvector
            self._wordnet.vs['weight'] = self._wordnet.authority_score(weights='weight',scale=False)
            total = sum(self._wordnet.vs['weight'])
            for v in self._wordnet.vs: v['weight'] /= total

        #topn = heapq.nlargest(100,self._wordnet.vs, key=lambda v:v['weight'])
        self.name_vertex_map =  {v['name']:v for v in self._wordnet.vs}
        #{v['name']:v  for v in self._wordnet.vs}

    def is_active(self):
        return self.active

    def get_doc_num(self):
        return self.doc_num

    def make_children(self):
        wid_comus = self.group.community_detection.detect(self._wordnet)
        if len(wid_comus)<=1:
            self.active=False
            return []
        children = []
        for i in range(0, len(wid_comus)):
            subg = self._wordnet.subgraph(wid_comus[i])
            subcom = Community(subg, self.group)
            children.append(subcom)
        self.map_docs_to_coms(self.iterdoc(),children)
        return children

    def top_keywords(self, n=50):
        topn = heapq.nlargest(n,self._wordnet.vs, key=lambda v:v['weight'])
        return ((v['name'],v['weight']) for v in topn) 

    def get_community_name(self, num=4):
        """ extract community name from title """
        worddf = dict()
        for doc in self.iterdoc():
            for w in doc.get_title_keywords():
                worddf[w] = worddf[w]+1 if w in worddf else 1
        
        return heapq.nlargest(num, worddf.iterkeys(), key=lambda k:worddf[k])
    
    def top_keywords_name(self, n=15):
        topn = heapq.nlargest(n,self._wordnet.vs, key=lambda v:v['weight'])
        return (v['name'] for v in topn)     
    
    def community_edges(self):
        edges = {}
        for e in self._wordnet.es:
            edges[(self._wordnet.vs[e.source]['name'], self._wordnet.vs[e.target]['name'])] = e['weight']
        return edges
     
    def generate_description(self, n=50,max=20):
        des = []
#         db_con = odbc.odbc(config.db_connect_str)
#         cur = db_con.cursor()
        topn = self.top_keywords_name(n)
        for name in topn:
#             print('Community generate_description name:',name)
#             cur.execute('select TAG,word from T_WORD where Word_id = ?', (name,))
            r = ['/nr',name]
            try:
#                 r = cur.fetchone()
                tag =r[0]
                wordname = r[1]
            except:
                continue
            if(len(des)<max):
                if tag=='/ns':
                    des.append(wordname)
                elif tag=='/nr':
                    des.append(wordname)
                elif tag=='/nt':
                    des.append(wordname)
                elif tag=='/v':
                    des.append(wordname)
                elif tag=='/vn':
                    des.append(wordname)
                elif tag=='/n':
                    des.append(wordname)
                
        return ' '.join(des)
   
    def add_doc(self, doc):
        doc.add_community(self.comid)
        self.doc_num += 1
        self.newdoc_num += 1

    def iterdoc(self):
        return self.group.iterdoc(self.comid)

    def __unicode__(self):
        s = str(self.comid) + u':'  + u' '.join(self._wordnet.vs['name'])
        return s

    def get_comid(self):
        return self.comid

    @classmethod
    def correlative_commuities(cls, doc, children, multi = False, min_correl = 0.1):
        correls = []
        m,m_child = min_correl, None
        for child in children:
            v = child.correlativeValue(doc)
            correls.append(v)
            if v>m: m, m_child=v, child 

        if m_child == None: return [] 
        if not multi:
            return [m_child, ] 
        else:
            # normalization
            s = sum(correls)
            correls = [v/s for v in correls]
            m = max(correls)
            ret_coms = []
            for i, v in enumerate(correls):
                if (m-v)/m <= 0.2:
                    ret_coms.append(children[i])
 
            return ret_coms 


    def correlativeValue(self, doc, only_sim=True):
        sim,asso = 0.0,0.0
        wn = self._wordnet
        com_words = set(iter(doc)) & set(iter(self.name_vertex_map.keys()))
        for w in com_words:
            vex = self.name_vertex_map[w]
            sim += float(vex['weight']) * float(doc[w])
            if not only_sim:
                for v2 in vex.neighbors():
                    asso += vex['weight'] * v2['weight'] * wn[vex,v2]
        if sim == 0: return asso
        x,y=0,0
        for v in iter(self.name_vertex_map.values()):
            x += v['weight'] * v['weight']
        for w,wt in doc.iter_word_weight():
            y += float(wt)*float(wt)
        sim =sim/(math.sqrt(x)*math.sqrt(y))
        return sim+asso

    def rebuild_wordnet(self, min_coocur=3, min_weight=1e-3):
        if self.doc_num == 0 or self.newdoc_num/float(self.doc_num)<.2:
            return

        print ('rebuilding wordnet of Community %d' % self.comid)
        gwn = self.group.cowordnet
        
        #words = set()
        edges = {}
        #for e in self._wordnet.es:
            #edges[(self._wordnet.vs[e.source]['name'], self._wordnet.vs[e.target]['name'])] = e['weight']
        for w1,w2,co,wg in self.wordpair_weight(self.iterdoc(), min_coocur, min_weight):
            #words.add(w1); words.add(w2)
            weight = 0
            try:
                weight = ( self._wordnet[w1,w2] + wg) * co / gwn[w1,w2]
            except ValueError:
                weight = wg * co / gwn[w1,w2]
            if weight > min_weight:
                edges[(w1,w2)] = weight

        self._wordnet = Graph.TupleList(edges=self.__dict_to_name(edges), weights=True)
        self.evaluate_word_weight()
        
        self.newdoc_num = 0

    def __dict_to_name(self,wp_dict):
        for wp,w in iter(wp_dict.items()):
            yield (wp[0],wp[1],w)

    @classmethod
    def similarity(cls, com1, com2):
        sim = 0.0
        #name_weight1 = [(n,v['weight']) for n,v in com1.name_vertex_map.iteritems()]
        #name_weight2 = [(n,v['weight']) for n,v in com2.name_vertex_map.iteritems()]
        #name_weight1.sort(key=lambda x:x[1],reverse=True)
        #name_weight2.sort(key=lambda y:y[1],reverse=True)
        #name_weight1 = dict(name_weight1)
        #name_weight2 = dict(name_weight2)
        
        com_names = set(iter(com1.name_vertex_map.keys())) & set(iter(com2.name_vertex_map.keys()))
        sim = 0.0
        for n in com_names:
            sim += com1.name_vertex_map[n]['weight'] * com2.name_vertex_map[n]['weight']
        x,y = 0,0 
        if sim == 0: return 0
        for v in iter(com1.name_vertex_map.values()):
            w = v['weight']
            x += w*w
        for v in iter(com2.name_vertex_map.values()):
            w = v['weight']
            y += w*w
        return sim/math.sqrt(x)/math.sqrt(y)


    @classmethod
    def map_docs_to_coms(cls, docs, coms):
        """ map documents to communities """
        if len(coms)<=1: return
        for doc in docs:
            #if parent_comid>=0:
            doc.remove_community() # document remove from current community
            corrChildren = cls.correlative_commuities(doc, coms)
            for child in corrChildren:
                child.add_doc(doc)

    @classmethod
    def wordpair_weight(cls, docs, min_coocur=3, min_weight=1e-3):
        codict, dfdict = dict(),dict() # wordpair:co_num
        for doc in docs:
            for wp in doc.iter_wordpair():
                codict[wp] = (codict[wp]+1) if wp in codict else 1
            for w in doc:
                dfdict[w] = (dfdict[w]+1) if w in dfdict else 1

        for wp,co in iter(codict.items()):
            if co >= min_coocur:
                weight = co/math.sqrt(dfdict[wp[0]]*dfdict[wp[1]])
                if weight > min_weight:
                    yield (wp[0],wp[1],co,weight)
    # ??????????????????????????????
    def __str__(self):
        return 'self._wordnet:'+ str(self._wordnet) + '\n' +\
            'self.group:'+str(self.group) + '\n' + \
            'self.comid:'+str(self.comid) + '\n' +\
            'self.active:'+str(self.active) + '\n' +\
            'self.doc_num:'+str(self.doc_num) + '\n' +\
            'self.newdoc_num:'+str(self.newdoc_num) + '\n'



class CommunityGroup:
    """Semantic Communty Group"""
    def __init__(self, cowordnet, docs, comdect):
        self.cowordnet = cowordnet
        self.docs = docs
        self.community_detection = comdect
        self.communities = []
        self.g_comid = 0

    def total_doc_size(self):
        return len(self.docs)

    def iterdoc(self,comid):
        for doc in self.docs:
            if doc.belongto_commmunity(comid):
                yield doc

    def make_comid(self):
        self.g_comid+=1
        return self.g_comid

    def unclassified_docs(self):
        for doc in self.docs:
            if not doc.is_classifed():
                yield doc
          
    def add_community(self, com):
        if isinstance(com,Community):
            self.communities.append(com)
        else:
            raise TypeError

    def remove_community(self,com):
        if com in self.communities:
            self.communities.remove(com)
            for doc in self.iterdoc(com.get_comid()):
                doc.remove_community()
        else:
            raise Exception

    def merge_communities(self, coms):
        assert(len(coms)>1)
        print ('merging comminities: ' + ' '.join([str(c.get_comid()) for c in coms]))
        maxcom = coms[0]
        for com in coms:
            if com.get_doc_num()>maxcom.get_doc_num():
                maxcom = com

        for com in coms:
            if com.get_comid() != maxcom.get_comid():
                for doc in self.iterdoc(com.get_comid()):
                    #doc.remove_community(com.get_comid())
                    maxcom.add_doc(doc)
                self.remove_community(com)
    
    def remove_null_community(self,min_docnum=15):
        for c in list(self.communities):
            if c.doc_num < min_docnum:
                self.remove_community(c)

    def active_coms(self):
        ac = []
        for com in self.communities:
            if com.is_active():
                ac.append(com)
        return ac

    def iter_community_docnum(self):
        for c in self.communities:
            yield c.doc_num

    def __iter__(self):
        return iter(self.communities)

    def __len__(self):
        return len(self.communities)

    def doc_labels(self):
        return {d.get_docid():d.get_comid() for d in self.docs}

    # ??????????????????????????????
    def __str__(self):
        return 'CommunityGroup:' + \
            ' '.join(str(d.get_comid()) for d in self.docs) +'\n' +\
            '\tself.cowordnet' + str(self.cowordnet)+'\n' +\
            '\tself.docs' + str(self.docs)+'\n' +\
            '\tself.community_detection' + str(self.community_detection)+'\n' +\
            '\tself.communities' + str(self.communities)+'\n' +\
            '\tself.g_comid' + str(self.g_comid) + '\n***********************************************************************************************\n'
    
            
            

class Document:
    """ represent a document """
    def __init__(self, docid, word_dict, title_keywords_list):
        self.docid = docid
        self.comids = 0
        self.word_dict = word_dict
        self.wordpairs = list()
        self.title_keywords = title_keywords_list

    def get_docid(self):
        return self.docid
    
    def get_comid(self):
        return self.comids

    def iter_wordpair(self):
        if not self.wordpairs:
            for wp in itertools.combinations(iter(self.word_dict.keys()),2):
                if wp[0]>wp[1]: wp = (wp[1],wp[0])
                self.wordpairs.append(wp)
        return self.wordpairs
    
    def get_title_keywords(self):
        return self.title_keywords

    def belongto_commmunity(self, comid):
        return comid == self.comids
    def is_classifed(self):
        return self.comids != 0
    def add_community(self, comid):
        self.comids = comid
    def remove_community(self):
        self.comids = 0

    def __iter__(self):
        return iter(self.word_dict.keys())

    def iter_word_weight(self):
        return iter(self.word_dict.items())

    def __getitem__(self, word):
        return self.word_dict[word]

    def __hash__(self):
        return self.docid
    
    def __eq__(self, other):
        return self.docid == other.docid

    def __cmp__(self, other):
        return self.docid - other.docid
