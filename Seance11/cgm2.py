import copy
import fractions
import itertools
import matplotlib.pyplot as plt
import networkx
import random 
from typing import *


class ProbabilitySpace:
    
    def __init__(self,nb_events=None,measure:Dict=None):
        assert(nb_events != None or measure != None)
        if measure == None: # uniform measure on events by default
            self.measure = {omega: fractions.Fraction(1,nb_events) for omega in range(nb_events)}
        else:
            self.measure = measure

    def __iter__(self):
        return self.measure.__iter__()
    

def name_function(name:str,function:Callable)->Callable:
    function = copy.deepcopy(function)
    function.__name__ = name
    function.__qualname__ = name
    return function
    
class RandomVariable:

    def __init__(self,Omega:ProbabilitySpace,name:str,function:Callable,codomain=None,debug=False):
        self.Omega = Omega # also the domain of function
        self.name = name
        if function.__qualname__ != name:
            self.function = name_function(name,function)
        else:
            self.function = function
        self.codomain = codomain
        if debug:
            for omega in self.Omega:
                assert( self(omega) in codomain)

    def __call__(self,omega,debug=False):
        if debug:
            assert(omega in self.Omega)
        return self.function(omega)
                

def generate_single_argument_random_function(domain:range,codomain:range)->Callable:
    data = { t : random.choice(codomain) for t in domain }
    return lambda omega, data=data: data[omega] 
    
def generate_random_function(domains:List[range],codomain:range)->Callable:
    # Warning: arguments are tuples even with a single argument
    data = { t : random.choice(codomain) for t in itertools.product(*domains) }
    return lambda omega, data=data: data[omega]

def generate_random_random_variable(Omega:ProbabilitySpace,codomain:range,name:str):
    return RandomVariable(Omega,name,generate_single_argument_random_function(Omega,codomain))
    
    
class RandomVariables:
    
    def __init__(self,Omega:ProbabilitySpace,domain:range,Xs:Tuple[RandomVariable]=None,\
                     Ys:Tuple[RandomVariable]=None,Zs:Tuple[RandomVariable]=None): 
        ''' 
            .Xs: (may be) conditionned variables
            .Ys: (may be) conditionning variables
            .Zs: (may be) hidden variables
            '''
        if type(Xs) == tuple:
            self.Xs = Xs
        elif Xs == None:
            self.Xs = tuple([])
        else:
            self.Xs = (Xs,)
        if type(Ys) == tuple:
            self.Ys = Ys
        elif Ys == None:
            self.Ys = tuple([])
        else:
            self.Ys = (Ys,)
        self.Zs = tuple([])
        self.Omega = Omega
        self.domain = domain

        
    def eval_at_event(self,omega):
        return tuple([ self.Xs[i](omega) for i in range(len(self.Xs)) ])
        
    def distribution(self):
        # Remark: it corresponds to conditionning by all free variables !
        #assert(len(self.Ys) == 0) # Make sens only if no conditionning variables
        d = {}
        for omega in self.Omega:
            Xomega = self.eval_at_event(omega)
            if Xomega not in d:
                d[Xomega] = 0
            d[Xomega] += self.Omega.measure[omega]
        return d
    
    def distributions(self)->Dict:
        # Return distributions indexed by evaluation of conditionning variables:
        # - probabilities of evaluation
        # - random variable of a part of the original probability space (with change of measure)
        Omega_partition = {}
        for omega in self.Omega:
            ys = tuple([ self.Ys[j](omega) for j in range(len(self.Ys))])
            if ys not in Omega_partition:
                Omega_partition[ys] = {}
            Omega_partition[ys][omega] = self.Omega.measure[omega]
        ys_probabilities = { ys: sum(p for p in Omega_partition[ys].values()) for ys in Omega_partition}
        ys_probabilities = { ys: ys_probabilities[ys] for ys in ys_probabilities if ys_probabilities[ys] > 0}
        Omega_partition = { ys: {omega : Omega_partition[ys][omega]/ys_probabilities[ys]\
                                 for omega in Omega_partition[ys]} for ys in ys_probabilities}
        ys_conditional_variables = {ys: RandomVariables(ProbabilitySpace(measure=Omega_partition[ys]),self.domain,Xs=self.Xs)\
                                   for ys in ys_probabilities}
        return (ys_conditional_variables,ys_probabilities)
       
    def move_by_name_from_to(self,names,from_Zs,to_Zs):
        for a_name in names:
            for Z in from_Zs:
                if Z.name == a_name:
                    from_Zs.remove(Z)
                    to_Zs.insert(0,Z)
        return (from_Zs,to_Zs)
        
    def condition(self,names_new_conditionning_variables:List[str]):
        (new_Xs,new_Ys) = self.move_by_name_from_to(names_new_conditionning_variables,list(self.Xs),list(self.Ys))
        self.Xs = tuple(new_Xs)
        self.Ys = tuple(new_Ys)
        
    def uncondition(self,names_new_conditionned_variables:List[str]):
        (new_Ys,new_Xs) = self.move_by_name_from_to(names_new_conditionned_variables,list(self.Ys),list(self.Xs))
        self.Xs = tuple(new_Xs)
        self.Ys = tuple(new_Ys)
        
    def hide(self,names_hidden_variables:List[str]):
        (new_Xs,new_Zs) = self.move_by_name_from_to(names_hidden_variables,list(self.Xs),list(self.Zs))
        self.Xs = tuple(new_Xs)
        self.Zs = tuple(new_Zs)
        
    def show(self,name_shown_variables:List[str]):
        (new_Zs,new_Xs) = self.move_by_name_from_to(name_shown_variables,list(self.Zs),list(self.Xs))
        self.Zs = tuple(new_Zs)
        self.Xs = tuple(new_Xs)
        
    def are_independant(self,name_Xs:List[str],name_Ys:List[str],print_counter_example=False,debug=False):
        if debug:
            print(f'all names {[X.name for X in list(self.Xs+self.Ys+self.Zs)]}')
            print(f' name_Xs = {name_Xs}, name_Ys = {name_Ys}')
        Xs = [ X for X in list(self.Xs+self.Ys+self.Zs) if X.name in name_Xs ]
        Ys = [ Y for Y in list(self.Xs+self.Ys+self.Zs) if Y.name in name_Ys ]
        if debug:
            print(f'all Xs {Xs}, all Ys {Ys}')
        assert(len(Xs) == len(name_Xs) and len(Ys) == len(name_Ys))
        marginal_X = RandomVariables(self.Omega,self.domain,tuple(Xs)).distribution()
        marginal_Y = RandomVariables(self.Omega,self.domain,tuple(Ys)).distribution()
        joint_XY = RandomVariables(self.Omega,self.domain,tuple(Xs+Ys)).distribution()
        for x in marginal_X:
            for y in marginal_Y:
                if debug:
                    print(f'joint_XY({x+y})={joint_XY[x+y]} =?'+\
                              f'(marginal_X[{x}]={marginal_X[x]})*(marginal_Y[{y}]={marginal_Y[y]})={marginal_X[x]*marginal_Y[y]}')
                if joint_XY[x+y] != marginal_X[x]*marginal_Y[y]:
                    if print_counter_example:
                        print(f'join({x+y}) = {joint_XY[x+y]},'+\
                        f'marginal_X[{x}]={marginal_X[x]}, marginal_Y[{y}]={marginal_Y[y]},'+\
                        f'marginal_X[{x}]*marginal_Y[{y}]={marginal_X[x]*marginal_Y[y]}')
                    return False
        return True
        
    
    def are_conditionaly_independant(self,name_Xs,name_Ys,name_Zs):
        Xs = [ X for X in list(self.Xs+self.Ys+self.Zs) if X.name in name_Xs ]
        Ys = [ Y for Y in list(self.Xs+self.Ys+self.Zs) if Y.name in name_Ys ]
        Zs = [ Z for Z in list(self.Xs+self.Ys+self.Zs) if Z.name in name_Zs ]
        assert(len(Xs) == len(name_Xs) and len(Ys) == len(name_Ys) and len(Zs) == len(name_Zs))
        XYZ = RandomVariables(self.Omega,self.domain,Xs=tuple(Xs+Ys),Ys=tuple(Zs))
        (zs_conditional_variables,zs_probabilities) = XYZ.distributions()
        for zs in zs_probabilities:
            are_Xs_Ys_independant_given_Zs_set_to_zs = \
                    zs_conditional_variables[zs].are_independant(name_Xs,name_Ys)
            if not are_Xs_Ys_independant_given_Zs_set_to_zs:
                return False
        return True
    
    def list_all_possible_conditional_independances(self):
        all_names = [X.name for X in self.Xs+self.Ys+self.Zs]
        assert(len(all_names)==len(set(all_names)))
        # Xs ind Ys | Zs where:
        # - Xs, Ys and Zs disjoint subsets
        # - Xs and Ys non-empty sets of variables while Zs possibly empty 
        triplets = []
        for triplet_size in range(2,len(all_names)+1):
            for Xs_size in range(1,triplet_size):
                for Ys_size in range(1,triplet_size-Xs_size+1):
                    Zs_size = triplet_size-Xs_size-Ys_size # not really used except for debugging
                    #print(f' Triplets where |Xs|={Xs_size}, |Ys|={Ys_size}, |Zs|={Zs_size}')
                    for triplet_variables in itertools.combinations(all_names,triplet_size):
                        for Xs_names in itertools.combinations(triplet_variables,Xs_size):
                            Ys_Zs_names = [ X for X in triplet_variables if X not in Xs_names]
                            for Ys_names in itertools.combinations(Ys_Zs_names,Ys_size):
                                Zs_names = tuple([ X for X in Ys_Zs_names if X not in Ys_names])
                                if Xs_names[0] <= Ys_names[0]:
                                    # We use symmetry of the relation with respect to the two first variables
                                    triplets += [(Xs_names,Ys_names,Zs_names)]
        return triplets
        
    def list_all_conditional_independances(self,debug=False):
        triplets = self.list_all_possible_conditional_independances()
        if debug:
            print(f'There are {len(triplets)} triplets')
        conditionally_independant_triplets = []
        for t in triplets:
            if self.are_conditionaly_independant(t[0],t[1],t[2]):
                conditionally_independant_triplets += [t]
                if debug:
                    print(t,True)
        return conditionally_independant_triplets


    
    def print_distribution(self):
        d = self.distribution()
        names = [self.Xs[i].name for i in range(len(self.Xs))]
        concatenated_names = ','.join(names)
        result = f'Distribution of ({concatenated_names})<:>\n'
        for xs in d:
            data = '('+','.join([ (f'{names[i]}={xs[i]}')for i in range(len(xs))])+')'
            result += f'P{data}={d[xs]}\n'
        result += '</:>'
        print(result)
        
    def print_distributions(self):
        (ys_conditional_variables,ys_probabilities) = self.distributions()
        print(f'ys conditional variables {list(ys_conditional_variables.keys())}')
        for ys in ys_conditional_variables:
            conditioning_evaluations = [f'{self.Ys[j].name}={ys[j]}' for j in range(len(self.Ys))]
            print('|('+','.join(conditioning_evaluations)+f') has probability {ys_probabilities[ys]}')
            ys_conditional_variables[ys].print_distribution()
            
    def print_variables(self):
        str_indices = ''
        str_variables = ''
        for i in range(len(self.Xs)):
            index_size = len(str(i))+1
            variable_size = len(self.Xs[i].name)
            if i < len(self.Xs)-1:
                variable_size += 1
            str_indices += f'{i}:'+max(variable_size-index_size,0)*' '
            str_variables += self.Xs[i].name+max(index_size-variable_size,0)*' '
            if i < len(self.Xs)-1:
                str_variables += ','
        str_indices += '|'
        str_variables += '|'
        for i in range(len(self.Ys)):
            index_size = len(str(len(self.Xs)+i))+1
            variable_size = len(self.Ys[i].name)
            if i < len(self.Ys)-1:
                variable_size += 1
            str_indices += f'{len(self.Xs)+i}:'+max(variable_size-index_size,0)*' '
            str_variables += self.Ys[i].name+max(index_size-variable_size,0)*' '
            if i < len(self.Ys)-1:
                str_variables += ','
        print(str_indices)
        print(str_variables)


def generate_random_joint_distribution(Omega:ProbabilitySpace,domain:range,nb_variables:int):
    Xs = tuple([ RandomVariable(Omega,domain,f'X{i}') for i in range(nb_variables)])
    return RandomVariables(Omega,domain,Xs)
        
def find_causes_for_joint_distribution(joint_distribution:RandomVariables): # unused ? (to be programmed)
    n = len(joint_distribution.Xs)
    joint_distribution.hide([X.name for X in joint_distribution.Xs])
    for i in range(n):
        print(i)
        joint_distribution.show([joint_distribution.Zs[0].name])
        joint_distribution.print_variables()
        joint_distribution.print_distributions()
        joint_distribution.condition([joint_distribution.Xs[0].name])
        

class CausalGraphModel:
    
    def __init__(self,causal_functions: Dict):
        self.causal_functions = causal_functions
        (self.observed_variables,self.unobserved_variables,self.dag) = self.list_variables_and_build_DAG_graph()
        self.variables = self.observed_variables+self.unobserved_variables # XXX unused
        self.sort_topologicaly()
    
    def sort_topologicaly(self):
        print(f'observed {self.observed_variables} unobserved {self.unobserved_variables}')
        self.variables = list(networkx.topological_sort(self.dag))
        print(f'topological sort {list(self.variables)}')
        self.observed_variables = [ v for v in self.variables if v in self.observed_variables]
        self.unobserved_variables = [ v for v in self.variables if v in self.unobserved_variables]

    def list_variables_and_build_DAG_graph(self):
        dag = networkx.DiGraph()
        observed_variables = list(self.causal_functions.keys())
        unobserved_variables = []
        for an_observed_variable in self.causal_functions:
            dag.add_node(an_observed_variable)
            for an_argument_variable in self.causal_functions[an_observed_variable]:
                dag.add_edge(an_argument_variable,an_observed_variable)
                if an_argument_variable not in observed_variables+unobserved_variables:
                    unobserved_variables += [an_argument_variable]
        assert(networkx.is_directed_acyclic_graph(dag))
        return (observed_variables,unobserved_variables,dag)

    def generate_random_causes(self,Omega:ProbabilitySpace,domain:range,debug=False)->Dict:#[name] = Callable
        self.sort_topologicaly()
        # assume that variables are topologicaly sorted
        random_causes = {}
        for U in self.unobserved_variables:
            random_causes[U] = generate_single_argument_random_function(Omega,domain)
        for O in self.observed_variables:
            random_causes[O] = generate_random_function(len(self.causal_functions[O])*[domain],domain)
        if debug:
            for X in random_causes:
                print(f' X {X} random_causes[X] {random_causes[X]}')
        return random_causes
    
    def generate_variables_model(self,Omega:ProbabilitySpace,domain:range,causes=None,debug=False)->RandomVariables:
        if causes == None: # default is random
            causes = self.generate_random_causes(Omega,domain,debug=debug)
        random_variables = {}
        for X in self.variables:
            if X in self.unobserved_variables:
                random_variables[X] = RandomVariable(Omega,X,causes[X])
            else: # X in self.observed_variables:
                X_data = {} 
                for omega in Omega:
                    parents = tuple([ random_variables[Y](omega) for Y in self.causal_functions[X]])
                    X_data[omega] = causes[X](parents)
                X_function = name_function(X,lambda  omega_tuple, X_data=X_data: X_data[omega_tuple])
                random_variables[X] = RandomVariable(Omega,X,X_function)
        return RandomVariables(Omega,domain,Xs=tuple([ random_variables[X] for X in random_variables]))
            

    def search_empirical_conditional_independances(self,Omega:ProbabilitySpace,domain:range,nb_search=1):
        conditional_independances = None
        for _ in range(nb_search):
            R = self.generate_variables_model(Omega,domain)
            R_conditional_independances = R.list_all_conditional_independances()
            if conditional_independances == None:
                conditional_independances = set(R_conditional_independances)
            else:
                conditional_independances = conditional_independances.intersection(set(R_conditional_independances))
        return conditional_independances



    def project_dag_to_semi_markovian_model(self):
        # According to Tian (PhD Thesis, Studies in causal reasoning and learning 2002, page 60, quoting [Verma 93])
        # From a dag G over V union U build "DAG" PJ(G,V) on V with bidirected edges. 
        projected_dag = networkx.DiGraph()
        # 1. Add each variable in V as node of PJ(G,V)
        for v in self.observed_variables:
            projected_dag.add_node(v)
        # 2. For each pair of variables X,Y in V, if there is an edge between them in G, add the edge to PJ(G,V)
        for e in self.dag.edges():
            (X,Y) = e
            if X in self.observed_variables and Y in self.observed_variables:
                projected_dag.add_edge(X,Y)
        # 3. For each pair of variables X,Y in V, if there is a directed path from X to Y in G such that every internal node of the path is in U,
        #  then add edge X to Y if not present.
        for X in self.observed_variables:
            for Y in self.observed_variables:
                if (X,Y) not in self.dag.edges():
                    subgraph = networkx.subgraph(self.dag,[X,Y]+self.unobserved_variables)
                    if Y in networkx.descendants(subgraph,X):
                        projected_dag.add_edge(X,Y)
        # 4. For each pair of variables X,Y in V, if there is a divergent path between X and Y in G, such that every internal node on the path
        #   is in U (X <== Ui ==> y), add a bidirected edge X <===> Y to PJ(G).
        # divergent path in [Verma 93, page 10] X <== <== <== Z ===> ===> ==> Y
        projected_unobserved_variables = []
        for i in range(len(self.observed_variables)):
            X = self.observed_variables[i]
            for j in range(i+1,len(self.observed_variables)):
                Y = self.observed_variables[j]
                subgraph = networkx.subgraph(self.dag,[X,Y]+self.unobserved_variables)
                for Z in self.unobserved_variables:
                    if X in networkx.descendants(subgraph,Z) and Y in networkx.descendants(subgraph,Z):
                        uXY = f'u:{X}.{Y}'
                        projected_unobserved_variables += [uXY]
                        projected_dag.add_edge(uXY,X)
                        projected_dag.add_edge(uXY,Y)
                        break
        # Done
        projected_model = { X:tuple(list(projected_dag.predecessors(X))) for X in self.observed_variables}
        return projected_model
    
    def draw(self):
        layout = networkx.spring_layout(self.dag)
        networkx.draw(self.dag,pos=layout)
        networkx.draw_networkx_nodes(self.dag,pos=layout,nodelist=self.unobserved_variables,node_color='red')
        networkx.draw_networkx_nodes(self.dag,pos=layout,nodelist=self.observed_variables,node_color='green')
        networkx.draw_networkx_labels(self.dag,pos=layout)
        plt.show()

##### TESTS .......####

