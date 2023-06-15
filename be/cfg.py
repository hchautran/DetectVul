import ast
import astor
import copy
import graphviz as gv
import utils
class Block(object):
  

    __slots__ = ["id", "statements", "func_calls", "predecessors", "exits"]

    def __init__(self, id):
        self.id = id
        self.statements = []
        self.func_calls = []
        self.predecessors = []
        self.exits = []

    def __str__(self):
        if self.statements:
            return "block:{}@{}".format(self.id, self.at())
        return "empty block:{}".format(self.id)

    def __repr__(self):
        txt = "{} with {} exits".format(str(self), len(self.exits))
        if self.statements:
            txt += ", body=["
            txt += ", ".join([ast.dump(node) for node in self.statements])
            txt += "]"
        return txt

    def at(self):
        """
        Get the line number of the first statement of the block in the program.
        """
        if self.statements and self.statements[0].lineno >= 0:
            return self.statements[0].lineno
        return None

    def is_empty(self):
        """
        Check if the block is empty.

        Returns:
            A boolean indicating if the block is empty (True) or not (False).
        """
        return len(self.statements) == 0

    def get_source(self):
        src = ""
        for statement in self.statements:
            if type(statement) in [ast.If, ast.For, ast.While]:
                src += (astor.to_source(statement)).split('\n')[0] + "\n" 

            elif type(statement) == ast.FunctionDef or\
                 type(statement) == ast.AsyncFunctionDef:
                src += (astor.to_source(statement)).split('\n')[0] + "...\n"
            else:
                src += astor.to_source(statement) 
        return src
    
    def get_normalized_source(self, name_dict):
        src = ""
        for statement in self.statements:
            statement = copy.deepcopy(statement) 
            statement = utils.normalize_user_defined_VAR(name_dict=name_dict).visit(statement) 
            statement = utils.normalize_user_defined_FUNC(name_dict=name_dict).visit(statement)  
            if type(statement) in [ast.If, ast.For, ast.While]:
                src += (astor.to_source(statement)).split('\n')[0] + "\n" 

            elif type(statement) == ast.FunctionDef or\
                 type(statement) == ast.AsyncFunctionDef:
                src += (astor.to_source(statement)).split('\n')[0] + "...\n"
            else:
                src += astor.to_source(statement) 
        return src
        

    def get_lineno(self):
        lines = []
        for statement in self.statements:
            if not isinstance(statement, (
                ast.FunctionDef, 
                ast.AsyncFunctionDef, 
                ast.Module, 
                ast.ClassDef, 
                ast.If,
                ast.Try,
                ast.For,
            )):
                for lineno in range(statement.lineno, statement.end_lineno + 1):
                    lines.append(lineno)
            else:
                lines.append(statement.lineno)
        return lines
            
    def get_calls(self):
        txt = ""
        for func_name in self.func_calls:
            txt += func_name + '\n'
        return txt


class Link(object):
    __slots__ = ["source", "target", "exitcase"]

    def __init__(self, source, target, exitcase=None):
        assert type(source) == Block, "Source of a link must be a block"
        assert type(target) == Block, "Target of a link must be a block"
        # Block from which the control flow jump was made.
        self.source = source
        # Target block of the control flow jump.
        self.target = target
        # 'Case' leading to a control flow jump through this link.
        self.exitcase = exitcase

    def __str__(self):
        return "link from {} to {}".format(str(self.source), str(self.target))

    def __repr__(self):
        if self.exitcase is not None:
            return "{}, with exitcase {}".format(str(self),
                                                 ast.dump(self.exitcase))
        return str(self)

    def get_exitcase(self):
        if self.exitcase:
            return astor.to_source(self.exitcase)
        return ""


class CFG(object):

    def __init__(self, name, asynchr=False):
        assert type(name) == str, "Name of a CFG must be a string"
        assert type(asynchr) == bool, "Async must be a boolean value"
        self.name = name
        self.asynchr = asynchr
        self.entryblock = None
        self.finalblocks = []
        self.functioncfgs = {} 
        self.name_dict = None

    def __str__(self):
        return "CFG for {}".format(self.name)

    def _visit_blocks(self, graph, block, visited=[], calls=True):
        if block.id in visited:
            return

        nodelabel = block.get_source()
        if self.name_dict is not None:
            nodelabel = block.get_normalized_source(self.name_dict)


        graph.node(str(block.id), label=nodelabel)
        visited.append(block.id)

        if calls and block.func_calls:
            calls_node = str(block.id)+"_calls"
            calls_label = block.get_calls().strip()
            graph.node(calls_node, label=calls_label,
                       _attributes={'shape': 'box'})
            graph.edge(str(block.id), calls_node, label="calls",
                       _attributes={'style': 'dashed'})

        for exit in block.exits:
            self._visit_blocks(graph, exit.target, visited, calls=calls)
            edgelabel = exit.get_exitcase().strip()
            graph.edge(str(block.id), str(exit.target.id), label=edgelabel)

    def _build_visual(self, format='pdf', calls=True):
        graph = gv.Digraph(name='cluster'+self.name, format=format,
                           graph_attr={'label': self.name})
        self._visit_blocks(graph, self.entryblock, visited=[], calls=calls)

        for subcfg in self.functioncfgs:
            subgraph = self.functioncfgs[subcfg]._build_visual(format=format,
                                                               calls=calls)
            graph.subgraph(subgraph)

        return graph

    def build_visual(self, filepath, format, calls=False, show=False):
        graph = self._build_visual(format, calls)
        graph.render(filepath, view=show)

    def __iter__(self):
        visited = set()
        to_visit = [self.entryblock]
        while to_visit:
            block = to_visit.pop(0)
            visited.add(block)
            for exit_ in block.exits:
                if exit_.target in visited or exit_.target in to_visit:
                    continue
                to_visit.append(exit_.target)
            yield block

        # for subcfg in self.functioncfgs.values():
        #     yield from subcfg