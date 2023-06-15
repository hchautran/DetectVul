import ast
from cfg import Block, Link, CFG
import sys


def is_py38_or_higher():
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        return True
    return False


NAMECONSTANT_TYPE = ast.Constant if is_py38_or_higher() else ast.NameConstant

def find_class_from_line(src, lineno):
    candidate = None
    for item in ast.walk(ast.parse(src)):
        if isinstance(item, (ast.ClassDef)):
            if item.lineno > lineno:
                continue
            if candidate:
                distance = lineno - item.lineno
                if distance < (lineno - candidate.lineno):
                    candidate = item
            else:
                candidate = item

    if candidate:
        return candidate.name
    else:
        return ''

def find_func_from_line(src, lineno):
    candidate = None
    for item in ast.walk(ast.parse(src)):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if item.lineno > lineno:
                continue
            if candidate:
                distance = lineno - item.lineno
                if distance < (lineno - candidate.lineno):
                    candidate = item
            else:
                candidate = item

    if candidate:
        return candidate.name
    else:
        return 'global'

def invert(node):
    inverse = {ast.Eq: ast.NotEq,
               ast.NotEq: ast.Eq,
               ast.Lt: ast.GtE,
               ast.LtE: ast.Gt,
               ast.Gt: ast.LtE,
               ast.GtE: ast.Lt,
               ast.Is: ast.IsNot,
               ast.IsNot: ast.Is,
               ast.In: ast.NotIn,
               ast.NotIn: ast.In}

    if type(node) == ast.Compare:
        op = type(node.ops[0])
        inverse_node = ast.Compare(left=node.left, ops=[inverse[op]()],
                                   comparators=node.comparators)
    elif isinstance(node, ast.BinOp) and type(node.op) in inverse:
        op = type(node.op)
        inverse_node = ast.BinOp(node.left, inverse[op](), node.right)
    elif type(node) == NAMECONSTANT_TYPE and node.value in [True, False]:
        inverse_node = NAMECONSTANT_TYPE(value=not node.value)
    else:
        inverse_node = ast.UnaryOp(op=ast.Not(), operand=node)

    return inverse_node


def merge_exitcases(exit1, exit2):
    if exit1:
        if exit2:
            return ast.BoolOp(ast.And(), values=[exit1, exit2])
        return exit1
    return exit2


class CFGBuilder(ast.NodeVisitor):

    def __init__(self, src, separate=True, max_statement_len=5):
        super().__init__()
        self.after_loop_block_stack = []
        self.curr_loop_guard_stack = []
        self.current_block = None
        self.separate_node_blocks = separate
        self.max_statement_len = max_statement_len
        self.current_class = '' 
        self.src = src

    def build(self, name, tree, asynchr=False, entry_id=0):
        self.cfg = CFG(name, asynchr=asynchr)
        self.current_id = entry_id
        self.current_block = self.new_block()
        self.cfg.entryblock = self.current_block
        self.visit(tree)
        self.clean_cfg(self.cfg.entryblock)
        return self.cfg

    def build_from_src(self, name, src):
        tree = ast.parse(src)
        return self.build(name, tree)

    def build_from_file(self, name, filepath):
        with open(filepath, 'r') as src_file:
            src = src_file.read()
            return self.build_from_src(name, src)

    def new_block(self):
        self.current_id += 1
        return Block(self.current_id)

    def add_statement(self, block, statement):
        block.statements.append(statement)

    def add_exit(self, block, nextblock, exitcase=None):
        newlink = Link(block, nextblock, exitcase)
        block.exits.append(newlink)
        nextblock.predecessors.append(newlink)

    def new_loopguard(self):
        if (self.current_block.is_empty() and
                len(self.current_block.exits) == 0):
            loopguard = self.current_block
        else:
            loopguard = self.new_block()
            self.add_exit(self.current_block, loopguard)
        return loopguard

    def new_functionCFG(self, node, asynchr=False):
        func_body = ast.Module(body=node.body)

        func_builder = CFGBuilder(
            src=self.src,
            separate=self.separate_node_blocks, 
            max_statement_len=self.max_statement_len
        )

        if self.current_class != '':
            name = self.current_class + '.' + node.name
        elif isinstance(node, (ast.Try, ast.ExceptHandler)):
            print(node.body)
            name = 'try block'
        else: 
            name = node.name

        def_block = self.new_block()
        self.add_statement(def_block, node) 
        
        self.cfg.functioncfgs[name] = func_builder.build(
            name,
            func_body,
            asynchr,
            self.current_id 
        )

        self.add_exit(def_block, self.cfg.functioncfgs[name].entryblock)
        self.cfg.functioncfgs[name].entryblock = def_block

        self.current_id = func_builder.current_id + 1

    def clean_cfg(self, block, visited=[]):
        if block in visited:
            return
        visited.append(block)

        if block.is_empty():
            for pred in block.predecessors:
                for exit in block.exits:
                    self.add_exit(pred.source, exit.target,
                                  merge_exitcases(pred.exitcase,
                                                  exit.exitcase))
                    if exit in exit.target.predecessors:
                        exit.target.predecessors.remove(exit)
                if pred in pred.source.exits:
                    pred.source.exits.remove(pred)

            block.predecessors = []
            for exit in block.exits[:]:
                self.clean_cfg(exit.target, visited)
            block.exits = []
        else:
            for exit in block.exits[:]:
                self.clean_cfg(exit.target, visited)



    def goto_new_block(self, node):
        if self.separate_node_blocks and self.max_statement_len <= len(self.current_block.statements):
            newblock = self.new_block()
            self.add_exit(self.current_block, newblock)
            self.current_block = newblock
        self.generic_visit(node)
    
    def visit_Import(self, node):
        self.add_statement(self.current_block, node)
        self.goto_new_block(node)

    def visit_ImportFrom(self, node):
        self.add_statement(self.current_block, node)
        self.goto_new_block(node)

    def visit_Expr(self, node):
        self.add_statement(self.current_block, node)
        self.goto_new_block(node)

    def visit_Call(self, node):
        def visit_func(node):
            if type(node) == ast.Name:
                return node.id
            elif type(node) == ast.Attribute:
                # Recursion on series of calls to attributes.
                func_name = visit_func(node.value)
                func_name += ("." + node.attr)
                return func_name
            elif type(node) == ast.Str:
                return node.s
            elif type(node) == ast.Subscript:
                node = node.value
                if type(node) == ast.Name:
                    return node.id
                elif type(node) == ast.Attribute: 
                    func_name = visit_func(node)
                    func_name += "." + node.attr
                    return func_name
                else:
                    return type(node).__name__
            else:
                return type(node).__name__

        func = node.func
        func_name = visit_func(func)
        self.current_block.func_calls.append(func_name)

    def visit_Assign(self, node):
        self.add_statement(self.current_block, node)
        self.goto_new_block(node)

    def visit_AnnAssign(self, node):
        self.add_statement(self.current_block, node)
        self.goto_new_block(node)

    def visit_AugAssign(self, node):
        self.add_statement(self.current_block, node)
        self.goto_new_block(node)

    def visit_Try(self, node):
        # self.add_statement(self.current_block, node.body)
       
        # New block for the case in which the assertion 'fails'.
        try_block = self.new_block()
        handler_block = self.new_block()
        self.add_exit(self.current_block, try_block)
        self.add_exit(self.current_block, handler_block)

        after_try_block = self.new_block()

        if len(node.orelse) != 0 or len(node.finalbody) != 0:
            else_block = self.new_block()
            final_block = self.new_block()
            self.add_exit(self.current_block, else_block)
            if len(node.finalbody) != 0:
                self.add_exit(self.current_block, final_block)

            self.current_block = else_block
            for child in node.orelse:
                self.visit(child)
            if not self.current_block.exits:
                self.add_exit(self.current_block, after_try_block)

            if len(node.finalbody) != 0:
                self.current_block = final_block
                for child in node.finalbody:
                    self.visit(child)
                if not self.current_block.exits:
                    self.add_exit(self.current_block, after_try_block)
        else:
            self.add_exit(self.current_block, after_try_block)
            
        self.current_block = handler_block
        for handler in node.handlers:
            handler_block = self.new_block()
            for child in handler.body:
                self.visit(child)
            if not self.current_block.exits:
                self.add_exit(self.current_block, after_try_block)

        self.current_block = try_block
        for child in node.body:
            self.visit(child)
        if not self.current_block.exits:
            self.add_exit(self.current_block, after_try_block)

        self.current_block = after_try_block
    
    def visit_ExceptHandler(self, node ):
        self.add_statement(self.current_block, node)
        self.goto_new_block(node)

    def visit_Raise(self, node):
        pass

    def visit_Assert(self, node):
        self.add_statement(self.current_block, node)
        failblock = self.new_block()
        self.add_exit(self.current_block, failblock, invert(node.test))
        self.cfg.finalblocks.append(failblock)
        successblock = self.new_block()
        self.add_exit(self.current_block, successblock, node.test)
        self.current_block = successblock
        self.goto_new_block(node)

    def visit_If(self, node):
        if len(self.current_block.statements) > 0: 
            current_block = self.new_block()
            self.add_exit(self.current_block, current_block)
            self.current_block = current_block
            
        self.add_statement(self.current_block, node)

        if_block = self.new_block()
        self.add_exit(self.current_block, if_block, node.test)

        afterif_block = self.new_block()

        if len(node.orelse) != 0:
            else_block = self.new_block()
            self.add_exit(self.current_block, else_block, invert(node.test))
            self.current_block = else_block
            for child in node.orelse:
                self.visit(child)
            if not self.current_block.exits:
                self.add_exit(self.current_block, afterif_block)
        else:
            self.add_exit(self.current_block, afterif_block, invert(node.test))

        self.current_block = if_block
        for child in node.body:
            self.visit(child)
        if not self.current_block.exits:
            self.add_exit(self.current_block, afterif_block)

        self.current_block = afterif_block

    def visit_While(self, node):
        loop_guard = self.new_loopguard()
        self.current_block = loop_guard
        self.add_statement(self.current_block, node)
        self.curr_loop_guard_stack.append(loop_guard)
        while_block = self.new_block()
        self.add_exit(self.current_block, while_block, node.test)

        afterwhile_block = self.new_block()
        self.after_loop_block_stack.append(afterwhile_block)
        inverted_test = invert(node.test)
        if not (isinstance(inverted_test, NAMECONSTANT_TYPE) and
                inverted_test.value is False):
            self.add_exit(self.current_block, afterwhile_block, inverted_test)

        self.current_block = while_block
        for child in node.body:
            self.visit(child)
        if not self.current_block.exits:
            self.add_exit(self.current_block, loop_guard)

        self.current_block = afterwhile_block
        self.after_loop_block_stack.pop()
        self.curr_loop_guard_stack.pop()

    def visit_For(self, node):
        loop_guard = self.new_loopguard()
        self.current_block = loop_guard
        self.add_statement(self.current_block, node)
        self.curr_loop_guard_stack.append(loop_guard)
        for_block = self.new_block()
        self.add_exit(self.current_block, for_block, node.iter)

        afterfor_block = self.new_block()
        self.add_exit(self.current_block, afterfor_block)
        self.after_loop_block_stack.append(afterfor_block)
        self.current_block = for_block

        for child in node.body:
            self.visit(child)
        if not self.current_block.exits:
            self.add_exit(self.current_block, loop_guard)

        self.current_block = afterfor_block
        self.after_loop_block_stack.pop()
        self.curr_loop_guard_stack.pop()

    def visit_Break(self, node):
        assert len(self.after_loop_block_stack), "Found break not inside loop"
        self.add_exit(self.current_block, self.after_loop_block_stack[-1])

    def visit_Continue(self, node):
        assert len(self.curr_loop_guard_stack), "Found continue outside loop"
        self.add_exit(self.current_block, self.curr_loop_guard_stack[-1])

    def replace_docstring(self, node):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
            raise TypeError("%r can't have docstrings" % node.__class__.__name__)
        if not(node.body and isinstance(node.body[0], ast.Expr)):
            return None
        node = node.body[0].value
        if isinstance(node, ast.Str):
            node.s = 'docstring'
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = 'docstring'
        else:
            return None

    def visit_FunctionDef(self, node):
        self.replace_docstring(node)
        self.current_class = find_class_from_line(self.src, node.lineno)
        self.add_statement(self.current_block, node)
        self.new_functionCFG(node, asynchr=False)
        self.goto_new_block(node)

        
    def visit_AsyncFunctionDef(self, node):
        self.replace_docstring(node)
        self.current_class = find_class_from_line(self.src, node.lineno)
        self.add_statement(self.current_block, node)
        self.new_functionCFG(node, asynchr=True)
        self.goto_new_block(node)

    def visit_Await(self, node):
        afterawait_block = self.new_block()
        self.add_exit(self.current_block, afterawait_block)
        self.goto_new_block(node)
        self.current_block = afterawait_block

    def visit_Return(self, node):
        self.add_statement(self.current_block, node)
        self.cfg.finalblocks.append(self.current_block)
        self.current_block = self.new_block()

    def visit_Yield(self, node):
        self.cfg.asynchr = True
        afteryield_block = self.new_block()
        self.add_exit(self.current_block, afteryield_block)
        self.current_block = afteryield_block

   