import json

from src.utils.qpl.schema import DBSchema
# from src.inference.qpl.types.qpl_to_type import qpl_tree_to_type, types_str
# from src.inference.qpl.types.type_checker import rec_type_check
from src.utils.qpl.tree import QPLTree
from src.utils.qpl import paths as p

from graphviz import Digraph
from tqdm import tqdm


with open('output/qpl/text_to_qpl/decomposer_cot/completed_trees_decomposer=3616@sampling_completer=3996.json', 'r') as f:
    trees = json.load(f)

with open('output/qpl/text_to_qpl/decomposer_cot/results_decomposer=3616@sampling_completer=3996.json', 'r') as f:
    results = json.load(f)

incorrect_trees = []
for tree, result in zip(trees, results):
    if not result['is_correct']:
        incorrect_trees.append(tree | result)


def viz_qpl_qd(dot, node: dict, fillcolor, schema, parent_id=None, node_id=0):
    # Give each node a unique id for graphviz
    this_id = str(node_id)
    # Customize label as you wish
    label = node['op']
    if 'question' in node:
        label += f'\n{node["question"]}'
    label += f'\n{node["qpl_line"]}'
    # try:
    #     node_tree = QPLTree.from_qpl_lines([l.strip() for l in node.get('pred_qpl', node['qpl']).split(';') if l])
    #     node_type = qpl_tree_to_type(node_tree, schema, strict=False)
    #     type_str = f"Type Inference: " + types_str(node_type.type_count)
    #     type_check_status, type_check_msg = rec_type_check(node_tree, schema, strict=False)
    #     type_str += f"\nType Check: {type_check_status.value} - {type_check_msg}"
    # except Exception as e:
    #     type_str = f"Type Inference Error: {str(e)}"
    # label += f'\n{type_str}'
    dot.node(this_id, label, fillcolor=fillcolor)
    if parent_id is not None:
        dot.edge(parent_id, this_id)
    next_id = node_id + 1
    if node['children']:
        for child in node["children"]:
            dot, next_id = viz_qpl_qd(dot, child, fillcolor, schema, this_id, next_id)
    return dot, next_id


def viz_incorrect_correct(tree_dict: dict, schema) -> tuple[Digraph, int]:
    dot = Digraph()
    dot.attr('node', shape='rectangle', margin='0.2,0.1', style='filled')
    dot, next_id = viz_qpl_qd(dot, tree_dict, fillcolor='lightcoral', schema=schema)
    gold_tree = QPLTree.from_qpl_lines(tree['gold_qpl'].split(' ; ')).to_dict()
    dot, _ = viz_qpl_qd(dot, gold_tree, fillcolor='palegreen', schema=schema, node_id=next_id)
    return dot, next_id

def visualize_tree(tree, schema):
    dot, gold_id = viz_incorrect_correct(tree, schema)

    schema_str = "".join(line+"\\l" for line in str(schema).splitlines())
    dot.node('schema', schema_str, shape='note', fillcolor='lightgrey')
    dot.edge('schema', '0', style="dashed", color="grey")
    dot.edge('schema', str(gold_id), style="dashed", color="grey")
    if tree['error']:
        err_idx = tree['error'].rfind("]")
        error_str = tree['error'][:err_idx+1] + "\n" + tree['error'][err_idx+1:]
        dot.node('error', error_str, shape='note', fillcolor='red')
        dot.edge('error', '0', style="dashed", color="red")
    
    return dot

for i, tree in tqdm(enumerate(incorrect_trees), desc="Visualizing incorrect trees", total=len(incorrect_trees)):
    db_schemas = DBSchema.from_db_schemas_file(p.DB_SCHEMAS_JSON_PATH, apply_lower=False)
    filename = f"{i}_{tree['db_id']}"
    if tree['error'] is not None:
        filename += "_error"
    try:
        dot = visualize_tree(tree, db_schemas[tree['db_id']])
        dot.render(filename=filename, directory="output/qpl/text_to_qpl/graphviz_trees_decomposer_cot", format="svg", cleanup=True)
    except KeyError:
        continue
