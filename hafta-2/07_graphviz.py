import os
import runpy

from graphviz import Digraph


def trace(root):
    nodes = set()
    edges = set()

    def build(node):
        if node not in nodes:
            nodes.add(node)

            for child in node._prev:
                edges.add((child, node))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    graph = Digraph(
        format="png",
        graph_attr={"rankdir": "LR"},
    )

    nodes, edges = trace(root)

    for node in nodes:
        node_id = str(id(node))

        graph.node(
            name=node_id,
            label=(
                "{ "
                + f"{node.label} | "
                + f"data {node.data:.4f} | "
                + f"grad {node.grad:.4f}"
                + " }"
            ),
            shape="record",
        )

        if node._op:
            operation_id = node_id + node._op

            graph.node(
                name=operation_id,
                label=node._op,
            )

            graph.edge(operation_id, node_id)

    for first_node, second_node in edges:
        graph.edge(
            str(id(first_node)),
            str(id(second_node)) + second_node._op,
        )

    return graph


# 04_backward.py dosyasını çalıştırıp içindeki nöron çıktısını al
context = runpy.run_path("04_backward.py")
output = context["o"]

os.makedirs("graphs", exist_ok=True)

graph = draw_dot(output)

output_path = graph.render(
    filename="neuron_computation_graph",
    directory="graphs",
    cleanup=True,
)

print("\nGraph oluşturuldu:")
print(output_path)