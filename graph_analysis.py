# Brandon Cazares
# CECS 427 Sec 1
# Professor Ponce
# Due Date: 2/28/2025
# Assignment 2: Social and Large Networks
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse

# Our graph is set to a local variable
graph = None

# these are our parser arguments
def parser_arguments():
    parser = argparse.ArgumentParser()
    # first argument is graph file
    parser.add_argument("graph_file", type=str, help="We input the GML graph file")
    # second argument is components
    parser.add_argument("--components", type=int, help="These are number of components to partition to")
    # third argument is plot
    parser.add_argument("--plot", type=str, choices=['C', 'N', 'P'], help="Plot graph type")
    # fourth argument is verify homophily
    parser.add_argument("--verify_homophily", action="store_true", help="Verify homophily in the graph")
    # fifth argument is verify balanced graph
    parser.add_argument("--verify_balanced_graph", action="store_true", help="Verify if the graph is balanced")
    # sixth argument is our output
    parser.add_argument("--output", type=str, help="Output GML file")

    return parser.parse_args() # This ensures it returns args
# we're going to read from our graph
def read_graph(file_path):
    try:
        return nx.read_gml(file_path)
    except FileNotFoundError:
        print(f"File {e} is not found.")
        exit()
    except Exception as e:
        print(f"Error reading graph: {e}")
        exit()

# next, we partition our graph
def partition_graph(graph, n):
    if n < 1 or n > len(graph.nodes):
        raise ValueError("The number of input components need to be larger or equal to 1 and smaller than or equal to the amount of nodes in our Graph.")
    if nx.is_connected(graph):
        print(f"The graph already is already connected.")
    while nx.number_connected_components(graph) < n:
        betweenesss = nx.edge_betweenness_centrality(graph)
        edge_remove = max(betweenesss, key=betweenesss.get)
        graph.remove_edge(*edge_remove)
        print(f"Edge Removed: {edge_remove}")
    print(f"This graph has been partitioned into {n} connected components.")
    return graph

# then, we plot our graph
def plot_graph(graph, plot_type):
    position = nx.spring_layout(graph)

    if "N" in plot_type:
        overlap = {
            (u, v): len(set(graph.neighbors(u)) & set(graph.neighbors(v))) /
            max(len(set(graph.neighbors(u)) | set(graph.neighbors(v))), 1)
            for u, v in graph.edges()
        }

        min_o, max_o = min(overlap.values(), default=0), max(overlap.values(), default=1)
        edge_widths = [
            2 + 8 * (overlap[e] - min_o) / (max_o - min_o) if max_o > min_o else 2
            for e in graph.edges()
        ]

        minimum_overlap, maximum_overlap = minimum(overlap.values(), default=0), maximum(overlap.vales(), default=1)
        edge_widths = [2 + 8 * (overlap[e] - minimum_o) / (maximum_o - minimum_o) if maximum_o > minimum_o else 2 for e in Graph.edges()]

        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(Graph, position, ax=ax, node_size=200, edge_color="black", width=edge_widths, with_labels=True, picker=True)
        plt.title("We have a neighborhood overlap graph")
        plt.show()

    if "C" in plot_type:
        clustering_values = nx.clustering(graph)
        min_c, max_c = min(clustering_values.values()), max(clustering_values.values())
        node_sizes = [100 + 900 * ((clustering_values[n] - min_c) / (max_c - min_c)) if max_c > min_c else 100 for n in
                      graph.nodes()]
        node_colors = [(graph.degree(n) / max(graph.degree(), default=1), 0, 1) for n in graph.nodes()]

        # cluster_values = nx.clustering(graph)
        # position = nx.spring_layout(Graph)
        # min_c, max_c = minimum(cluster_values.values()), maximum(cluster_values.values())
        #
        # min_s, max_s = 100, 1000
        # scale_factor = max_s - min_s if max_x > min_c else 0;
        # node_sizes = [min_s + ((cluster_values[n] - min_c) / (max_c - min_c) * scale_factor)
        #             if max_c > min_c else min_s for n in Graph.nodes()];
        #
        # degs = {node: val for node, val in Graph.degree()};
        # peak_degree = max(degs.values(), default=1);
        # node_shades = [(degs[n] / peak_degree, 0, 1) for n in Graph.nodes()];

        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(Graph, position, ax=ax, node_size=node_sizes, node_color=node_shades, with_labels=True);
        plt.title("Clustering Coefficients & Degree-Based Color Graph");
        plt.show()

    if "P" in plot_type:
        default_color = (0.5, 0.5, 0.5)
        node_colors = [getattr(graph.nodes[n], "color", default_color) for n in graph.nodes()]

        plt.figure(figsize=(10, 8))
        nx.draw(Graph, layout, node_color=node_colors, with_labels=True)
        plt.title("Node colored by attributes graph")
        plt.show()

# def on_click(event):
#     index = event.ind
#     if index:
#         chosen_node = list(graph.nodes())[index[0]]
#         print("You chose a node: {chosen_node}")
#
#         plt.close()
#         bfs = nx.bfs_tree(graph, chosen_node)
#         position = nx.spring_layout(bfs, chosen_node)
#
#         plt.figure(figsize=(10, 8))
#         nx.draw(bfs, position, with_labels=True, node_size=200)
#         plt.title(f"BFS Tree starting from the chosen node {chosen_node}")
#         plt.show()

def verify_homophily(graph):
    arrtibutes = nx.get_node_attributes(graph, "club") or nx.get_node_attributes(graph, "color")
    if not attributes:
        print("Error: Neither 'club' nor 'color' attributes are found in graph nodes.")
        return
    sameness = sum(1 for u, v in Graph.edges if assocation == assocation[v])
    total_edges = graph.number_of_edges()
    homophily_index = sameness / total_edges if total_edges > 0 else 0
    print(f"Homophily: {homophily_index: .2f}")
    if homophily_index > 0.5:
        print("Our graph has strong homophily")
        return "Yes"
    elif homophily_index == 0.5:
        print("Homophily is neutral")
        return "Neutral"
    else:
        print("Homophily is weak")
        return "None"
# our complete graph is verfiied
def verify_balanced_graph(graph):
    cycles = list(nx.cycle_basis(graph))
    if not cycles:
        print("There are no cycles found in graph")
        return False
    # our cycles for loop
    for cycle in cycles:
        edge_sign_product = 1
        for i in range(len(cycle)):
            node1, node2 = cycle[i], cycle[(i + 1) % len(cycle)]
            sign = graph[node1][node2].get("sign", "+")
            edge_sign_product *= 1 if sign == "+" else -1
        if edge_sign_product == -1:
            print("Graph is imbalanced.")
            return False
        print("Graph is balanced.")
        return True

def main():
    args = parser_arguments()
    print(args)  # Debugging: Check if args is None

    if args is None:
        print("Error: No arguments were parsed.")
        return

    global graph
    graph = read_graph(args.graph_file)

    if args.components:
        graph = partition_graph(graph, args.components)

    if args.plot:
        plot_graph(graph, args.plot)

    if args.verify_homophily:
        verify_homophily(graph)

    if args.verify_balanced_graph:
        verify_balanced_graph(graph)

    if args.output:
        nx.write_graph(graph, args.output)
        print(f"Graph is saved to {args.output}")

if __name__ == "__main__":
    main()