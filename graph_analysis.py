# Brandon Cazares
# CECS 427 Sec 1
# Professor Ponce
# Due Date: 2/28/2025
# Assignment 2: Social and Large Networks
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import numpy as np
import random
from scipy.stats import ttest_1samp

# These are both our global variables
position = None
original_graph = None
# these are our parser arguments
def parser_arguments():
    parser = argparse.ArgumentParser(description="Analyze a social network graph")
    # first argument is graph file
    parser.add_argument("graph_file", type=str, help="Input the GML graph file as a .gml graph file")
    # second argument is components
    parser.add_argument("--components", type=int, help="These are number of components to partition to")
    # third argument is plot
    parser.add_argument("--plot", type=str, help="We will plot graph type C, N or P")
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
        # file is not found
        print(f"File {file_path} is not found.")
        exit(1)
    except Exception as e:
        # graph reads an error
        print(f"Error reading graph: {e}")
        exit(1)

# next, we partition our graph
def partition_graph(graph, n):
    # our number of connected components
    if nx.number_connected_components(graph) >= n:
        print(f"The graph already has {nx.number_connected_components} or more connected components.")
        return graph
    # a while loop about number of connected components in graph
    while nx.number_connected_components(graph) < n:
        betweenness = nx.edge_betweenness_centrality(graph)
        # edge remove
        edge_remove = max(betweenness, key=betweenness.get)
        # edge remove for graph
        graph.remove_edge(*edge_remove)
        print(f"Edge removed: {edge_remove}")
    # This is successful if it's partitioned
    print(f"This graph has been partitioned into {n} connected components.")
    return graph

# we plot the graph with our higlighting cluster coefficients
def plot_cluster_coefficient(graph):
    # Compute our node positions
    position = nx.spring_layout(graph)

    # Then, we compute our cluster coefficients
    cluster_coefficients = nx.clustering(graph)
    # cluster minimum
    cluster_minimum = min(cluster_coefficients.values())
    # cluser maximum
    cluster_maximum = max(cluster_coefficients.values())

    # this computes node sizes based on cluster coefficient
    # we define both maximum and minimum sizes
    minimum_pixel, maximum_pixel = 100, 1000
    node_size = {
        v: minimum_pixel + ((cluster_coefficients[v] - cluster_minimum) / (cluster_maximum - cluster_minimum) * (maximum_pixel - minimum_pixel))
        if cluster_maximum > cluster_minimum else minimum_pixel
        for v in graph.nodes()
    }

    # Then, compute nodes based on degree
    degrees = dict(graph.degree())
    max_degrees = max(degrees.values())

    # We normalize to [0, 1]
    normalize_degrees = {v: degrees[v] / max_degrees for v in graph.nodes()}

    # We need to assign colors so blue has low and purple has high
    node_colors = [(sv, 0, 1) for sv in normalize_degrees.values()]

    # Draw graph
    plt.figure(figsize=(8, 10))
    nx.draw(
        graph, position,
        node_size = [node_size[v] for v in graph.nodes()],
        # This uses normalized colors
        node_color = node_colors,
        with_labels=True
    )

    plt.show()

# computes neighborhood overlap based on our adjacent nodes
def compute_neighborhood_overlap(graph):
    overlap= {}
    # This stores our overlap until all edges are done
    for u, v in graph.edges():
        # u neighbors
        neighbors_u = set(graph.neighbors(u))
        # v neighbors
        neighbors_v = set(graph.neighbors(v))
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        overlap[(u, v)] = intersection / union if union > 0 else 0
    return overlap

# then, we plot our graph
def plot_graph(graph):
    # This uses global varaibles to retrieve unaltered graphs
    global original_graph, position
    original_graph = graph
    position = nx.spring_layout(graph) # this generates graph layout

    # This is our neighborhood overlap graph
    overlap = compute_neighborhood_overlap()
    # our min and max overlap values are both defined
    min_o, max_o = min(overlap.values(), default=0), max(overlap.values(), default=1)
    edge_widths = [
        2 + 8 * (overlap[e] - min_o) / (max_o - min_o) if max_o > min_o else 2
        for e in graph.edges()
    ]

    # We now draw our graph
    fig, ax = plt.subplots(figsize=(10, 8))

    node_scattering = nx.draw_networkx_nodes(graph, position, ax=ax, node_size=300)

    # Sets picker manually
    node_scattering.set_picker()

    # our network edges
    nx.draw_networkx_edges(graph, position, ax=ax, edge_color="gray", width=edge_widths)
    # our network labels
    nx.draw_networkx_labels(graph, position, ax=ax)

    plt.title("This graph has Neighborhood Overlap")

    # This connects our click
    fig.canvas.mpl_connect("pick_event", on_click)

    plt.show()
# We plot our bfs tree here
def plot_bfs_tree(root):
    # This sets up our BFS of the graph
    bfs_tree = nx.bfs_tree(original_graph, root)
    # our position as hierarchy
    position = hierarchy_position(bfs_tree, root)

    # This draws our BFS graph
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw(bfs_tree, position, with_labels=True, node_size=300, edge_color="blue", node_color="light blue", ax=ax)
    plt.title(f"BFS Tree from Node {root}")

    # This sets button to return to our original graph
    fig.canvas.mpl_connect("button_press_event", lambda event: return_to_original(event, fig))

    plt.show()

# we define our hierarchy position
def hierarchy_position(Graph, root=None, width=2.0, vertical_gap=0.5, vertical_loc=0, x_center=0.5):
    # This recursively iterates through our tree
    position = _hierarchy_position(Graph, root, width, vertical_gap, vertical_loc, x_center)
    # we return a position
    return position

def _hierarchy_position(Graph, root, width=2.0, vertical_gap=0.5, vertical_loc=0, x_center=0.5, position=None, parent=None, parsed=[]):
    # This explores our graph leaves
    if position is None:
        # our position has both a center and vertical location
        position = {root: (x_center, vertical_loc)}
    else:
        # our position as a root
        position[root] = (x_center, vertical_loc)

    # our children as a list to graph our parents
    children = list(Graph.neighbors(root))
    if not isinstance(Graph, nx.DiGraph) and parent is not None:
        children.remove(parent)

    # This organizes the layout because it's easier to see
    if len(children) != 0:
        dx = width / len(children)
        nextx = x_center - width/2 - dx/2
        # our children are defined
        for child in children:
            nextx += dx
            position = _hierarchy_position(Graph, child, width=dx, vertical_gap=vertical_gap, vertical_loc=vertical_loc, x_center=nextx, position=position, parent=root, parsed=parsed)

    # we return our position again
    return position

# This closes our BFS plot since it replots our original graph when clicked
def return_to_original(fig):
    plt.close(fig)
    plot_graph(original_graph)

def on_click(event):
    # This gets the index of the clicked node(s)
    ind = event.ind
    if ind is not None and len(ind) > 0:
        # This gets a corresponding node
        clicked_node = list(original_graph.nodes()[ind[0]])
        # our node is clicked
        print(f"Clicked node: {clicked_node}")

        # This closes the current plot because we need to show our BFS tree
        plt.show()
        plot_bfs_tree(clicked_node)

def plot_attribute_color(graph):
    # This fetches a color
    attribute = "color"
    unique_attributes = set(nx.get_node_attributes(graph, attribute).values())
    if not unique_attributes:
        unique_attributes = {"default"}

    # our color map
    color_map = {attribute: (random.random(), random.random(), random.random()) for attribute in unique_attributes}
    # our node colors
    node_colors = [color_map.get(graph.nodes[node].get(attribute, "default"), (0.5, 0.5, 0.5)) for node in graph.nodes()]
    # our edge colors 
    edge_colors = []

    # This fetches the edge to know if they're negative (black), positive (red) or neither (gray)
    for u, v in graph.edge():
        if 'sign' in graph_edge[u, v]:
            # positive
            if graph.edge[u, v]['sign'] == '+':
                edge_colors.append('red')
            # negative
            elif graph.edge[u, v]['sign'] == '-':
                edge_colors.append('black')
            # neither
            else:
                edge_colors.append('gray')
        # Draws graph
        position = nx.spring_layout(graph)
        plt.figure(figsize=(8, 10))
        nx.draw(graph, position, node_color=node_colors, edge_colors=edge_colors, with_labels=True)
        plt.show()

# We then verify our graph if it has homophily or not
def verify_homophily(graph, attr="color"):
    # This checks if all nodes have this same attribute
    if not all(attr in graph.nodes[n] for n in graph.nodes):
        print(f"Error: Some nodes don't have the '{attr}' attribute")
        return

    # we check if our edges are the same
    sameness = 0
    total_edges = graph.number_of_edges()
    # This counts edges if both sides have the same attribute
    for u, v in graph.edges():
        if graph.nodes[u][attr] == graph.nodes[v][attr]:
            sameness += 1

    # Compute homophily info
    H = sameness / total_edges if total_edges > 0 else 0

    # This generates random expectation for homophily (null hypothesis)
    node_attributes = [graph.nodes[n][attr] for n in graph.nodes]
    random_homophily_results = []

    # 1000 random shuffles
    for _ in range(1000):
        np.random.shuffle(node_attributes)
        shuffled_homophily = sum(
            node_attributes[list(graph.nodes).index(u)] == node_attributes[list(graph.nodes).index(v)]
            for u, v in graph.nodes
        ) / total_edges
        random_homophily_results.append(shuffled_homophily)

    # We use a random student's t-test
    t_stat, p_value = ttest_1samp(random_homophily_results, H)

    # This displays results
    print(f"Observed Homophily: {H:.4f}")
    print(f"T-test p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Homophily is strongly sufficient (p < 0.05)")
    else:
        print("Weak evidence of homophily")

# our complete graph is both verified and balanced
def verify_balanced_graph(graph):
    balance = True
    # this fetches all the cycles in the graph
    cycles = list(nx.cycle_basis(graph))
    # this iterates through every cycle
    for cycle in cycles:
        neg_edges = 0
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            if graph[u][v]["sign"] == "-":
                neg_edges += 1
        # If our negative edges are odd, then we break and keep it false
        if neg_edges % 2 != 0:
            balance = False
            break
    # If negative edges are even then, our graph is balanced
    if balance is True:
        print("Graph is balanced.")
    else:
        print("Graph is imbalanced.")
# main function
def main():
    # We fetch our arguments
    args = parser_arguments()

    # This reads gml file
    graph = nx.read_gml(args.graph_file)

    # This calls partition graph
    if args.components:
        graph = partition_graph(graph, args.components)

    # this plots our graph
    if args.plot:
        if "C" in args.plot:
            plot_cluster_coefficient(graph)
        if "N" in args.plot:
            plot_graph(graph)
        if "P" in args.plot:
            plot_attribute_color(graph)
        else:
            print("Input Invalid: Needs to be C, N or P")

    # this verifies if the graph has homophily
    if args.verify_homophily:
        verify_homophily(graph)

    # this verifies if the graph is balanced
    if args.verify_balanced_graph:
        verify_balanced_graph(graph)

    # this saves the file as an output file
    if args.output:
        nx.write_graph(graph, args.output)
        print(f"Graph saved to {args.output}")

if __name__ == "__main__":
    main()
