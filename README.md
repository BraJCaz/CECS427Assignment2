# Brandon Cazares
# CECS427 Sec 1 
# Assignment 2 Social and Large Networks 
# Due Date: 2/28/2025

Objective
# In this assignment, we're going to modify our graph.py file from assignment 1 because our aim is to practice class-learned concepts to generate graphical representations in Python. 
# These representations mean they critically identify both nodes and edges. 
# These tasks are to master our Python skills including computer cluster coefficients and neighborhood overlaps with a given graph. 
# Therefore, these graphs will be partioned to store thier information from one graph to another because primary groups will be stored in another graph. 

Requirement
# This assignment's requirement is to create a Python program that runs in a terminal because we need to understand that the program accepts optional parameters. The syntax to run this program has this command 

python ./graph_analysis.py graph_file.gml --components n --plot [C|N|P] --verify_homophily --verify_balanced_graph --output out_graph_file.gml

Description of Parameters
# this is the command to execute the Python script graph_analysis.py located in the current directory because we need to understand that this reads the graph_file.gml. 
# Here, we need to know that graph_file.gml is the file that will be used for the analysis and the format here is Graph Modeling Language (.gml), which describes the graph's structure with those attributes. 
# Also, the program should read the attributes of both nodes and edges in a file. 

-- components n 

# This specifies that the graph should be partitioned into n components. 
# This also divides the graph into n subgraphs or clusters.
# This computes the betweeness and removes the edge with the highest value. Repeat this process until you have n components.

-- plot[CINIP]

# This determines how the graph should be plotted. 
# C: if this option's selected, the script will plot the highlighted cluster coefficients.
# The cluster coefficient is proportional to its size. Let cluster_min, and cluster_max be the min and max cluster coefficients, and let cv be the clustering 
# coefficient node of v and pv = (cv - cluster_min) / (cluster_max - cluster_min) of node v.
# The size v is proportional to pv. Let max_pixel and min_pixel be the minimum and maximum pixel sides of our nodes. 
# Therfore, the node v will have size min_pixel + pv(max_pixel - min_pixel).
# N: If this option is selected, the script will plot the graph highlighting neighborhood overlap (which measures how much overlap there is between 
# neighbohoods of adjacent nodes.) Similar to the clustering coefficient.
# In this option, the plot must be interactive. In other words, when the user clicks on a node u, it must display the BFS with u as a root.
# P: If this option's selected, the script will color the node according to the attribute if it is assigned, or a default color if not.

-- verify_homophily 

# The tests for graph homophily are based on the given node colors because we use Student t-test. Homophily determines wheter nodes with the same color are likely to have a connection 

--verify_balanced_graph 

# We also need to check if the graph is balanced based on the assigned edge signs. A balanced graph is one where signs on the edges are consistent with the node attributes.
--output out_graph_file.gml

# Then, we must specify the file to which the final graph and results should be saved. In this part,  out_graph_file.gml is the output file that will receive the updated graph nodes and edge attributes should be also saved in the out_graph_file.gml.

python ./graph_analysis.py graph_file.gml --components 3 --plot CN --output out_graph_file.gml

# After that, we read graph_file.gml and partion it to 3 different components, plot the graph and highlight the clustering coefficient, and save the graph in the out_graph_file.gml

python ./graph_analysis.py homophily.gml --plot P --verify_homophily 

# Finally, we read our graph file because we must also read our balanced_graph.gml, plot the graph and verify that the graph is balanced. 


