# Importing the necessary libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the 3SAT input from the CSV
data = pd.read_csv("./3SAT_input.csv", header=None)

# Extracting the clauses
clauses = []
a,b = data.shape
for i in range(a):
    temp=[]
    for j in range(b):
        if not(np.isnan(data[j][i])):
            temp.append(int(data[j][i]))
    clauses.append(temp)

# Create a graph
G = nx.Graph()

# Maintain the distinct variables in the expression
variables = set()

i=0

# Add nodes and triangular edges for each clause
for grp in clauses:
    grp_nodes=[]
    for j in range(len(grp)):

        G.add_node(i, name=grp[j])
        grp_nodes.append(i)
        variables.add(abs(int(grp[j])))

        if j>0:
            for k in range(len(grp_nodes)-1):
                G.add_edge(grp_nodes[k],i)
        i+=1

# Set the labels to be the node names
labels = {node: G.nodes[node]["name"] for node in G.nodes()}

# Store the variables and their complements
complements=[]
for ele in variables:
    complements.append([ele, -ele])

# Add edges between each variable and it's complement in every clause
for comp in complements:

    # Ensure nodes with the specified names exist
    node1 = [node for node, data in G.nodes(data=True) if data["name"] == comp[0]]
    node2 = [node for node, data in G.nodes(data=True) if data["name"] == comp[1]]

    # Add edges between nodes with the specified names
    for i in range(len(node1)):
        for j in range(len(node2)):
            G.add_edge(node1[i], node2[j])


# Set the layout
pos = nx.spring_layout(G, dim=3)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Using greedy approach to find the maximal independent set
def maximal_independent_set(graph):
    # Create a copy of the graph to avoid modifying the original
    graph_copy = graph.copy()

    # Initialize an empty set to store the maximal independent set
    max_independent_set = set()

    while graph_copy:
        # Choose a vertex with the maximum degree
        max_degree_vertex = max(graph_copy, key=graph_copy.degree)

        # Add the chosen vertex to the maximal independent set
        max_independent_set.add(max_degree_vertex)

        # Remove the chosen vertex and its neighbors from the graph
        neighbors_to_remove = set(graph_copy.neighbors(max_degree_vertex)) | {max_degree_vertex}
        graph_copy.remove_nodes_from(neighbors_to_remove)

    return max_independent_set


independent_set = maximal_independent_set(G)
print("Maximal Independent Set: ",independent_set)
print("Number of clauses: ", len(clauses))
print("Size of Maximal Independent Set: ", len(independent_set))

# Checking the satisfiability condition
if(len(independent_set) == len(clauses)):
    print("\nConclusion: 3SAT is satisfiable")
else:
    print("\nConclusion: 3SAT is non-satisfiable")

# Denoting independent set elements with color Red
node_colors = ['red' if node in independent_set else 'blue' for node in G.nodes()]

for node, (x, y, z), color in zip(pos.keys(), pos.values(), node_colors):
    ax.scatter(x, y, z, label=labels[node], color=color, s=50)
    ax.text(x+0.03, y, z, labels[node], fontsize=18, color='black')


# Draw edges in 3D space
for edge in G.edges:
    x = [pos[edge[0]][0], pos[edge[1]][0]]
    y = [pos[edge[0]][1], pos[edge[1]][1]]
    z = [pos[edge[0]][2], pos[edge[1]][2]]
    ax.plot(x, y, z, color='black')

# Set labels and title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Reduction: 3SAT to Independent Set')

# Create proxy artists for the legend
node_proxy = plt.Line2D([0], [0], linestyle='none', c='red', marker='o')
edge_proxy = plt.Line2D([0], [0], linestyle='-', c='black')

# Add legend
ax.legend([node_proxy, edge_proxy], ['Independent Set', 'Edges'], loc='upper right')

# Show the 3D graph
plt.show()
