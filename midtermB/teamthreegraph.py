import spacy
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
import nltk

from pypdf import PdfReader
from spacy.language import Language
from spacy.lang.en import STOP_WORDS
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')
# Read the PDF file

def read_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    number_of_pages = len(reader.pages)
    text = ""
    for i in range(number_of_pages):
        page = reader.pages[i]
        text += page.extract_text()
    return text

#  Remove stopwords and tokenizing the sentences
def preprocess_text(text):
  nlp = spacy.load("en_core_web_sm")
  sentences = sent_tokenize(text)
  nlp = spacy.load("en_core_web_sm")
  sentences = sent_tokenize(text)
  processed_sentences = []

  for sentence in sentences:
      doc = nlp(sentence.lower())
      words = {token.text for token in doc if token.is_alpha and token.text not in STOP_WORDS}
      processed_sentences.append(words)

  return processed_sentences
    
# Build Graph Using NetworkX and iGraph 
def build_graph_networkx(sentences):
    
    G = nx.DiGraph()
    
    for i, words1 in enumerate(sentences):
        G.add_node(i, words=words1)

        for j in range(i+1, len(sentences)):
            words2 = sentences[j]
            shared_words = words1.intersection(words2)

            if shared_words:
                G.add_edge(i, j, label=",".join(shared_words), weight=len(shared_words))

    return G

def convert_to_igraph(nx_graph):
    
    edges = [(u, v) for u, v in nx_graph.edges()]
    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(len(nx_graph.nodes()))
    ig_graph.add_edges(edges)
    return ig_graph

#Compute Graph Properties

def analyze_graph(nx_graph, ig_graph):
    
    num_vertices = nx_graph.number_of_nodes()
    num_edges = nx_graph.number_of_edges()
    num_connected_components = nx.number_connected_components(nx_graph.to_undirected())

    # Save results to file
    with open("size.txt", "w") as f:
        f.write(f"Vertices: {num_vertices}\nEdges: {num_edges}\nConnected Components: {num_connected_components}")

    print(f"Graph Analysis:\nVertices: {num_vertices}\nEdges: {num_edges}\nConnected Components: {num_connected_components}")

    # Compute and plot weighted degree distributions
    plot_degree_distribution(nx_graph, "in-deg-dist.png", "in")
    plot_degree_distribution(nx_graph, "out-deg-dist.png", "out")

    # Compute SCCs using iGraph
    scc_dag = ig_graph.connected_components()
    plot_scc_dag(scc_dag, "scc-dag.png")

    # Convert to undirected for BCC computation
    G_undirected = nx_graph.to_undirected()
    bccs = list(nx.biconnected_components(G_undirected))
    plot_bcc_forest(bccs, "bcc-forest.png")


#Visualization Functions

def plot_degree_distribution(G, filename, degree_type="in"):
    
    degrees = dict(G.in_degree(weight="weight") if degree_type == "in" else G.out_degree(weight="weight"))
    
    plt.figure()
    plt.hist(degrees.values(), bins=20)
    plt.xlabel(f"{degree_type.capitalize()}-Degree")
    plt.ylabel("Frequency")
    plt.title(f"Weighted {degree_type.capitalize()}-Degree Distribution")
    plt.savefig(filename)
    plt.close()

def plot_scc_distribution(sccs, filename):
    
    scc_sizes = [len(scc) for scc in sccs]

    plt.figure()
    plt.hist(scc_sizes, bins=20)
    plt.xlabel("SCC Size (vertices)")
    plt.ylabel("Frequency")
    plt.title("SCC Size Distribution")
    plt.savefig(filename)
    plt.close()

# Plot SCC DAG using iGraph.
def plot_scc_dag(scc_dag, filename):
    """Plot SCC DAG using iGraph."""

    # Convert SCCs to a new directed graph
    scc_graph = ig.Graph(directed=True)
    scc_graph.add_vertices(len(scc_dag))  # Each SCC is a node
    scc_graph.add_edges([(i, j) for i, comp1 in enumerate(scc_dag) 
                         for j, comp2 in enumerate(scc_dag) if i != j and any(v in comp2 for v in comp1)])

    # Generate layout and save plot
    layout = scc_graph.layout("kk")
    ig.plot(scc_graph, layout=layout, bbox=(600, 600), margin=20).save(filename)

#Plot BCCs forest using NetworkX.
def plot_bcc_forest(bccs, filename):
    
    BCC_Forest = nx.Graph()
    
    for i, bcc in enumerate(bccs):
        BCC_Forest.add_node(i, words=list(bcc))

    nx.draw(BCC_Forest, with_labels=True)
    plt.savefig(filename)
    plt.close()



if __name__ == "__main__":
    
    transcript_pdf = "lecture_transcript.pdf"  
    transcript_text = read_pdf(transcript_pdf)
    processed_sentences = preprocess_text(transcript_text)

    # Build NetworkX and iGraph representations
    nx_graph = build_graph_networkx(processed_sentences)
    ig_graph = convert_to_igraph(nx_graph)

    # Analyze and visualize graph
    analyze_graph(nx_graph, ig_graph)
