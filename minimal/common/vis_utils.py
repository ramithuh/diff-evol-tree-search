import jax
import numpy as np
import jax.nn as nn
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from netgraph import Graph

from networkx.drawing.nx_pydot import graphviz_layout

def generate_t(n_nodes, n_leaves, seed = None):
  assert n_nodes > n_leaves, "Hey! total nodes must be greater than leaves -_-"

  key = jax.random.PRNGKey(1701)
  if(seed!=None):
    key = jax.random.PRNGKey(seed)

  mat = jnp.zeros((n_nodes,n_nodes))

  order = jnp.arange(0,n_nodes-1,1)
  indices = order.copy()*0

  for i in range(0,n_nodes-1):
    key, subkey = jax.random.split(key)
    id = jax.random.choice(key,jnp.arange(max(n_leaves,i+1),n_nodes,1), [1])
    indices = indices.at[i].set(int(id))

  order = jnp.arange(0,n_nodes-1,1)

  mat = mat.at[order,indices].set(1)

  print(mat)
  return mat

def show_graph_with_labels(adjacency_matrix, n_leaves, return_img = False):
    label_names = {}

    for i in range(0,adjacency_matrix.shape[0]):
        label_names[i] = str(i)

    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())

    gr = nx.DiGraph()
    gr.add_edges_from(edges)

    color_map = []
    for node in gr:
        if(node >= n_leaves):
            color_map.append('red')
        else:
            color_map.append('yellow')

    fig = plt.figure()

    pos = graphviz_layout(gr, prog="dot")
    nx.draw(gr,pos, node_size=500 , labels = label_names,with_labels=True, node_color = color_map)
    plt.close(fig)

    if(return_img):
        return fig
    else:
        fig.show()

def animate_tree(adjacency_matrix, n_leaves, n_ancestors, total_frames = None):
    if(total_frames == None):
        total_frames = adjacency_matrix.shape[0]

    n_all = n_leaves + n_ancestors

    label_names = {}

    for i in range(0,adjacency_matrix.shape[1]):
        label_names[i] = chr(97+i)

    color_map = {}
    for i in range(0,n_all):
        if(i >= n_leaves):
            color_map[i] = 'red'
        else:
            color_map[i] = 'Yellow'

    partitions = [
        list(range(n_leaves)),
        list(range(n_leaves, n_leaves+n_ancestors-1)),
        list(range(n_leaves+n_ancestors-1, n_leaves+n_ancestors))
    ]

    fig, ax = plt.subplots()
    g = Graph(np.ones((n_all, n_all)), edge_layout='curved', edge_width=2, arrows=True, ax=ax,
              node_layout='multipartite', node_layout_kwargs=dict(layers=partitions, reduce_edge_crossings=True),
              node_labels = label_names, node_label_fontdict=dict(size=14),node_color = color_map)

    def update(ii):
        for (jj, kk), artist in g.edge_artists.items():
            if adjacency_matrix[ii, jj, kk]:
                artist.set_visible(True)
            else:
                artist.set_visible(False)
        return g.edge_artists.values()

    animation = FuncAnimation(fig, update, frames=total_frames, interval=200, blit=True)
    return animation
