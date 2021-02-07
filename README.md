govuk-network-embedding
==============================

> Train a node embedding model (graphSAGE) for automated "Related content" link suggestion.

This repo draws heavily from the [GraphSAGE](http://snap.stanford.edu/graphsage/) framework for inductive representation learning on large graphs. GraphSAGE is used to generate low-dimensional vector representations for nodes (aka node embeddings), and is especially useful for graphs that have rich node attribute information. This framework is manifest in python code as [stellargraph](https://stellargraph.readthedocs.io/en/stable/). We apply this framework within the context of GOV.UK.

The movement of users around GOV.UK can be considered a large graph with many nodes (pages / urls) and edges (movement of users between those pages). We refer to this as a functional network which is distinct from the underlying structural network (pages and their hyperlinks to other pages). We enrich the node attributes of the graph with various features derived from the content of the pages and other sources. GraphSAGE is an inductive framework that leverages node attribute information to efficiently generate representations on previously unseen data.

After training, GraphSAGE can be used to generate node embeddings for previously unseen nodes or entirely new input graphs, as long as these graphs have the same attribute schema as the training data.
