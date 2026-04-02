# Differentiable Search of Evolutionary Trees from Leaves

Our work introduces a differentiable approach to phylogenetic tree construction, optimizing both tree and ancestral sequences.

Pre-print - https://www.biorxiv.org/content/10.1101/2023.07.23.550206v1

* [ICML 2023 Workshop (SODS/DiffAE) Poster (PDF)](https://ramith.fyi/assets/pdf/Diff-Evol-Trees_ICML.pdf)
* Eric J. Ma has written a very detailed article on our paper's key contribution : making the trees and sequences differentiable. You can read it [here](https://ericmjl.github.io/blog/2023/8/7/journal-club-differentiable-search-of-evolutionary-trees/). It does a great job at explaining our method.

![Optimization of seqs and tree](https://github.com/diff-trees/diff-evol-tree-search/blob/main/intro_vid.gif)

To run examples in colab, click the below link

<a href="https://colab.research.google.com/github/diff-trees/diff-evol-tree-search/blob/main/run_on_colab.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 

#### **Setup**

```bash
conda create -n trees python=3.12 -y && conda activate trees
pip install -r requirements.txt
```

GPU is auto-detected. Use `-g 0` to select a specific GPU.

#### **Three optimization modes**

```bash
# Bilevel optimization (implicit differentiation) — best results
python search_bilevel.py -l 16 -m 50 -sl 256 -nl 20 -e 5000 -ai 1 -ic 100 -lr 0.1 -lr_seq 0.01 -tLs "[0,0.005,10,50]" -s 42

# Alternating optimization (tree update -> seq update loop)
python search_alt.py -l 16 -m 50 -sl 256 -nl 20 -e 5000 -ai 1 -ic 100 -lr 0.1 -lr_seq 0.01 -tLs "[0,0.005,10,50]" -s 42

# Joint optimization (single optimizer, both param sets)
python search_joint.py -l 16 -m 50 -sl 256 -nl 20 -e 5000 -ic 100 -lr 0.1 -tLs "[0,0.005,10,50]" -s 42
```

Key params :

* `-l` : number of leaves
* `-sl` : sequence length
* `-m` : mutations per bifurcation
* `-nl` : alphabet size
* `-e` : epochs/steps
* `-ic` : initialization count to run in parallel (vmapped)
* `-ai` : for alternating mode: number of seq updates per tree update. For bilevel mode: number of inner solver steps before implicit diff computes the outer gradient.

During running, every 200 steps it will print the `surrogate_cost`, `hard_cost` and `loss` side-by-side.
Tree visualizations and sequence heatmaps are saved to `figures/`.



-------
##### Current Limitations : 

- [ ] Groundtruth trees we evaluate against (optimal solutions) are [`perfect binary trees`](https://xlinux.nist.gov/dads/HTML/perfectBinaryTree.html). 
We need to evaluate on diverse grountruth trees of uneven leaf levels
  - [ ] Full binary trees (ramithuh/differentiable-trees#30)
  - [ ] Then, binary trees in general
- [ ] Get rid of site-wise independence assumption

We are working on these aspects in another repo : https://github.com/ramithuh/differentiable-trees.
Once those are tested and verified, this repo will be updated. 
If you have any suggestions/comments/feedback feel free to reach us.
