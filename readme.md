# Differentiable Search of Evolutionary Trees from Leaves

Our work introduces a differentiable approach to phylogenetic tree construction, optimizing both tree and ancestral sequences.

Pre-print - https://www.biorxiv.org/content/10.1101/2023.07.23.550206v1

* [ICML 2023 Workshop (SODS/DiffAE) Poster (PDF)](https://ramith.fyi/assets/pdf/Diff-Evol-Trees_ICML.pdf)
* Eric J. Ma has written a very detailed article on our paper's key contribution : making the trees and sequences differentiable. You can read it [here](https://ericmjl.github.io/blog/2023/8/7/journal-club-differentiable-search-of-evolutionary-trees/). It does a great job at explaining our method.

![Optimization of seqs and tree](https://github.com/diff-trees/diff-evol-tree-search/blob/main/intro_vid.gif)

To run examples in colab, click the below link

<a href="https://colab.research.google.com/github/diff-trees/diff-evol-tree-search/blob/main/run_on_colab.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 

#### **Checklist**

* Make sure to select GPU (or remove the `-g 0` flag when running)
* You can specify your wandb account if you intend to log statistics/tree illustrations


#### **Example : running for trees with 16 leaves**

* To run for different number of leaves change the -l to the desired value

Other params :

* sequence length : `-sl`
* mutations per bifurcation : `-m`
* alphabet size : `-nl`
* epochs/steps : `-e`
* initialization count to run in parallel : `-ic`

During running, every 200 steps it will print the `soft_parsimony_score` and `parsimony_score` (last two values in each line)

```bash
!python train_batch_implicit_diff.py -l 16 -nl 20 -m 50 -sl 256 -tLs [0,0.005,10,50] -lr 0.1 -lr_seq 0.01 -t float64-multi-init-run -p Batch-Run-Maximum-Parsimony -alt -n "Final Run" -g 0 -e 5000 -ai 1 -ic 50 -s 42
```



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
