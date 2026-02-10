Computers & Geosciences 196 (2025) 105845

Available online 15 January 2025
0098-3004/© 2025 The Authors. Published by Elsevier Ltd. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).

**Research paper**

# Roadster: Improved algorithms for subtrajectory clustering and map construction

Kevin Buchin $^a$, Maike Buchin $^b$, Joachim Gudmundsson $^c$, Jorren Hendriks $^d$, Erfan Hosseini Sereshgi $^{e,*}$, Rodrigo I. Silveira $^f$, Jorrick Sleijster $^d$, Frank Staals $^g$, Carola Wenk $^e$

$^a$ *TU Dortmund, Germany*
$^b$ *Ruhr University Bochum, Germany*
$^c$ *University of Sydney, Australia*
$^d$ *TU Eindhoven, Netherlands*
$^e$ *Tulane University, United States*
$^f$ *Universitat Politècnica de Catalunya, Spain*
$^g$ *Utrecht University, Netherlands*

---

### A R T I C L E I N F O

**Dataset link:** https://mapconstruction.org/

**Keywords:**
Trajectory data
Map construction
Map inference
Clustering
Computational geometry

### A B S T R A C T

The challenge of map construction involves creating a representation of a travel network using data from the paths traveled by entities within the network. Although numerous algorithms for constructing maps can effectively piece together the overall layout of a network, accurately capturing smaller details like the positions of intersections and turns tends to be more difficult. This difficulty is especially pronounced when the data is noisy or collected at irregular intervals. In this paper we present Roadster, a map construction system that combines efficient cluster computation and a sophisticated method to construct a map from a set of such clusters. First, edges are extracted by producing a number of subtrajectory clusters, of varying widths, which naturally correspond to paths in the network. Second, representative paths are extracted from the candidate clusters. The geometry of each representative path is improved in a process involving several stages, that leads to map edges. The rich information obtained from the clustering process is also used to compute map vertices, and to finally connect them using map edges. An experimental evaluation of Roadster, using vehicle and hiking GPS data, shows that the system can produce maps of higher quality than previous methods.

---

### 1. Introduction

The widespread adoption of smartphones and other devices capable of location tracking results in the collection of massive quantities of data on movement every day. This data includes, for instance, the trajectories of vehicles on roads, individuals hiking in isolated regions, or wildlife moving within their natural environments. The most common sources of such data are global navigation satellite systems (GNSS), such as GPS. GNSS receivers store movement data as tracks or trajectories (informally, a trajectory is a sequence of time-stamped locations).

One of the plentiful uses of trajectory data is *map construction* (Ahmed et al., 2015b) or map updates, especially in areas with frequent road changes. Map construction algorithms are used to automatically produce or update street maps based on GPS trajectories. In general, processing large amounts of trajectory data requires clustering contiguous portions of the trajectories together; this is referred to as *subtrajectory clustering*. Here the core problem is to determine the exact portions of the trajectories that are similar. A related problem, that is of particular interest in map construction, is for a given subtrajectory cluster to construct a *representative curve*, i.e., the road that the cluster corresponds to. Next, we discuss the state of the art in map construction and subtrajectory clustering before summarizing our approach and contribution.

#### 1.1. Map construction

In recent years, various methods for creating maps from movement data have been proposed. Some methods sequentially integrate trajectories to gradually build the map (Ahmed and Wenk, 2012). Other techniques begin by clustering points from the trajectories, followed by connecting these clusters in a subsequent step (Edelkamp and Schrödl, 2003). An alternative strategy involves identifying road intersections first and then linking them (Karagiorgou and Pfoser, 2012). There are approaches that rely on processing images, while others employ force-based strategies to simplify the collection of trajectories and minimize noise (Cao and Krumm, 2009).

The book by Ahmed et al. (2015a) offers a comprehensive review of methodologies developed up until 10 years ago. More recent innovations include CrowdAtlas (Wang et al., 2013), which integrates map matching, clustering, and the connection of intersections, and CellNet (Mariescu-Istodor and Fränti, 2018), primarily an intersection linking algorithm that employs a sophisticated density-based approach for detecting intersections. Additionally, algorithms utilizing topological concepts like Morse theory have been introduced (Wang et al., 2015b; Dey et al., 2017), alongside several other methods employing diverse techniques (Guo et al., 2021; He et al., 2018; Lyu et al., 2021; Prabowo et al., 2019; Zheng et al., 2017; Fu et al., 2020). Moreover, recent approaches such as Huang et al. (2018), Gu et al. (2022), Wang et al. (2015a) were introduced which employ structure learning and deep learning for constructing road networks. For insights on the evaluation of some of these methods, the survey by Hashemi (2017) provides valuable references, as does the more recent survey by Chao et al. (2022). While we are interested in methods that construct the map from trajectories only, there are also many methods to construct the map from satellite or aerial images or that use a combination of these data with trajectories. See Bastani et al. (2021), He et al. (2020) for two recent examples.

Recently, Duran et al. (2020) evaluated several map construction algorithms on hiking data. They found that most approaches are able to obtain a good map at a global scale, but they fail to represent the map accurately at a local scale. Moreover, the issues and artifacts identified reflect limitations of the methods that also appear in other types of trajectories, such as urban vehicle data. The lack of low-level accuracy has a high impact on the perceived quality of the generated map, meaning that current methods are still far from meeting the user’s needs.

**Cluster-based map construction.** The map construction algorithm by Buchin et al. (2017) was inspired by the dual objective of achieving high performance both on a *global* and a *local* scale. Their method involves identifying and linking together relevant clusters of similar subtrajectories. Utilizing clusters of relevant subtrajectories in building maps shows promise because these clusters inherently depict routes within the road network. However, the algorithm’s extensive reliance on the method by Buchin et al. (2011) for subtrajectory clustering means it struggles with scalability and is limited to handling only small datasets. Additionally, two crucial issues remain unresolved in the map construction phase: (i) determining an effective representative curve for a cluster, and (ii) figuring out the best way to geometrically link clusters, especially in terms of positioning intersections. Both are particularly challenging in the presence of noise in the data. In this work, we will tackle both the challenge of faster clustering and better map construction.

**Problem definition.** The input to our problem is a set of $n$ trajectories $\tau$. In this work, a *trajectory* is considered to be a polygonal curve, represented by an ordered sequence of points in 2D. The goal is to output a map, represented as a geometric graph, which the input trajectories traveled on. Each *map vertex* has a location, which corresponds to the intersection point of two or more roads, or the end of a road. Each *map edge* connects two map vertices through a polygonal line.

![Fig. 1. A (k, ℓ, ε) subtrajectory cluster (cyan). The trajectories are in black. The red trajectory is the cluster’s representative with the length ℓ inside the cluster (orange). There are k = 7 trajectories in the cluster.](https://placeholder_image_url)
*Fig. 1. A ($k$, $ℓ$, $\epsilon$) subtrajectory cluster (cyan). The trajectories are in black. The red trajectory is the cluster’s representative with the length $ℓ$ inside the cluster (orange). There are $k = 7$ trajectories in the cluster.*

#### 1.2. Subtrajectory clustering

We briefly review the concept of subtrajectory clustering used in this paper. Recall that $\tau$ is the set of input trajectories. A subtrajectory is simply a portion of a trajectory. A subtrajectory cluster is a set of similar subtrajectories from the trajectories in $\tau$. In particular, we use the notion of ($k$, $ℓ$, $\epsilon$)-clusters introduced by Buchin et al. (2011). They identify the following parameters describing a cluster: the length ($ℓ$) of the longest trajectory in the cluster, the size ($k$) denoting the number of trajectories in the cluster, and the spatial proximity ($\epsilon$) defined as the maximum distance between any two (sub)trajectories in the cluster. As distance measure between (sub)trajectories the popular Fréchet distance is used. Thus, intuitively, a ($k$, $ℓ$, $\epsilon$)-cluster is a set of $k$ subtrajectories, all of which have distance at most $\epsilon$ between each other, and where the longest subtrajectory in the set has length $ℓ$. See Fig. 1 for an illustration.

It has been shown that this form of subtrajectory clustering is NP-complete (Buchin et al., 2011), however the authors gave a polynomial-time 2-approximation algorithm that is based on sweeping the free space diagram (which is usually used for computing the Fréchet distance (Alt and Godau, 1995)) to identify sets of subtrajectories within Fréchet distance $\epsilon$, for given $\epsilon > 0$. In the free space diagram, a subtrajectory cluster of size $k$ and length $ℓ$ using one of the subtrajectories as representative shows up as $k$ monotone subpaths over length $ℓ$, which can be identified efficiently. Following the work by Buchin et al. (2011), subtrajectory clustering under the Fréchet distance has been studied extensively. Gudmundsson and Wong (2022) improved the running time of the algorithm by the use of a dynamic tree data structure. They further show lower bounds on how fast subtrajectory clusters can be computed. Agarwal et al. (2018), Brüning et al. (2022), and Brüning et al. (2023) present alternative models for subtrajectory clustering and algorithms for these models. A survey of recent trajectory clustering methods was provided by Bian et al. (2018). In particular, Damiani et al. (2018) and Gloaguen et al. (2023) discuss trajectory segmentation and clustering. Most recent approaches based on database and deep reinforcement learning have been explored by Pelekis et al. (2017) and Wang et al. (2020) respectively.

![Fig. 2. Trajectories (black) and their clusters (shown by colored polygons) with different parameters. The figure on the left shows some of the clusters with width = ε and on the right, we see computed clusters with width = 2ε. The yellow rectangle shows a cluster that is not maximal. The red polygons show two clusters that are not stable since they merge into one cluster in the right figure. The green clusters are all maximal and stable.](https://placeholder_image_url)
*Fig. 2. Trajectories (black) and their clusters (shown by colored polygons) with different parameters. The figure on the left shows some of the clusters with $width = \epsilon$ and on the right, we see computed clusters with $width = 2\epsilon$. The yellow rectangle shows a cluster that is not maximal. The red polygons show two clusters that are not stable since they merge into one cluster in the right figure. The green clusters are all maximal and stable.*

![Fig. 3. Main components of the Roadster system.](https://placeholder_image_url)
*Fig. 3. Main components of the Roadster system.*

![Fig. 4. A set of trajectories (left) and their free space diagram (right). A (k, ℓ, ε)-cluster (in purple bounding box) with center trajectory Tc[s, t] corresponds to a set of k xy-monotone curves in the free space (highlighted in dark purple) that cross the vertical slab (also in purple) defined by subtrajectory Tc[s, t].](https://placeholder_image_url)
*Fig. 4. A set of trajectories (left) and their free space diagram (right). A ($k$, $ℓ$, $\epsilon$)-cluster (in purple bounding box) with center trajectory $T_c[s, t]$ corresponds to a set of $k$ $xy$-monotone curves in the free space (highlighted in dark purple) that cross the vertical slab (also in purple) defined by subtrajectory $T_c[s, t]$.*

![Fig. 5. Efficient free space diagram computation by tracing feasible segments (blue). Red arrows indicate the order in which these are discovered. Purple dashed arrows show directions that are not feasible.](https://placeholder_image_url)
*Fig. 5. Efficient free space diagram computation by tracing feasible segments (blue). Red arrows indicate the order in which these are discovered. Purple dashed arrows show directions that are not feasible.*

**Relevant clusters.** As computing ($k$, $ℓ$, $\epsilon$)-clusters involves tackling a multi-criteria optimization challenge, the plethora of potential clusters makes it difficult to determine which ones are most suitable. Buchin et al. (2017) introduced an approach to pinpoint relevant clusters of subtrajectories, employing them in map construction. A subtrajectory cluster is deemed *relevant*, if it fulfills criteria of being maximal, stable, and long. In this context, *long* refers to maximizing the length of the subtrajectories. A cluster is considered *maximal* with respect to containment; a cluster $C_1$ contains a cluster $C_2$ if each subtrajectory in $C_1$ contains a subtrajectory in $C_2$. In practice, cluster $C_1$ can fail to contain $C_2$ by just a small margin. Therefore, the implementations of Buchin et al. (2017) and in our paper consider approximate containment by allowing $\lambda$-maximal clusters, with $\lambda = 2\epsilon$, which allows an error of $\epsilon$ on both ends of the cluster trajectories. The *lifespan* interval $[\epsilon_1, \epsilon_2]$ of a cluster encompasses all values of $\epsilon$ for which the cluster retains identical subtrajectories (with $k$ remaining constant and $ℓ$ adjusting only in accordance with $\epsilon$). A cluster is *stable* if $\epsilon_2 - \epsilon_1 \geq \epsilon_1$. See Fig. 2 for an illustration.

**Cluster representatives.** When working with a cluster of subtrajectories, it is convenient to have one curve that represents the whole cluster. This *representative* can be one of the subtrajectories in the cluster, a combination of them, or a totally different one, as long as the representative is close to all the subtrajectories in the cluster. The problem of finding a good representative for a cluster of similar subtrajectories is interesting in its own right. Buchin et al. (2013) proposed the concept of a *median trajectory*, that is, a trajectory that stays in the middle combinatorially and topologically. van Kreveld et al. (2017) proposed a *central trajectory* that stays in the middle with respect to the maximum distance. Ahn et al. (2016) proposed a middle curve that stays in the middle with respect to the discrete Fréchet distance. All of these approaches use parts of the input to construct the representative trajectory, however, they typically result in representatives of high complexity. Recent work has focused on finding representatives of low complexity (Buchin et al., 2019a,b), but for full trajectories. Cao and Krumm (2009) use a force-based approach for computing representative curves. For highly sampled data, this approach is good at producing smooth representatives. However, highly sampled data is often not available, for example, data collected by taxis or hikers is often sparse.

#### 1.3. Overview of Roadster

In this work, we present Roadster, a map construction system that employs subtrajectory clustering to extract potential map edges, followed by a sophisticated process to extract—from the clusters—refined map edges and vertices, which are then combined into a final map. The overall structure of Roadster is illustrated in Fig. 3. In a preprocessing step, we simplify the input trajectories, to ensure that they are not too densely sampled (this step is optional, depending on the sampling density of the input trajectories).

The first main component of Roadster is the efficient computation of relevant subtrajectory clusters. To this end, we build on top of the algorithm by Buchin et al. (2017). However, in order to obtain an algorithm that can scale to larger and more realistic inputs, we present a faster method to compute ($k$, $ℓ$, $\epsilon$)-clusters for the vertex-monotone Fréchet distance for fixed $\epsilon$, see Section 2. The performance of this routine is critical because it is invoked for a large number of combinations of values of ($k$, $ℓ$, $\epsilon$). A second novelty in our system is that we employ a combination of exponential and binary search among $\epsilon$-values to significantly reduce the number of calls to the ($k$, $ℓ$, $\epsilon$)-clustering algorithm, see Section 3.

The second main component of Roadster is the map construction phase. Each cluster identified in the previous phase is associated to one representative, but the geometry of these curves is usually rather poor. In Roadster, initial representatives go through an elaborate refinement process that involves resampling of the curve, repositioning the vertex positions to adjust them better to the whole set of subtrajectories in the cluster, and a specific procedure to detect and enhance turns within the curve. Based on the improved representatives, map vertices (i.e., road intersections) are computed, and later connected to each other by creating the final map edges. Details are presented in Section 4.

The performance of Roadster is evaluated experimentally in Section 5, using hiking and vehicle GPS data, and comparing it to state-of-the-art map construction methods.

#### 1.4. Novelty and contributions

Our main contribution is the Roadster system which combines efficient cluster computation and a sophisticated method to construct a map from a set of such clusters. The system combines and improves previous algorithms for computing Fréchet-based subtrajectory clusters (Section 2), for computing relevant subtrajectory clusters (Section 3), and for map construction based on these clusters (Section 4).

For the cluster computation, Roadster uses a significantly more efficient algorithm than previous methods, thus making it applicable on real-world data sets. The previous algorithm for cluster computation by Buchin et al. (2017) makes a large number of calls to the subtrajectory clustering by Buchin et al. (2011) and as such does not scale beyond small inputs. Roadster includes both a more efficient algorithm for subtrajectory clustering and a novel approach for reducing the number of times we need to invoke this algorithm to compute the relevant clusters.

Our map construction techniques are much more sophisticated than previous cluster-based map construction, resulting in geometrically more precise reconstructions. Buchin et al. (2017) focused on computing relevant clusters, and essentially simply stitch together cluster representatives to obtain a map. Roadster includes new methods for improving the geometry of cluster representatives and for determining suitable locations for vertices of the map, which are then combined to construct the map.

The work presented in this paper is partially based on the MSc theses of two of the authors (Sleijster, 2019; Hendriks, 2020), and a short summary of our results has been presented in Buchin et al. (2020).

### 2. Faster Fréchet clustering

In this section we show how to speed up the subtrajectory clustering algorithm from Buchin et al. (2017). Given parameters $k$, $ℓ$, and $\epsilon$, the algorithm returns a set (cluster) $C$ of at least $k$ subtrajectories whose pairwise distances are at most $2\epsilon$, and the *center trajectory* of the cluster. The center trajectory is one of the $k$ subtrajectories in $C$ that has length $ℓ$ and has distance at most $\epsilon$ to all other subtrajectories in the cluster. As distance between subtrajectories we use the Fréchet distance.

Given a value $\epsilon$, and a sequence of trajectories $T_1, \dots, T_n$, each represented as a function mapping the interval [0, 1] to the plane, the algorithm of Buchin et al. (2017) uses the *free space diagram* of the trajectories. A point $(i+s, j+t) \in [0, n]^2$, with $s, t \in [0, 1]$, in the free space diagram encodes the distance between point $T_i(s)$ and $T_j(t)$. Different distance measures can be used for this; we use the Euclidean distance between projected coordinates. In particular, when this distance is smaller than $\epsilon$ the point $p$ is in the *free space*, and otherwise it is in the *forbidden space*. See Fig. 4 for an illustration. Consider $s, t \in [0, 1]$ such that the subtrajectory $T_c[s, t]$ starting at $T_c(s)$ and ending at $T_c(t)$ has length at least $ℓ$. Then a ($k$, $ℓ$, $\epsilon$)-cluster $C$ with center trajectory $T_c[s, t]$, corresponds to a set of $k$ $xy$-monotone curves$^1$ in the free space, one per trajectory, that all cross the vertical slab defined by $c+s$ and $c+t$. See Fig. 4 for an illustration.

The algorithm of Buchin et al. (2017) explicitly constructs the entire free space diagram, and then searches for such $xy$-monotone curves in the diagram. The main issue is that this diagram has size $O(N^2)$, where $N$ is the total complexity of all trajectories. The main idea of our improved algorithm is to construct *only the free space* in the free space diagram. Since in practice, most of the free space diagram consists of forbidden space, this significantly improves the running time of the algorithm.

In order to efficiently construct only the free space, we use range searching data structures. In particular, we store the edges of the trajectories in a balanced R-tree (Guttman, 1984; Leutenegger et al., 1997) so that given a query point $q$ we can efficiently report all (parts of) edges that are within distance $\epsilon$ from $q$. Given a starting point $q = T_c(s)$ of a candidate center curve $T_c[s, t]$, we can then find all regions of free space on the vertical line through $q$ (See Fig. 4). For each such a region (interval) we can then trace the free space in the vertical slab defined by $T_c[s, t]$, and find the $xy$-monotone curves (see Fig. 5).

**Further Optimizations.** To speed up the clustering, for densely sampled sets (specifically hiking data) we simplify the trajectories using the approximation algorithm by Agarwal et al. (2005) with the Fréchet distance as distance measure. We then work with these simplified trajectories throughout the algorithm. To simplify the implementation, we require that the center trajectory of each cluster is a vertex-to-vertex subtrajectory of one of the input trajectories. Furthermore, we use a variant of the Fréchet distance—the vertex-monotone Fréchet distance (Jacobs, 2016)— in which the monotonicity condition is slightly relaxed, so that the mapping between the two trajectories may be non-monotone in between two adjacent vertices.

---
$^1$ A curve is said to be $xy$-monotone if it is intersected at most once by any vertical or horizontal line.

### 3. Faster computation of relevant clusters

The original map construction algorithm as detailed in Buchin et al. (2017) operates in two stages to compute pertinent clusters. Initially, it generates a superset of clusters by executing the 2-approximation Fréchet clustering algorithm proposed by Buchin et al. (2011) for every conceivable cluster size $k$ and distance threshold $\epsilon$, while maximizing the length $ℓ$ for each cluster. Subsequently, in a secondary filtering stage, the algorithm isolates maximal and stable clusters. The candidate cluster sizes $k$ are chosen from the range $k_{min}, \dots, n$ (we used $k_{min} = 3$ in our experiments), and the candidate distance thresholds $\epsilon$ are chosen from a discrete set $\mathcal{E} = \{i \cdot \Delta \mid 1 \leq i, i \cdot \Delta \leq \epsilon_{max}\}$, where $\Delta$ is some step size (we use $\Delta = 5$ meters), and $\epsilon_{max}$ expresses the maximum distance threshold we wish to consider (we use $\epsilon_{max} = 200$ meters).

#### 3.1. Efficient search for relevant clusters

One limitation of the original algorithm lies in its exhaustive exploration across the extensive array of potential distance thresholds $\epsilon \in \mathcal{E}$. Our method mitigates this issue by merging exponential and binary search techniques, thereby circumventing the computation of clusters that lack stability from the outset. Our algorithm computes a set $S$ of stable maximal clusters. We iterate in an exponential search over $\epsilon \leq \epsilon_{max}$, starting with $\epsilon = 5$ and doubling $\epsilon$ in each iteration. For fixed $\epsilon$ we do:

(1) Compute the set $C(\epsilon)$ of all maximal longest clusters over all $k \in \{k_{min}, \dots, n\}$, as Buchin et al. (2017). (Algorithm 1)

(2) For each $c \in C(\epsilon)$ check if $c \in C(2\epsilon)$; in this case, we know $c$ is stable and we add it to $S$. (Algorithm 2)

(3) Now, for each $c \in C(\epsilon) \setminus S$, we proceed as follows: We know that $c \in C(\epsilon)$ but $c \notin C(2\epsilon)$. Therefore we perform a variant of binary search to find an $\epsilon'$ such that $[\epsilon', 2\epsilon'] \subseteq [\epsilon_1, \epsilon_2]$, where $[\epsilon_1, \epsilon_2]$ is the lifespan of $c$. Namely, we seek $\epsilon'$ so that $c$ is a longest cluster $k = |c|$, the number of trajectories in the cluster $c$, and $\epsilon'$ as well as for $\epsilon$ and $2\epsilon'$. We say that $c$ is stable for $\epsilon'$. Let $d(\epsilon) = \epsilon/4$. We start the recursive search with $\epsilon' := \epsilon/2 + d(\epsilon)$. First we compute the set $L_k(\epsilon')$ of all longest clusters for $k$ and $\epsilon'$, and the set $L_k(2\epsilon')$ using the algorithm of Buchin et al. (2011). If $c \in L_k(\epsilon') \cap L_k(2\epsilon')$, then we know $c$ is stable for $\epsilon'$ and add it to $S$. And if $c \notin L_k(\epsilon')$ and $c \notin L_k(2\epsilon')$ then we know $c$ is not stable. The maximality of $c$ can be checked by computing $L_{k+1}(2\epsilon')$. Since $c$ is already maximal in $C(\epsilon)$ we do not have to compute $L_{k+1}(\epsilon')$. We continue the recursive search in the following two cases: (1) If $c \in L_k(\epsilon')$ but $c \notin L_k(2\epsilon')$, then we recurse on the left with $\epsilon' := \epsilon'/2 + d(\epsilon')$. (2) If $c \in L_k(2\epsilon')$ but $c \notin L_k(\epsilon')$, then we recurse on the right with $\epsilon' := \epsilon' + d(\epsilon')$. (Algorithm 3)

Refer to Fig. 6 for an illustration of the recursive search scenarios. The integration of exponential and binary search is effective because the clusters are monotone in $\epsilon$ in the sense that if a cluster $c \in L_k(\epsilon)$ then there exists a cluster $c' \in L_k(\epsilon')$ that contains $c$ for all $\epsilon' > \epsilon$. Such a cluster $c'$ can be identified within linear time. One can run the binary search until the desired precision of $\epsilon$ is achieved. Here we stop it at $\epsilon$ values that are multiples of five.

---
**Algorithm 1: GenerateMaximalClusters ($\tau$, $\epsilon$, $\lambda$)**
**Input:** Set $\tau = \{T_1, \dots, T_n\}$ of trajectories. Proximity parameter $\epsilon$. Error $\lambda$
**Output:** Set of maximal clusters $C$
1 $k \leftarrow 1$, $C \leftarrow \emptyset$, $M \leftarrow \emptyset$
2 **repeat**
3 $M \leftarrow$ Set of maximal-length ($k$, $ℓ$, $\epsilon$)-clusters on $\tau$
4 $C \leftarrow C \cup M$
5 $k \leftarrow \max\{k + 1, \min_{c \in C} |c|\}$
6 **until** $M = \emptyset$
7 **for all** $C_1 \in C$ in order of decreasing size:
8 **for all** $C_2 \in C$ and $C_1 \neq C_2$:
9 **if** $C_2$ is a sub-cluster of $C_1$:
10 $C \leftarrow C \setminus \{C_2\}$
11 **return** $C$

---

#### 3.2. Runtime improvement

Let $\mathcal{E}'$ be the set of all $\epsilon$ values that we iterate over in our algorithm. From the exponential search that we employ follows that $\mathcal{E}' = \{ \Delta \cdot 2^i \mid i = 1 \dots \log_2 \frac{\epsilon_{max}}{\Delta} \}$. We have $|\mathcal{E}| = \frac{\epsilon_{max}}{\Delta}$ and $|\mathcal{E}'| = \log_2 \frac{\epsilon_{max}}{\Delta}$. Computing $L_k(\epsilon)$ takes $O(N^2)$ time (Buchin et al., 2011), where $N$ is again the total number of vertices over all trajectories. Computing all maximal longest clusters $C(\epsilon)$ for all $k$ takes total $O(nN^2)$ time.

The original algorithm computes $C(\epsilon)$ for all $k$ and for all $\epsilon \in \mathcal{E}$ in time $O(\frac{\epsilon_{max}}{\Delta} nN^2)$. The stable clusters are computed in a postprocessing step. Our method only computes $C(\epsilon)$ for all $\epsilon \in \mathcal{E}'$, and in the worst-case it computes $L_k(\epsilon)$ twice for all $\epsilon \in \mathcal{E} \setminus \mathcal{E}'$. Thus the total runtime is $O(\log_2 \frac{\epsilon_{max}}{\Delta} nN^2 + (\frac{\epsilon_{max}}{\Delta} N^2)) = O(\max(n \log_2 \frac{\epsilon_{max}}{\Delta}, \frac{\epsilon_{max}}{\Delta}) N^2)$. This is an improvement of a factor of $\min(\frac{\epsilon_{max}/\Delta}{\log_2 \epsilon_{max}/\Delta}, n)$ compared to the original algorithm.

![Fig. 6. The initial call to binary search and the two recursive cases.](https://placeholder_image_url)
*Fig. 6. The initial call to binary search and the two recursive cases.*

---
**Algorithm 2: ComputeClusters ($\tau, \epsilon_{max}, \lambda$)**
**Input:** Set $\tau = \{T_1, \dots, T_n\}$ of trajectories, maximum distance threshold $\epsilon_{max}$, and error parameter $\lambda$.
**Output:** Set of stable maximal clusters $S$
1 $S \leftarrow \emptyset$, $U \leftarrow \emptyset$, $\epsilon \leftarrow \Delta$
2 **while** $\epsilon < \epsilon_{max}$ :
3 $U' \leftarrow \emptyset$
4 $C \leftarrow$ GenerateMaximalClusters($\tau, \epsilon, \lambda$)
5 **for all** $c_i \in C$:
6 **if** $c_i \in U$
7 $S \leftarrow S \cup \{c_i\}$
8 $U \leftarrow U \setminus \{c_i\}$
9 **else**
10 $U' \leftarrow U' \cup \{c_i\}$
11 **for all** $c_i \in U$ :
12 $S \leftarrow$ BinSearch($S, |c_i|, c_i, \frac{\epsilon + \epsilon/2}{2}, \frac{\epsilon + \epsilon/2}{4}$)
13 $U \leftarrow U'$
14 $\epsilon \leftarrow 2\epsilon$
15 **return** $S$

---
**Algorithm 3: BinSearch ($S, k, c_i, \epsilon, \epsilon'$)**
**Input:** Set of stable maximal clusters $S$. The cluster $c_i$ and its size $k$. Proximity parameters $\epsilon$ and $\epsilon'$.
**Output:** Updated set of stable maximal clusters $S$
1 $s \leftarrow$ Set of maximal-length ($k, ℓ, \epsilon$)-clusters on $\tau$
2 $s' \leftarrow$ Set of maximal-length ($k, ℓ, \epsilon'$)-clusters on $\tau$
3 **if** $c_i \in s$ and $c_i \in s'$
4 $S \leftarrow S \cup \{c_i\}$
5 **else if** $c_i \in s$ and $c_i \notin s'$
6 $S \leftarrow$ BinSearch($S, k, c_i, \frac{\epsilon + (\epsilon'-\epsilon)/2}{2}, \frac{\epsilon'-\epsilon}{4}$)
7 **else if** $c_i \notin s$ and $c_i \in s'$
8 $S \leftarrow$ BinSearch($S, k, c_i, \frac{\epsilon' + (\epsilon'-\epsilon)/2}{2}, \frac{\epsilon'-\epsilon}{4}$)
9 **return** $S$

---

### 4. Improved map construction from clusters

In this section, we describe how Roadster uses the relevant subtrajectory clusters to construct the road network. Recall from Section 1.1 that our goal is to identify the locations of map vertices, i.e., road intersections, and to compute map edges, i.e., polygonal lines connecting the map vertices. Our map construction algorithm from clusters does this in three steps:

1. Improving the geometry of the cluster center produced by the subtrajectory clustering algorithm, to produce an appropriate cluster representative.
2. Determining the locations of the map vertices, by computing the intersections and endpoints of the cluster representatives.
3. Determining for every pair of map vertices $u$ and $v$ the set of clusters that support an edge between $u$ and $v$, and use their representatives to compute the map edge connecting $u$ and $v$ (when appropriate).

We describe these steps in more detail in Sections 4.1, 4.2, and 4.3, respectively. Even more details can be found in the thesis of Sleijster (2019). When we compute the map vertices and map edges, we critically utilize that for each cluster we do not just have its geometric representation, but also information such as its size and accuracy.

#### 4.1. Improving the geometry of cluster representatives

Given a cluster, we first need to construct a polygonal curve that is a good representation of the path taken by the trajectories in the cluster. The algorithm for subtrajectory clustering (Sections 2 and 3) outputs a center curve per cluster, which we take as starting point. However, since the center curve is one of the subtrajectories in the cluster, it is likely to contain noise and other artifacts.

We start from the center trajectory resulting from the clustering algorithm. To better deal with varying sampling rates in the trajectory data, we resample this curve to make sure that its complexity is similar to that of the other trajectories in the cluster. In particular, we merge two consecutive trajectory points if they are closer than a threshold (in our implementation, 5 m), and then we subdivide edges of the center trajectory if they are too long with respect to the other edges in the cluster that have similar heading. Refer to Section 3.2.1 of Sleijster (2019).

We then move each vertex of the representative towards the middle of the cluster. While our algorithm is inspired by the force-based approach by Cao and Krumm (2009), we make significant changes because our trajectories might be sparsely sampled and because we can utilize the clusters.

Before moving the vertices, we first split long edges of representative. To decide which edges to split, we count the number of edges with similar heading from other trajectories from the corresponding cluster in a neighborhood of the edge (see Fig. 7). We split the edge when this number is large compared to the number of trajectories in the cluster.

After this, we can use a simple approach to move the vertices. For a vertex $v$, we first compute an appropriate normal vector. We do this by traversing the trajectory starting from $v$ in both directions 25 m. This gives us two locations $p$ and $p'$, and we take the normal to the edge $(p, p')$. We now check for edges of other trajectories from the cluster in the direction of the normal from $v$ to both sides. On all such edges that are nearby and have a similar direction as in $v$ (as deduced from the normal vector), we find a location and move $v$ to the median position of those locations (see Fig. 7).

![Fig. 7. To better deal with sparse trajectories, we resample the representative (red): we subdivide an edge when sufficiently many trajectories in the cluster have a similar heading and a vertex in the neighborhood of the edge (the orange region, of fixed width and covering 80% of the edge length). The new vertex positions (white disks) are chosen as the median of corresponding points.](https://placeholder_image_url)
*Fig. 7. To better deal with sparse trajectories, we resample the representative (red): we subdivide an edge when sufficiently many trajectories in the cluster have a similar heading and a vertex in the neighborhood of the edge (the orange region, of fixed width and covering 80% of the edge length). The new vertex positions (white disks) are chosen as the median of corresponding points.*

Next we use linear regression to accurately represent points where the represented trajectories make a turn. In particular, we do the following. For each trajectory in the cluster, we use a simple threshold (25 degrees in our implementation) on angular change to identify relatively straight subtrajectories. We group subtrajectories from the different trajectories when they have similar location and heading, and order them along the cluster. Intuitively, the cluster makes a turn in between two such consecutive groups $S_i$ and $S_{i+1}$ see Fig. 8 (shown in blue and yellow). We use linear regression to find a line $ℓ_i$ representing $S_i$ (specifically, $ℓ_i$ is the line minimizing the sum of the squared distances to the vertices in $S_i$). Note that points close to the turn (in red) are not used for the linear regression. We then consider the intersection point of $ℓ_i$ and $ℓ_{i+1}$ as the location of the turn. To incorporate a turn of the cluster, say at point $p$, into the representative $R$ we replace the vertices of $R$ within some distance $d$ from $p$ by $p$.

#### 4.2. Computing the vertices of the map

Conceptually, map vertices correspond to locations at which clusters intersect. To compute these vertices we compute a set of candidate locations, and group the candidates that are sufficiently close together. The candidate locations in each group are combined yielding a single vertex in the map. Our candidate locations consist of the set of endpoints of (the representatives of) the clusters and the (estimated) locations at which individual clusters make a turn.

Rather than directly combining the cluster endpoints (and estimated turn points) into a single location, we again use the additional information provided by the clusters to get a better estimate of the vertex location. In particular, for every cluster endpoint $p$ we compute a line representing the general direction of the cluster at $p$. As before, we do this by using linear regression on the trajectory vertices in the cluster that are sufficiently close to $p$. Similarly, locations representing a turn give us two such lines. For each pair of lines, we consider the intersection point of these lines and assign it a weight based on the size of the clusters involved and the relative orientation of the lines. The final location of the map vertex (intuitively; the “intersection point” of the clusters) is a weighted average of those points. See Fig. 9 for an example. We refer to Sleijster (2019, Section 4.2) for more details.

![Fig. 8. We estimate the position p of a turn of a set of sparsely sampled trajectories using linear regression.](https://placeholder_image_url)
*Fig. 8. We estimate the position $p$ of a turn of a set of sparsely sampled trajectories using linear regression.*

![Fig. 9. Computation of the map vertices (trajectories shown in blue, final map in red). Left: candidate locations for one intersection. Right: Final location. Source: Figure from the Delta data set.](https://placeholder_image_url)
*Fig. 9. Computation of the map vertices (trajectories shown in blue, final map in red). Left: candidate locations for one intersection. Right: Final location. Source: Figure from the Delta data set.*

#### 4.3. Computing the edges of the map

In the second step of our algorithm we essentially find map edges and represent each of them using (pieces of the representatives of) the clusters. In particular, for each cluster, we compute the list of pairs of adjacent map vertices visited by the cluster. Combining this information from all clusters, we get a list of edges (pairs of vertices) $(u, v)$ together with all clusters $C_{(u,v)}$ that travel from $u$ directly to $v$ (or vice versa). We sort this list in decreasing order on the number of clusters associated with each edge. In case of ties, we prefer edges $(u, v)$ whose endpoints $u$ and $v$ both have a high degree. Since there may be multiple distinct “routes” between $u$ and $v$ in the network (so $(u, v)$ may actually be a multi-edge) we will use all clusters $C_{(u,v)}$ to compute (draw) an accurate geometric representation of $(u, v)$. We order the clusters in $C_{(u,v)}$ by their size $k$ and accuracy $\epsilon$, so that we draw (the representatives of) the larger and most accurate clusters first.

When we process a cluster in $C_{(u,v)}$ we compute the relevant part of its representative $R = r_1, \dots, r_k$ to represent $(u, v)$. In particular, we find the point $r_i$ closest to $u$ on $R$. Starting from $r_i$ we walk along the representative to find a good vertex $r_j$ to connect to $u$; i.e., a vertex such that the angle between the line segments $ur_j$ and $r_j r_{j+1}$ is large. Analogously, we find an appropriate vertex $r_j$ to connect to $v$, and represent the cluster by the polygonal chain $R' = u, r_i, \dots, r_j, v$. If this is the first cluster of $(u, v)$ processed, we simply draw the entire chain $R'$. Otherwise, we want to draw only those parts at which the route between $u$ and $v$ represented by this cluster deviates sufficiently from what has been already drawn. To this end, we filter out the parts of $R'$ that are too close to the chains connecting $u$ and $v$ that are already drawn. For each piece, we use a method similar to the one above to make sure that the sub-chains are connected to the existing chain(s) appropriately. We refer to Hendriks (2020, Chapter 4) for more details.

![Fig. 10. Reconstructed map (red) overlaid on trajectory data (blue) for Athens_s.](https://placeholder_image_url)
*Fig. 10. Reconstructed map (red) overlaid on trajectory data (blue) for Athens_s.*

### 5. Experimental evaluation

In this section we experimentally compare Roadster to previous methods. Roadrunner (He et al., 2018) and Kharita (Stanojevic et al., 2018) are on their respective public Github repositories. For all the other previous methods (Edelkamp (Edelkamp and Schrödl, 2003), Davies (Davies et al., 2006), Cao (Cao and Krumm, 2009), Ge (Ge et al., 2011), Biagioni (Biagioni and Eriksson, 2012), Ahmed (Ahmed and Wenk, 2012), Karagiorgou (Karagiorgou and Pfoser, 2012)) we used the implementations provided by Ahmed et al. (2015a). Our method is assessed using two distinct categories of trajectory data sets: urban and hiking. These particular data sets were selected for their variety and their prior use in evaluating map construction techniques (Ahmed et al., 2015a; Duran et al., 2020).

The urban trajectory data sets include *Athens_s*, *Athens_l*, and *Chicago*, which can be found at mapconstruction.org. The *Athens* data sets feature relatively sparse sampling, in contrast to *Chicago*, which boasts a dense sampling rate.

For hiking trajectories, the data sets used for evaluation are *Aiguamolls*, *Delta*, *Montseny*, and *Garraf*. These are hiking data sets consisting of user-contributed GPS tracks from four different parts of Catalonia, downloaded by Duran et al. (2020) from Wikiloc website.

Table 1 gives an overview of the data sets. All hiking data sets have a relatively high sampling rate and low speed which results in a large number of small edges. Therefore, as mentioned in Section 2, we applied a simplification algorithm (Agarwal et al., 2005) to the four hiking data sets (with $\epsilon = 5$ m). In particular, we used the implementation of the simplification algorithm by Buchin et al. (2019b).

We compare the runtime and quality of the different map construction algorithms on both types of data sets. In addition, for the hiking data sets, we analyze the output of our method for ten common artifacts identified by Duran et al. (2020), which consist of situations where achieving map accuracy at the local level is particularly challenging.

**Table 1**
The data sets we use in our evaluation, with the number of trajectories $n$, their total number of vertices $N$, and their sampling rate (i.e., average number of seconds between consecutive samples).

| Data set | $n$ | Original $N$ | Simplified $N$ | Original average sampling rate (s) |
| :--- | :---: | :---: | :---: | :---: |
| Athens_s | 129 | 2 840 | -- | 34.07 |
| Athens_l | 120 | 72 439 | -- | 30.14 |
| Chicago | 889 | 118 360 | -- | 3.61 |
| Aiguamolls | 101 | 46 116 | 4 676 | 7.29 |
| Delta | 161 | 38 029 | 5 151 | 10.15 |
| Montseny | 101 | 128 181 | 18 901 | 8.56 |
| Garraf | 630 | 288 472 | 51 562 | 8.63 |

**Table 2**
Timings to compute relevant clusters for the athens_s data sets (seconds).

| | Orig ($\times 8$) | FC | FC ($\times 8$) | FC & RC |
| :--- | :---: | :---: | :---: | :---: |
| Athens_xxs | 19 | 5 | 3 | 3 |
| Athens_xs | 454 | 44 | 19 | 13 |
| Athens_s | 3439 | 204 | 69 | 54 |

#### 5.1. Runtime comparison

In this section, we compare the previous implementation by Buchin et al. (2017) (Orig) to our method using only faster clustering (FC) and faster clustering combined with faster relevant clusters (FC&RC). The goal is to evaluate the effectiveness of FC and RC in terms of running time improvement.

Since the previous implementation has a relatively high runtime due to its brute-force nature, we first compare it against our methods on *Athens_s* and two subsets of it, then we compare FC against FC&RC on the larger data sets. Algorithms marked with ‘($\times 8$)’ indicate a (partially) parallelized execution on eight cores.

Table 2 lists the running times, where *Athens_xs* contains the first 60 and *Athens_xxs* the first 24 trajectories of *Athens_s*.

Our methods show a significant speedup compared to the previous implementation and scale better as inputs get larger. Running times for the remaining data sets are listed in Table 3.

**Table 3**
Runtime comparison for other data sets (seconds).

| | Relevant cluster computation | | Map construction |
| :--- | :---: | :---: | :---: |
| | FC | FC & RC | |
| Athens_s | 204 | 54 | 2 |
| Athens_l | 470 663 | 130 422 | 1452 |
| Aiguamolls | 312 | 41 | 14 |
| Delta | 776 | 224 | 19 |
| Montseny | 7393 | 3845 | 88 |

#### 5.2. Global quantitative evaluation

In this section, we assess the performance of our map construction algorithm using the urban data sets. Fig. 10 displays the result of applying our algorithm to the *Athens_s* data set. The generated map accurately reflects the trajectory data. Although it overlooks certain less frequently sampled areas, the overall map is notably “clean” and free from unwanted artifacts. Our reconstructed maps for the other data sets are shown in Figs. 16 and 17.

We conduct a comparative analysis between the outcomes of our map construction algorithm and those generated by the algorithms proposed by Edelkamp and Schrödl (2003), Davies et al. (2006), Cao and Krumm (2009), Ge et al. (2011), Biagioni and Eriksson (2012), Ahmed and Wenk (2012), Karagiorgou and Pfoser (2012), Stanojevic et al. (2018) and He et al. (2018). These results are compared against groundtruth maps sourced from OpenStreetMap (OSM) covering identical regions. Following earlier work on map construction (Biagioni and Eriksson, 2012; He et al., 2018), we use a graph sampling measure to evaluate the similarity between a reconstructed map $G$ with a groundtruth map $G'$, In particular, we use the implementation from Aguilar et al. (2024). The measure has a single parameter matched_distance, and is defined as follows.

All connected components ($CCs$) on $G$ and $G'$ are traversed starting at $v$ and $v'$, an arbitrary vertex of each connected component, respectively. The traversed parts (edges) of the graphs are sampled at a uniform distance of 5 meters, and then a maximum weighted bipartite matching is performed between the sampled point sets, allowing points to be matched only if their distances are at most $matched\_distance_{max}$. Let $s$ be the total number of generated point samples in $G$, let $s'$ be the total number of points samples in $G'$, and let $m$ be the total number of matched point samples, accumulated over all repetitions. Then the precision is defined as $m/s$, recall as $m/s'$, and the F-score $\frac{2m}{s+s'}$ is the harmonic mean of precision and recall.

Note that trying to accurately capture the similarity of two (embedded) graphs by a single number is extremely challenging, as one has to account for both the topology as well as the geometry involved. Hence, a wide range of graph similarity measures has been proposed (Biagioni and Eriksson, 2012; Cheong et al., 2009; Bunke, 1997; Dwivedi and Singh, 2019). As graph sampling has been used before we use it here as well. However, it does not capture all aspects relevant to comparing road networks. For example, it may underestimate the importance of the topology of the graphs, and it may not capture differences in path lengths in the graphs well. So, having a high recall or F-score on this measure unfortunately does not immediately imply that the quality of the output map is good. Therefore, one should be careful by drawing conclusions directly from the results listed here. Instead, we evaluate the results in the next section, using a more qualitative measure. Developing a more comprehensive quantitative map evaluation measure would certainly be interesting. Unfortunately, that is a significant undertaking that is out of the scope for this paper. However, to facilitate visual comparisons, we provide reconstructed maps produced by previous methods in Appendix B.

In Table 4 and Table 5 we show the precision and recall values computed for *Chicago* and *Athens_s* data sets respectively. In terms of precision, our map construction algorithm consistently performs well regardless of the $matched\_distance_{max}$, and outperforms the other algorithms with similar precision, in recall values for the *Chicago* and *Athens_s* data sets.

Since none of the trajectory data sets cover the entire ground-truth map, care has to be taken to evaluate the precision and recall values. Precision assesses the proportion of samples on the reconstructed map that are matched with samples on the ground truth, serving as a reliable quality metric even when the ground truth includes numerous additional streets. However, recall, which gauges the proportion of samples on the ground truth matched to samples on the reconstructed map, is negatively influenced by the presence of extra roads in the ground truth. Consequently, recall values, along with the F-score, are less suitable for comparing with OpenStreetMap (OSM) ground truth maps. In order to fix this issue, in Table 6 and 7 we have also cropped the ground truth maps based on the input trajectories using the method in Aguilar et al. (2024) to achieve more meaningful values for recall and F-score.

Table 8 shows our evaluation on *Athens_l*. While Karagiorgou’s method outperforms ours on the *Athens_l* data set specifically, ROADSTER achieves superior results across all data sets combined, ranking second-best on *Athens_l*.

#### 5.3. Local quantitative evaluation

To supplement our quantitative evaluation, we also evaluate the effectiveness of our approach using the hiking data sets from Duran et al. (2020), which include four distinct sets of hiking paths from various regions across Spain, sourced from Wikiloc (2020) (see Table 1). The data sets *Aiguamolls* and *Delta* are located in flat terrains, featuring a network of paths akin to those found in urban settings. In contrast, *Garraf* and *Montseny* are situated in mountainous areas, some of which are densely forested, leading to higher GPS inaccuracies. Given that all the trajectories are contributed by users, the data sets display considerable variability, having been recorded with a range of devices and varied sampling rates. The overall maps generated for each of the four data sets can be seen in Figs. 18, 19, 20, and 21.

To assess the quality of the maps we produced on a local scale, we adopt the methodology used by Duran et al. (2020). Their analysis covered maps created by five different map construction algorithms for each of the four hiking data sets (Ahmed and Wenk, 2012; Cao and Krumm, 2009; Davies et al., 2006; Edelkamp and Schrödl, 2003; Karagiorgou and Pfoser, 2012). The overarching finding was that while most algorithms could generate maps that appeared satisfactory at a broad level, a detailed examination revealed numerous flaws, leading to an overall impression of poor map quality. They pinpointed ten specific common artifacts, that posed challenges to many of the algorithms under comparison, such as paths with noisy zig-zag patterns or closely situated parallel roads. Our evaluation focuses on the performance of our algorithm at the same sites and using the same data sets as those ten common artifacts, which are labeled [C1] through [C10]. Duran et al. quantified the presence of each artifact by an artifact-specific score that measures to what extent the artifact is present in a map (we refer to Duran et al. (2020) for the definition of each score measure).

Fig. 11 displays the results produced by our algorithm for each identified artifact, while Table 9 shows the performance scores of the five algorithms analyzed by Duran et al. (2020), alongside our method, ordered from best (top) to worst (bottom). As indicated in Table 9, our approach achieves exceptional outcomes, securing the top position in seven out of the ten scenarios. This demonstrates that, overall, our method surpasses the performance of all other algorithms included in the comparison. It is worth commenting on some of these artifacts in some detail, and comparing them to the methods evaluated by Duran et al. (2020). Artifact [C1] (Fig. 12) presents a closed turn followed by a large number of trajectories. While most methods have a tendency to shortcut the turn, Roadster is the only one that obtains an accurate representation. Artifact [C2] (Fig. 13) presents another difficult situation, with a bifurcation at a small angle. Most methods tend to detect the fork too late. In contrast, Roadster obtains a virtually perfect splitting point. There are only three artifacts for which our method does not get a maximum score. In Artifact [C4] the score is based on how well the two parallel paths were reproduced compared to the ground truth. Our method produces a very good map, only with a slight vertical shift, which explains the score. Visual comparison with the best method from Duran et al. (2020) (Fig. 14) confirms that the map produced is very accurate. The score measure in Artifact [C7] evaluates the total road length in the output in relation to the ground truth. The relatively low score of our method is due to a few missing paths. However, it should be noted that the ones included follow the trajectories closely. In contrast, the best method in Duran et al. (2020) (Fig. 15) obtains a better score by adding a large number of non-existent paths. Finally, Artifact [C6] shows a set of trajectories going back and forth. This shows a limitation of our method, which is not designed to detect this situation. Segmenting the trajectories may be a way to overcome this.

### 6. Conclusions and discussion

We introduced Roadster, a novel map construction system based on subtrajectory clustering. We highlighted the versatility of subtrajectory clustering and its potential for map construction. Additionally, we provided various measurements on our reconstructed maps of vehicle and hiking GPS data sets comparing our approach with popular map construction algorithms.

As demonstrated in the experimental evaluation, our contribution to Fréchet-clustering, finding relevant clustering and map construction has successfully allowed us to reconstruct more visually appealing and precise maps. These improvements resulted in a significant decrease in runtime and better quality outputs by incorporating both global and local network features. Roadster adeptly manages irregular sampling rates and noisy datasets, efficiently generating maps for extensive data sets. Our local and global evaluations of Roadster revealed significant detail accuracy while ensuring extensive area coverage, irrespective of the input dataset. An interesting question that arose in this context is which measures to use for map comparison. As we saw in our evaluation, not only F-score, but also coverage, local and global quality determine the resulting map quality.

Though promising avenues for improvement exist, such as enhancements in Fréchet-clustering and novel data structures for further runtime and memory optimization, we believe our current implementation demonstrates substantial utility for mid-to-large road maps and constitutes a valuable contribution to the spatial community. In future work, we would like to further improve efficiency to allow to also construct very large road maps with a similar quality, in particular w.r.t. a local quantitative evaluation. For the evaluation it would be desirable to have a broader range of open benchmark data for map construction, including different types of trajectories (e.g., vehicles, hiking, cycling) and different map scenarios. In addition, we hope this work encourages new advancements in subtrajectory clustering and similarity measures for road networks. In turn, it is an interesting question, how other approaches for subtrajectory clustering, i.e., the ones referenced in Section 1.2, can be integrated into our system. Perhaps using a different clustering model may also further decrease the runtime. Furthermore, it would be of interest to integrate other types of input (e.g., satellite images) into our system.

---

**CRediT authorship contribution statement**

**Kevin Buchin:** Writing – review & editing, Writing – original draft, Validation, Supervision, Resources, Conceptualization. **Maike Buchin:** Writing – review & editing, Writing – original draft, Validation, Supervision, Conceptualization. **Joachim Gudmundsson:** Writing – original draft, Supervision, Conceptualization. **Jorren Hendriks:** Writing – original draft, Visualization, Validation, Software, Methodology, Investigation. **Erfan Hosseini Sereshgi:** Writing – review & editing, Writing – original draft, Visualization, Validation, Software, Methodology, Investigation. **Rodrigo I. Silveira:** Writing – review & editing, Writing – original draft, Validation, Supervision, Resources, Conceptualization. **Jorrick Sleijster:** Writing – original draft, Software, Methodology. **Frank Staals:** Writing – review & editing, Writing – original draft, Supervision, Conceptualization. **Carola Wenk:** Writing – review & editing, Writing – original draft, Validation, Supervision, Resources, Conceptualization.

**Code availability section**

Roadster was implemented in Java. A copy of our code is available at https://github.com/Erfanh1995/Roadster. Links to the individual Wikiloc hiking tracks can be found at https://github.com/Erfanh1995/Roadster/tree/main/hiking_links. All urban data sets are from https://mapconstrution.org/. Please contact shosseinisereshgi@tulane.edu for any additional assistance.

**Declaration of competing interest**

The authors declare the following financial interests/personal relationships which may be considered as potential competing interests: Erfan Hosseini Sereshgi reports financial support was provided by National Science Foundation. Carola Wenk reports financial support was provided by National Science Foundation. Rodrigo I. Silveira reports financial support was provided by Spain Ministry of Science and Innovation. If there are other authors, they declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Acknowledgments**

We thank Roel Jacobs for the initial implementation of the bundling and map construction algorithms, and Vera Sacristán for her participation in an earlier version of this work. We also thank Song-tao He for providing us with Roadrunner’s reconstructed map on Chicago (He et al., 2018). Yes. E. Hosseini Sereshgi and C. Wenk were partially supported by National Science Foundation, United States grants CCF-1637576 and CCF-2107434. R.I. Silveira was partially supported by MICINN through grant PID2019-104129GB-I00/ MCIN/ AEI/ 10.13039/501100011033.

**Appendix A. Constructed maps from our method**

Figs. 10, 16, 17, 18, 19, 20, 21 show the results of our map construction algorithm for all considered datasets.

**Appendix B. Constructed maps from previous methods**

In Figs. 22–36 we include examples of the reconstructed maps obtained by previous methods for the Chicago and Athens_s datasets.

**Data availability**

Urban data is publicly available on https://mapconstruction.org/ and links to individual hiking tracks from Wikiloc are in the code repository.

---

### [List of References Omitted for Brevity - See original document for full reference list]