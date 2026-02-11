# DOGE: Differentiable Bézier Graph Optimization for Road Network Extraction

Jiahui Sun, Junran Lu, Jinhui Yin, Yishuo Xu, Yuanqi Li, Yanwen Guo*  
Nanjing University, Nanjing, China  
{cgjiahui, junranlu, 522025330134, yishuoxu}@smail.nju.edu.cn, {yuanqili, ywguo}@nju.edu.cn

### Abstract

Automatic extraction of road networks from aerial imagery is a fundamental task, yet prevailing methods rely on polylines that struggle to model curvilinear geometry. We maintain that road geometry is inherently curve-based and introduce the **Bézier Graph**, a differentiable parametric curve-based representation. The primary obstacle to this representation is to obtain the difficult-to-construct vector ground-truth (GT). We sidestep this bottleneck by reframing the task as a global optimization problem over the Bézier Graph. Our framework, **DOGE**, operationalizes this paradigm by learning a parametric Bézier Graph directly from segmentation masks, eliminating the need for curve GT. DOGE holistically optimizes the graph by alternating between two complementary modules: **DiffAlign** continuously optimizes geometry via differentiable rendering, while **TopoAdapt** uses discrete operators to refine its topology. Our method sets a new state-of-the-art on the large-scale SpaceNet and CityScale benchmarks, presenting a new paradigm for generating high-fidelity vector maps of road networks. We will release our code and related data.

---

![Figure 1. Polyline versus curve-based road representations. (a) A polyline approximates a curve with discrete polylines. (b) A parametric Bézier curve representation. A road segment is shown with its four control points (red) that define its geometry.](https://files.oaiusercontent.com/file-2mP0l2VvD3pX055O6Sj9h7A7?se=2024-12-05T17%3A05%3A22Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D604800%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3D1b689a9c-07f9-4171-8b94-811c752697a5.webp&sig=G0N3fR7y6X0I76qXjH9/D8pGvK5n0p0N6q3N2k5C7uI%3D)

### 1. Introduction

Road network graphs, a form of standard-definition (SD) map, represent the geometry and topology of drivable roads. They are foundational to critical applications like route planning, autonomous driving, and urban modeling, where up-to-date and accurate data is paramount [6, 9, 11, 32, 34, 38, 42]. However, manually creating these maps remains a costly and labor-intensive bottleneck [26, 27, 41]. The growing abundance of high-resolution satellite imagery presents a powerful alternative: automatically reconstructing road networks to enable rapid, city-scale updates at a low cost.

Most existing methods rely on a polyline representation, approximating road centerlines with sequences of connected line segments, as shown in Fig. 1. This representation has fundamental limitations: polylines are inefficient, requiring a high density of vertices to accurately model curves; they are cumbersome to edit, as local changes can disrupt geometric smoothness; and only guaranteeing positional (C0) continuity [2, 7, 13, 16, 51].

Our motivation is simple: road geometry is inherently curvilinear (Fig. 1). Thus, we introduce the **Bézier Graph**, to our knowledge the first parametric representation for SD road networks using cubic Bézier curves. This representation is inherently smooth and analytically differentiable, enabling intuitive geometric editing via a few control points while preserving topological structure. It can be uniformly sampled to polylines at arbitrary resolution to remain compatible with existing pipelines. As shown in Sec. 4.4, our curve-based representation better models road networks with fewer nodes and edges than polyline methods, which substantially reduces the computational complexity.

A conventional strategy to reconstruct the Bézier Graph would be to train a model to predict it directly. However, such an approach is immediately confronted by a significant bottleneck: the difficulty of creating the necessary vector GT. This challenge is twofold. First, the vectorization process is inherently ambiguous, leading to many equally valid curve factorizations for any given road network [3, 30]. And relying on a rule-based heuristic transformation algorithm could encode bias in the label. Second, the conversion process can often be complex and brittle. Existing pipelines [5] rely on multi-stage, heuristic-driven procedures including node selection, path splitting, and least-squares fitting with global consistency. The thresholds during the conversion are dataset-dependent and difficult to standardize at scale.

We sidestep these challenges with **DOGE** (**D**ifferentiable **O**ptimization of a **B**ézier **G**raph for road network **E**xtraction), a framework that reconstructs road networks by optimizing a Bézier Graph directly against segmentation masks, thus eliminating the need for vector GT. Instead of building the graph with fixed, sequential decisions, DOGE treats the road network as a parametric graph composed of differentiable curves (a Bézier graph) and holistically refines it against a segmentation mask. This reframes road extraction as a global optimization problem over the graph's continuous geometry and discrete topology. DOGE accomplishes this by decoupling the graph's geometry and topology. It addresses them with two complementary modules: **DiffAlign** pioneers the use of differentiable rendering for continuous geometric alignment, while **TopoAdapt** applies discrete operators to evolve the graph's topology. By alternating between them, DOGE achieves a global, iterative refinement of the graph's shape and connectivity.

In practice, we use a fine-tuned SAM2 [18, 29] for high-quality segmentation supervision. Without requiring vector GT, our method learns a curve-based representation and sets a new state-of-the-art on the large-scale SpaceNet and CityScale benchmarks. We hope our work will encourage adoption of differentiable rendering for challenging, GT-free vector reconstruction tasks across the remote sensing domain. Our key contributions are:
* We reframe road network extraction as a global optimization problem over a parametric, curve-based graph. To our knowledge, our work is the first to enable the end-to-end optimization of such a representation directly from segmentation masks, without vector or topology GT.
* We propose the Bézier Graph, a curve-based representation for road networks that is compact, inherently smooth, and fully differentiable, providing the foundation for our optimization-based approach.
* We introduce DOGE, a differentiable framework that decouples geometric and topological optimization via two complementary modules: *DiffAlign* for differentiable geometry alignment and *TopoAdapt* for discrete topology refinement.

### 2. Related Work

#### 2.1. Polyline-based Road Network Modeling

Prior work leveraging polyline representations can be grouped into three main paradigms based on how they construct the graph.

**Detection and Connection.** These methods first identify road and junctions, then infer connectivity to assemble the graph in a two-stage process [1, 15, 16, 31, 37, 48, 50]. While their specifics vary from node extraction (dense detection, heatmaps, segmentation) to edge inference (pairwise classification, orientation cues, or learned relational reasoning), they all rely on an explicit node-edge decomposition.

**Iterative Growth.** In contrast, iterative methods reconstruct the network via a sequential decision process, growing the graph from seed points [2, 8, 33, 46, 47]. RoadTracer [2] is a canonical example, using a CNN to guide expansion. While effective at capturing local dependencies, these agent-based models can accumulate errors and are inherently local, as previously generated sections of the graph remain fixed.

**One-Shot Reconstruction.** This paradigm infers the entire graph in a single forward pass, either by decoding compact topological tensors [14, 49] or by predicting vectorized primitives that are then assembled [1, 17, 45]. For instance, PaLiS [45] predicts patch-wise line segments, while PolyRoad [17] generates polyline road instances with a transformer.

While effective, the polyline approximation often limits geometric fidelity and editability, and often rely on post-processing for vectorization. These limitations motivate our exploration of curve-based representation, paired with a global dynamic optimization strategy.

#### 2.2. Curve-based Road Network Modeling

The use of parametric curves for end-to-end SD road network extraction from satellite imagery remains largely unexplored. We therefore turn to the related, finer-grained domain of High-Definition (HD) mapping for autonomous driving, where parametric curves like Bézier curves and splines are well-established for modeling precise lane-level geometry [7, 19, 21, 24]. Prior work in this domain has demonstrated various strategies: directly regressing curve control points [12], predicting piecewise segments [28], unifying detection across 2D and 3D modalities [10], and, more recently, forming complete lane graphs with explicit topological constraints [5, 22].

For instance, the lane-level Bézier graphs in [5] enforce geometric smoothness by binding edge tangents to node-shared directions. In contrast, our SD-map formulation offers greater flexibility by optimizing per-edge offsets for internal control points, as detailed in Sec. 3.1.

However, these HD mapping techniques are ill-suited for our task. They operate at lane level in relatively constrained driving scenes, whereas we aim to reconstruct topologically complete SD road networks at city scale from satellite imagery. Moreover, this entire line of work fundamentally relies on dense vector GT annotations for supervision [5, 10, 12, 23, 28], which are subjective and prohibitively expensive to obtain at satellite scale [24]. In contrast, we sidestep this dependency by reframing the problem as a global optimization over a curve-based road graph directly supervised by segmentation masks.

#### 2.3. Differentiable Vector Graphics Rasterization

Differentiable rasterization allows gradients to flow from a pixel-based loss back to the parameters of vector graphics, such as the control points and stroke widths of Bézier curves. This is typically achieved by approximating the non-differentiable, hard-edged pixel boundaries of traditional rasterization with a smooth, differentiable function that measures coverage. The canonical implementation, diffvg [20], provides stable gradients that enable optimizing vector shapes to match a target image.

This technique has proven effective for learning compact vector representations in various domains. Applications include layer-wise image vectorization [25], vector font synthesis [39, 40], and generative art where Bézier strokes are optimized to create sketches or SVGs guided by semantic models like CLIP or diffusion models [35, 36, 43, 44].

In contrast to prior applications that optimize largely unstructured vector primitives, our work is the first to leverage differentiable rendering for large-scale, topology-aware road network extraction, coupling a structured curve-graph representation with a dynamic optimization process that jointly enforces geometric fidelity and topological correctness without requiring vector GT.

### 3. Method

This section details our framework, **DOGE**, which reconstructs road networks by optimizing a Bézier Graph (Sec. 3.1). Our approach decouples this task into two complementary modules: **DiffAlign** (Sec. 3.2), which uses differentiable rendering for continuous geometric optimization, and **TopoAdapt** (Sec. 3.3), which applies discrete operators for topology refinement. These modules form an optimization loop (Sec. 3.4) that holistically refines the graph.

#### 3.1. Parametric Representation of Bézier Graph

We formally model the road layout as a Bézier Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. The node set, $\mathcal{V} = \{v^k\}$, consists of vertices with optimizable 2D positions, $\mathbf{p}_k \in \mathbb{R}^2$, representing intersections, termini, or points along a road's path based on their degree. The edge set, $\mathcal{E} = \{e_k\}$, connects these nodes with curvilinear road segments.

As illustrated in Fig. 2, each edge's geometry is given by a cubic Bézier curve:
$$C_k(t) = \sum_{r=0}^{3} \binom{3}{r} (1-t)^{3-r} t^r \mathbf{P}_{k,r}, \quad t \in [0, 1]. \tag{1}$$

![Figure 2. Parametric definition of a Bézier Graph edge $e_k$. The edge's geometry is defined by a cubic Bézier curve with four control points, $\{\mathbf{P}_{k,r}\}_{r=0}^3$, and an optimizable width, $w_k$. The endpoints $\mathbf{P}_{k,0}$ and $\mathbf{P}_{k,3}$ are anchored to the node positions, while the intermediate points $\mathbf{P}_{k,1}$ and $\mathbf{P}_{k,2}$ control the curvature.](https://files.oaiusercontent.com/file-2mP0l2VvD3pX055O6Sj9h7A7?se=2024-12-05T17%3A05%3A22Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D604800%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3D1b689a9c-07f9-4171-8b94-811c752697a5.webp&sig=G0N3fR7y6X0I76qXjH9/D8pGvK5n0p0N6q3N2k5C7uI%3D)

To ensure a regularized and well-posed representation amenable to optimization, we do not optimize its four control points $\{\mathbf{P}_{k,r}\}_{r=0}^3$ directly. Instead, they are deterministically constructed as follows:
$$\mathbf{P}_{k,0} = \mathbf{p}_i \tag{2a}$$
$$\mathbf{P}_{k,3} = \mathbf{p}_j \tag{2b}$$
$$\mathbf{P}_{k,1} = ((1 - \alpha_{k,0})\mathbf{P}_{k,0} + \alpha_{k,0}\mathbf{P}_{k,3}) + d_{k,0} \cdot \mathbf{n}_{ij} \tag{2c}$$
$$\mathbf{P}_{k,2} = ((1 - \alpha_{k,1})\mathbf{P}_{k,0} + \alpha_{k,1}\mathbf{P}_{k,3}) + d_{k,1} \cdot \mathbf{n}_{ij} \tag{2d}$$

While the endpoints $\mathbf{P}_{k,0}$ and $\mathbf{P}_{k,3}$ are anchored to their corresponding node positions to ensure topological connectivity, we reparameterize the intermediate points $\mathbf{P}_{k,1}$ and $\mathbf{P}_{k,2}$ to reduce the degrees of freedom. As detailed in Eq. 2, their positions are defined by two learnable scalars each: a projection parameter $\alpha_{k,i} \in [0, 1]$ that places a point along the chord $\mathbf{P}_{k,0}\mathbf{P}_{k,3}$, and an offset distance $d_{k,i} \in \mathbb{R}$ that displaces it perpendicularly. This reparameterization regularizes the curve's shape—preventing degeneracies like self-intersections—while maintaining sufficient flexibility. The normal vector $\mathbf{n}_{ij}$ is the normalized perpendicular of the chord vector.

#### 3.2. DiffAlign: Differentiable Geometric Optimization

The geometric optimization is guided by a target segmentation $S$, obtained from a fine-tuned SAM2 model. As illustrated in Fig. 3, the process begins by serializing each Bézier curve into a closed polygon representing a road segment. Further details on this serialization are provided in the supplementary material. These polygons are then passed to a differentiable rasterizer $\mathcal{R}$ (DiffVG [20]) to produce a rendered map of the road network. We then optimize the Bézier Graph by computing losses on this rendered output and applying direct, vector-space regularization between curves. This process refines the graph's geometric parameters $\theta$, which include the node positions $\{\mathbf{p}_k\}$, edge widths $\{w_k\}$, and the reparameterized curve attributes $\{\alpha_{k,i}, d_{k,i}\}$.

![Figure 3. Overview of the DOGE framework. Given a satellite image, a fine-tuned SAM2 provides a target road segmentation $S$. DOGE reconstructs the road network by iteratively optimizing a Bézier Graph $\mathcal{G}$ (Sec. 3.1). The optimization loop alternates between two complementary modules: DiffAlign, which continuously refines the graph's geometry by aligning a differentiable rendering of the graph with $S$ (Sec. 3.2), and TopoAdapt, which discretely evolves the graph's topology (Sec. 3.3).](https://files.oaiusercontent.com/file-2mP0l2VvD3pX055O6Sj9h7A7?se=2024-12-05T17%3A05%3A22Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D604800%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3D1b689a9c-07f9-4171-8b94-811c752697a5.webp&sig=G0N3fR7y6X0I76qXjH9/D8pGvK5n0p0N6q3N2k5C7uI%3D)

**Objective Function** The optimization is driven by a composite objective, $\mathcal{L}_{\text{total}}$, which combines five weighted loss terms. These terms are grouped into a target-alignment loss for data fidelity and four geometric priors that regularize the graph's structure. The total loss is:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{cover}}\mathcal{L}_{\text{cover}} + \lambda_{\text{overlap}}\mathcal{L}_{\text{overlap}} + \lambda_{G1}\mathcal{L}_{G1} + \lambda_{\text{offset}}\mathcal{L}_{\text{offset}} + \lambda_{\text{spacing}}\mathcal{L}_{\text{spacing}}. \tag{3}$$
The $\lambda$ terms are weighting coefficients of each loss components. The specific values are detailed in Sec. 4.1.

**Target–Image Alignment** The **Coverage Loss** is our data-fidelity term. It penalizes the pixel-wise L2 discrepancy between the target segmentation $S$ and the union of rendered edges, ensuring the graph covers the road regions:
$$\mathcal{L}_{\text{cover}} = \left\| \left( \bigcup_{e_k \in \mathcal{E}_t} \mathcal{R}(e_k) \right) - S \right\|_2^2 \tag{4}$$

**Geometric and Topological Priors** To guide the optimization towards a plausible road network, we introduce four regularization priors. These terms operate independently of $S$ and enforce desirable geometric properties.

First, the **Overlap Loss** promotes correct topology by penalizing the total area of improper intersections between rendered road segments. To compute this, an overlap map is formed by summing all individual edge renderings and clipping the result at a value of 1. The loss is the L1 norm of this map, normalized by the number of edges $N_t$:
$$\mathcal{L}_{\text{overlap}} = \frac{1}{N_t} \left\| \max\left(0, \left( \sum_{e_k \in \mathcal{E}_t} \mathcal{R}(e_k) \right) - 1\right) \right\|_1 \tag{5}$$

Second, the **G1 Continuity Loss** encourages tangent alignment at degree-2 nodes. For a node $v_j$ connecting edges $e_a$ and $e_b$, the tangents are defined by vectors $\mathbf{v}_{\text{in}} = \mathbf{p}_j - \mathbf{P}_{a,2}$ and $\mathbf{v}_{\text{out}} = \mathbf{P}_{b,1} - \mathbf{p}_j$. The loss is:
$$\mathcal{L}_{G1} = \frac{1}{N_t} \sum_{\substack{v_j \in \mathcal{V}_t \\ \text{deg}(v_j)=2}} (1 - \cos(\theta_j)) \cdot \mathbb{1}_{(\theta_j < T_{G1})} \tag{6}$$
where $\theta_j$ is the angle between tangents at node $v_j$. The loss activates only when $\theta_j$ falls below a threshold $T_{G1}$, penalizing nearly straight connections to preserve legitimate turns.

Finally, two curve regularization terms penalize ill-formed Bézier geometries. The **Offset Loss** discourages excessive perpendicular offsets of the intermediate control points, while the **Spacing Loss** encourages their projection parameters, $\alpha_{k,0}$ and $\alpha_{k,1}$, to be evenly spaced along the chord connecting the endpoints.
$$\mathcal{L}_{\text{offset}} = \frac{1}{N_t} \sum_{\substack{e_k \in \mathcal{E}_t \\ i \in \{0, 1\}}} \max \left(0, \exp \left( \frac{|d_{k,i}|}{L_k} - \tau_d \right) - 1\right) \tag{7}$$
$$\mathcal{L}_{\text{spacing}} = \frac{1}{N_t} \sum_{e_k \in \mathcal{E}_t} \left( (\alpha_{k,0} - \hat{\alpha}_0)^2 + (\alpha_{k,1} - \hat{\alpha}_1)^2 \right) \tag{8}$$
Here, $L_k = \|\mathbf{P}_{k,3} - \mathbf{P}_{k,0}\|$ is the chord length of edge $e_k$, $\tau_d$ is a threshold for the maximum allowable offset ratio, and the constants $\hat{\alpha}_0, \hat{\alpha}_1$ are set to 1/3 and 2/3 respectively, representing the ideal equidistant positions for the control points along the chord.

#### 3.3. TopoAdapt: Discrete Topology Refinement

While *DiffAlign* continuously refines geometry, *TopoAdapt* complements by applying a set of discrete, heuristic operators to dynamically refine the graph's topology. Notably, these operators rely on a handful of simple, robust thresholds that are kept fixed across all datasets. In Fig. 4, this allows for corrections like adding missing roads or merging redundant nodes. These operators, accelerated by a spatial grid for efficient querying, fall into three categories.

**Road Addition** To grow the graph, we identify regions where the current graph $\mathcal{G}_t$ inadequately covers the target segmentation $S$ by computing a difference map using the current rendered graph:
$$M_{\text{unfit}} = \mathbb{1}[S > \tau_{\text{seg}}] \odot \mathbb{1}[\text{Render}(\mathcal{G}_t) < \tau_{\text{render}}]. \tag{9}$$
From this map, we sample $k$ candidate locations $\{p_i\}$ to instantiate new road segments. For each location $p_i$, a new edge is created by initializing two endpoints, $v'_{i,0}$ and $v'_{i,1}$, at positions randomly offset from the center (e.g., $p'_{i,0/1} = p_i \pm \frac{L}{2}\mathbf{u}_i$ for a small length $L$ and a random unit vector $\mathbf{u}_i$). The connecting Bézier edge is initialized with small, random scalar offsets ($d_{i,0}, d_{i,1}$) to form a nearly straight line. The new nodes and edges are then added to the graph:
$$\mathcal{G}_t \leftarrow (\mathcal{V}_t \cup \mathcal{V}_{\text{new}}, \mathcal{E}_t \cup \mathcal{E}_{\text{new}}). \tag{10}$$

![Figure 4. Optimization dynamics of the Bézier Graph. This figure illustrates the interplay between DiffAlign and TopoAdapt. Key operations are highlighted: graph initialization (iter 0); geometric optimization towards the target (iter 10); overlap separation driven by $\mathcal{L}_{\text{overlap}}$ (iter 20); road addition (iter 30); node merging (iter 40); T-junction creation (iter 50); and collinear edge merging (iter 60).](https://files.oaiusercontent.com/file-2mP0l2VvD3pX055O6Sj9h7A7?se=2024-12-05T17%3A05%3A22Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D604800%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3D1b689a9c-07f9-4171-8b94-811c752697a5.webp&sig=G0N3fR7y6X0I76qXjH9/D8pGvK5n0p0N6q3N2k5C7uI%3D)

**Connectivity Enhancement** We connect nearby graph components with two operators based on proximity.
* **Node Merging:** Pairs of nodes $(v_i, v_j)$ closer than a distance $\epsilon_{\text{merge}}$ are merged into a single new node at their midpoint, which inherits all incident edges.
* **T-Junction Creation:** A node $v_i$ is snapped to a nearby edge $e_j$ if their minimum distance is less than $\epsilon_{\text{merge}}$. This is achieved by splitting $e_j$ at the closest point to $v_i$ and merging the new vertex with $v_i$.

**Graph Simplification and Pruning** To maintain a clean and efficient representation, we periodically apply simplification operators.
* **Collinear Edge Merging:** Degree-2 nodes that lie on nearly straight paths are removed, and their two incident edges are replaced by a single, refitted Bézier curve. A path is deemed straight if the angle between its edge tangents exceeds a threshold.
* **Invalid Edge Pruning:** Geometrically implausible edges, such as those that are too short or too thin, are pruned. Any resulting isolated (degree-0) nodes are also removed.

#### 3.4. Global Dynamic Optimization

Our framework uses global dynamic optimization to holistically refine the road network. As summarized in Algorithm 1, this is achieved via a loop that alternates between *TopoAdapt* and *DiffAlign*. Concretely, we initialize the graph $\mathcal{G}_0$ by applying the Road Addition procedure on the segmentation $S$ to obtain an initial set of Bézier edges covering high-confidence road regions. In each iteration, *TopoAdapt* refines the graph's connectivity $(\mathcal{V}, \mathcal{E})$ before *DiffAlign* optimizes its geometric parameters $(\theta)$. Detailed operator algorithms are in the supplementary material.

| **Algorithm 1:** Optimization Workflow of DOGE |
| :--- |
| **Input:** Target segmentation $S$, Max iterations $T_{\text{max}}$, Learning rate $\eta$ |
| **Output:** Optimized Bézier Graph $\mathcal{G}^* = (\mathcal{V}^*, \mathcal{E}^*, \theta^*)$ |
| 1 $\mathcal{G}_0 = (\mathcal{V}_0, \mathcal{E}_0, \theta_0) \leftarrow \text{InitializeGraph}()$; |
| 2 **for** $t = 0$ **to** $T_{\text{max}} - 1$ **do** |
| 3 $\quad (\mathcal{V}'_t, \mathcal{E}'_t) \leftarrow \text{TopoAdapt}(\mathcal{V}_t, \mathcal{E}_t)$; |
| $\quad \mathcal{G}'_t \leftarrow (\mathcal{V}'_t, \mathcal{E}'_t, \theta_t)$; |
| 4 $\quad \theta_{t+1} \leftarrow \text{DiffAlign}(\mathcal{G}'_t, S, \eta)$; |
| 5 $\quad \mathcal{G}_{t+1} \leftarrow (\mathcal{V}'_t, \mathcal{E}'_t, \theta_{t+1})$; |
| 6 **end** |
| 7 **return** $\mathcal{G}_{T_{\text{max}}}$; |

---

### 4. Experiments

We evaluate on two public benchmarks: **City-Scale (Sat2Graph)** and **SpaceNet**. Unless otherwise stated, all images are standardized to 1 m/pixel following prior work, consistent with SAMRoad and SAMRoad++ [16, 48].

**City-Scale (Sat2Graph).** City-Scale (Sat2Graph) [14] contains 180 RGB satellite tiles at 2048 × 2048 covering multiple U.S. cities, with road networks provided as vector graphs. We adopt the common split 144/9/27 for train/val/test, identical to Sat2Graph, RNGDet++, and SAMRoad for fair comparison.

**SpaceNet.** SpaceNet [11] comprises roughly 2.5k RGB tiles at 400 × 400 from diverse world cities, with vector road-graph annotations. We follow prior work (Sat2Graph/RNGDet++/SAMRoad) and use the 2042/127/382 train/val/test split. Following SAMRoad/SAMRoad++ preprocessing, images are resampled to 1 m/pixel for consistent resolution.

#### 4.1. Implementation Details

Our framework is implemented in PyTorch, with differentiable rendering built upon DiffVG [20]. The optimization is conducted on a single NVIDIA RTX 4090 GPU.

**Target Road Segmentation.** We use a fine-tuned SAM2 model to generate target road segmentation masks. The model is trained on the respective training splits of each dataset. Further details on the segmentation model's architecture and training are in the supplementary material.

**Optimization.** We use the Adam optimizer for geometric parameters. The optimization runs for a maximum of $T_{\text{max}} = 300$ iterations, although we employ an early stopping strategy detailed in the supplementary material. Loss weights in Eq. 3 are: $\lambda_{\text{cover}} = 1.0, \lambda_{\text{overlap}} = 0.3, \lambda_{G1} = 0.012$, and $\lambda_{\text{offset}} = \lambda_{\text{spacing}} = 6 \times 10^{-3}$. The rendering resolution is $512 \times 512$. For *TopoAdapt*, we use a unified proximity threshold of $\epsilon_{\text{merge}} = 4\text{m}$ for both node merging and T-junction creation. All *TopoAdapt* hyperparameters are kept fixed across both datasets, underscoring the module's robustness. A comprehensive list of all hyperparameters can be found in the supplementary material.

#### 4.2. Evaluation Metrics

We follow standard practice [14, 16, 47] and evaluate with the **TOPO** [4] and **APLS** [11] metrics. These metrics primarily assess topology but include weak geometric constraints. For fair comparison with prior work, our Bézier graph is converted to a polyline compatible with the official evaluation scripts, using their public implementations and default parameters. We obtain these polylines by uniformly sampling points along each Bézier edge, ensuring a consistent discretization across all methods. The conversion is detailed in the supplementary material.

#### 4.3. Comparative Results

We evaluate our method, DOGE, against state-of-the-art methods on the SpaceNet and City-Scale benchmarks. The qualitative results in Fig. 5 highlight the advantages of our approach. DOGE produces road networks that are significantly smoother and more geometrically accurate than those from prior methods like RNGDet++ and SAMRoad++. This is a direct benefit of our Bézier curve representation, which naturally models the curvilinear nature of roads. Furthermore, our adaptive algorithm efficiently partitions road geometry, using more segments for high-curvature areas and fewer for straight sections. This allows DOGE to effectively model complex intersections while avoiding the jagged artifacts common in other approaches.

The quantitative results in Tab. 1 confirm the superiority of our approach. DOGE sets a new state-of-the-art on both datasets, leading in TOPO F1 on SpaceNet (**84.58**) and APLS on City-Scale (**70.24**). This success is driven by our method's ability to substantially improve recall for a more complete reconstruction, while maintaining a high precision that leads to a superior trade-off on both benchmarks. This demonstrates our method's strength in reconstructing both topologically complete and geometrically accurate road networks. Our full analysis, including cross-dataset tests and experiments, can be found in the supplementary material.

![Figure 5. Qualitative comparison on SpaceNet (top two rows) and City-Scale (bottom two rows). Our method produces geometrically precise, smooth, and topologically correct road graphs, outperforming prior methods across different scales. Notably, our approach uses a more compact graph representation with fewer nodes.](https://files.oaiusercontent.com/file-2mP0l2VvD3pX055O6Sj9h7A7?se=2024-12-05T17%3A05%3A22Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D604800%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3D1b689a9c-07f9-4171-8b94-811c752697a5.webp&sig=G0N3fR7y6X0I76qXjH9/D8pGvK5n0p0N6q3N2k5C7uI%3D)

**Table 1. Performance comparison with state-of-the-art methods on the SpaceNet and City-Scale datasets. The best results are shown in bold, and the second-best are underlined.**

| Method | SpaceNet | | | | City-Scale | | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | TOPO F1 ↑ | Precision ↑ | Recall ↑ | APLS ↑ | TOPO F1 ↑ | Precision ↑ | Recall ↑ | APLS ↑ |
| Sat2Graph | 80.97 | 85.93 | 76.55 | 64.43 | 76.26 | 80.70 | 72.28 | 63.14 |
| RNGDet | 81.13 | 90.91 | 73.25 | 65.61 | 76.87 | 85.97 | 69.87 | 65.75 |
| RNGDet++ | 82.81 | 91.34 | 75.24 | 67.73 | 78.44 | 85.65 | 72.58 | 67.76 |
| SAMRoad | 80.52 | 93.03 | 70.97 | 71.64 | 77.23 | **90.47** | 67.69 | 68.37 |
| SAMRoad++ | 81.57 | **93.68** | 72.23 | 73.44 | 80.01 | 88.39 | 73.39 | 68.34 |
| **DOGE** | **84.58** | 93.55 | **78.43** | **73.48** | **80.59** | 84.42 | **77.40** | **70.24** |

---

![Figure 6. Performance versus compactness on the City-Scale dataset. The plot shows APLS against edge density (edges/km). DOGE achieves the best result, delivering the highest APLS score with a significantly more compact graph representation—using 73.2% fewer edges per kilometer than the GT.](https://files.oaiusercontent.com/file-2mP0l2VvD3pX055O6Sj9h7A7?se=2024-12-05T17%3A05%3A22Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D604800%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3D1b689a9c-07f9-4171-8b94-811c752697a5.webp&sig=G0N3fR7y6X0I76qXjH9/D8pGvK5n0p0N6q3N2k5C7uI%3D)

#### 4.4. Analysis of Topological Compactness

Our method produces road networks that are more accurate and significantly more compact than prior work, a dual advantage evident on the City-Scale dataset. As Fig. 6 illustrates, our approach achieves a higher topological accuracy (APLS) with a substantially lower edge density (edges/km). This superior trade-off between fidelity and compactness is a direct result of our global dynamic optimization strategy. Tab. 2 further quantifies this efficiency, and a full analysis is provided in the supplementary material.

**Table 2. Compactness and performance of graph representations on the City-Scale test set.**

| Method | Nodes/km ↓ | Edges/km ↓ | APLS ↑ |
| :--- | :---: | :---: | :---: |
| GT | 49.36 | 52.17 | — |
| Sat2Graph | 18.67 | 22.69 | 63.14 |
| RNGDet | 42.64 | 45.38 | 65.75 |
| RNGDet++ | 43.37 | 46.22 | 67.76 |
| SAMRoad | 39.25 | 41.34 | 68.37 |
| SAMRoad++ | 41.70 | 44.49 | 68.34 |
| **DOGE** | **11.26** | **14.00** | **70.24** |

#### 4.5. Ablation Studies

We conduct ablation studies on the SpaceNet dataset to validate our key contributions. We first establish a baseline model that only uses a basic coverage loss to optimize the Bézier graph against the segmentation mask, without any of our proposed geometric priors or the *TopoAdapt* module. As shown in Tab. 3, this baseline performs poorly, confirming that a simple application of differentiable rendering is insufficient.

**Effectiveness of Geometric Priors.** Our geometric priors are crucial for plausible road geometry. Removing them entirely causes a severe degradation in performance (Tab. 3). Fig. 7 visually confirms this, illustrating how ablating these priors leads to artifacts like unnatural overlaps and sharp curvatures.

**Effectiveness of Topology Editing.** The *TopoAdapt* module is vital for achieving a complete and correct road network topology. Disabling this module, as shown in the corresponding ablation, causes a significant drop in metric scores as the graph can no longer be topologically refined. The baseline's poor performance further underscores that both geometric and topological enhancements are indispensable.

**Table 3. Ablation study of key components on the SpaceNet dataset. We start with the full model and ablate key components to show their impact.**

| Method | Geometric Priors | | | | TopoAdapt | TOPO F1↑ | APLS↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | $\mathcal{L}_{\text{overlap}}$ | $\mathcal{L}_{G1}$ | $\mathcal{L}_{\text{offset}}$ | $\mathcal{L}_{\text{spacing}}$ | | | |
| Full Model | ✓ | ✓ | ✓ | ✓ | ✓ | **84.58** | **73.48** |
| w/o TopoAdapt | ✓ | ✓ | ✓ | ✓ | | 82.80 | 69.45 |
| w/o Overlap Loss | | ✓ | ✓ | ✓ | ✓ | 83.38 | 65.95 |
| w/o Curve Reg. | ✓ | ✓ | | | ✓ | 82.57 | 65.97 |
| w/o Geom. Priors | | | | | ✓ | 74.58 | 51.18 |
| Baseline ($\mathcal{L}_2$ Only) | | | | | | 70.94 | 37.10 |

---

![Figure 7. Effectiveness of our proposed geometric and topological priors. The top row shows the ideal result from our full model. The bottom row illustrates distinct failure cases (circled) when specific priors are removed: (a) without the Overlap Loss, roads incorrectly intersect; (b) without G1 Continuity, junctions have sharp, unnatural angles; and (c) without the Offset and Spacing Losses, curves become degenerate.](https://files.oaiusercontent.com/file-2mP0l2VvD3pX055O6Sj9h7A7?se=2024-12-05T17%3A05%3A22Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D604800%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3D1b689a9c-07f9-4171-8b94-811c752697a5.webp&sig=G0N3fR7y6X0I76qXjH9/D8pGvK5n0p0N6q3N2k5C7uI%3D)

### 5. Conclusion

We introduced DOGE, a new paradigm for road network extraction that reframes the task as a global optimization of a parametric Bézier Graph. By pioneering the use of differentiable rendering to align with segmentation masks, our method eliminates the need for vector GT. By decoupling geometric optimization (via *DiffAlign*) and topological refinement (via *TopoAdapt*), we reconstruct topologically accurate, geometrically smooth, and compact road networks, achieving SOTA results on the SpaceNet and City-Scale benchmarks. We hope to extend the DOGE framework to other GT-free vector reconstruction tasks.

---

### References

[1] Gaetan Bahl, Mehdi Bahri, and Florent Lafarge. Single-Shot End-to-end Road Graph Extraction. In *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pages 1402–1411. IEEE, 2022.

[2] Favyen Bastani, Songtao He, Sofiane Abbar, Mohammad Alizadeh, Hari Balakrishnan, Sanjay Chawla, Sam Madden, and David DeWitt. RoadTracer: Automatic Extraction of Road Networks from Aerial Images. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4720–4728. IEEE, 2018.

[3] Mikhail Bessmeltsev and Justin Solomon. Vectorization of line drawings via polyvector fields. *ACM Transactions on Graphics (TOG)*, 38(1):1–12, 2019.

[4] James Biagioni and Jakob Eriksson. Inferring road maps from global positioning system traces: Survey and comparative evaluation. *Transportation research record*, 2291(1):61–71, 2012.

[5] Hugh Blayney, Hanlin Tian, Hamish Scott, Nils Goldbeck, Chess Stetson, and Panagiotis Angeloudis. Bézier Everywhere All at Once: Learning Drivable Lanes as Bézier Graphs. In *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 15365–15374. IEEE, 2024.

[6] Ming-Fang Chang, John Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett, De Wang, Peter Carr, Simon Lucey, Deva Ramanan, and James Hays. Argoverse: 3d tracking and forecasting with rich maps. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 8740–8749, 2019.

[7] Ziyi Chen, Liai Deng, Yuhua Luo, Dilong Li, Jose Marcato Junior, Wesley Nunes Gonçalves, Abdul Awal Md Nurunnabi, Jonathan Li, Cheng Wang, and Deren Li. Road extraction in remote sensing data: A survey. *International journal of applied earth observation and geoinformation*, 112: 102833, 2022.

[8] Hang Chu, Daiqing Li, David Acuna, Amlan Kar, Maria Shugrina, Xinkai Wei, Ming-Yu Liu, Antonio Torralba, and Sanja Fidler. Neural turtle graphics for modeling city road layouts. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019.

[9] Ilke Demir, Krzysztof Koperski, David Lindenbaum, Guan Pang, Jing Huang, Saikat Basu, Forest Hughes, Devis Tuia, and Ramesh Raskar. DeepGlobe 2018: A challenge to parse the earth through satellite images. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2018.

[10] Zhiwei Dong, Xi Zhu, Xiya Cao, Ran Ding, Wei Li, Caifa Zhou, Yongliang Wang, and Qiangbo Liu. BézierFormer: A unified architecture for 2d and 3d lane detection. *2024 IEEE International Conference on Multimedia and Expo (ICME)*, pages 1–6, 2024.

[11] Adam Van Etten, David Lindenbaum, and Todd M. Bacastow. SpaceNet: A remote sensing dataset and challenge series, 2018.

[12] Zhengyang Feng, Shaohua Guo, Xin Tan, Ke Xu, Min Wang, and Lizhuang Ma. Rethinking Efficient Lane Detection via Curve Modeling. In *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 17041–17049. IEEE, 2022.

[13] James D Foley. *Computer graphics: principles and practice*. Addison-Wesley Professional, 1996.

[14] Songtao He, Favyen Bastani, Satvat Jagwani, Mohammad Alizadeh, Hari Balakrishnan, Sanjay Chawla, Mohamed M. Elshrif, Samuel Madden, and Mohammad Amin Sadeghi. Sat2Graph: Road Graph Extraction Through Graph-Tensor Encoding. In *Computer Vision – ECCV 2020*, pages 51–67. Springer International Publishing, 2020.

[15] Yang He, Ravi Garg, and Amber Roy Chowdhury. TD-Road: Top-Down Road Network Extraction with Holistic Graph Construction. In *Computer Vision – ECCV 2022*, pages 562–577. Springer Nature Switzerland, 2022.

[16] Congrui Hetang, Haoru Xue, Cindy Le, Tianwei Yue, Wenping Wang, and Yihui He. Segment Anything Model for Road Network Graph Extraction, 2024.

[17] Yuan Hu, Zhibin Wang, Zhou Huang, and Yu Liu. PolyRoad: Polyline Transformer for Topological Road-Boundary Detection. *IEEE Transactions on Geoscience and Remote Sensing*, 62:1–12, 2024.

[18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment Anything. In *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 3992–4003. IEEE, 2023.

[19] Qi Li, Yue Wang, Yilun Wang, and Hang Zhao. HDMapNet: An Online HD Map Construction and Evaluation Framework. In *2022 International Conference on Robotics and Automation (ICRA)*, pages 4628–4634, 2022.

[20] Tzu-Mao Li, Michal Lukač, Michaël Gharbi, and Jonathan Ragan-Kelley. Differentiable vector graphics rasterization for editing and learning. *ACM Transactions on Graphics*, 39(6):1–15, 2020.

[21] Bencheng Liao, Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Qian Zhang, Wenyu Liu, and Chang Huang. MapTR: Structured modeling and learning for online vectorized HD map construction. In *International Conference on Learning Representations*, 2023.

[22] Bencheng Liao, Shaoyu Chen, Bo Jiang, Tianheng Cheng, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Lane Graph as Path: Continuity-Preserving Path-Wise Modeling for Online Lane Graph Construction. In *Computer Vision – ECCV 2024*, pages 334–351. Springer Nature Switzerland, 2025.

[23] Yicheng Liu, Tianyuan Yuan, Yue Wang, Yilun Wang, and Hang Zhao. VectorMapNet: End-to-end vectorized HD map learning. In *International conference on machine learning*. PMLR, 2023.

[24] Xiaoyan Lu and Qihao Weng. Deep learning-based road extraction from remote sensing imagery: Progress, problems, and perspectives. *ISPRS Journal of Photogrammetry and Remote Sensing*, 228:122–140, 2025.

[25] Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, and Humphrey Shi. Towards layer-wise image vectorization. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2022.

[26] Emmanuel Maggiori, Yuliya Tarabalka, Guillaume Charpiat, and Pierre Alliez. Can semantic labeling methods generalize to any city? the Inria aerial image labeling benchmark. In *2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, pages 3226–3229, 2017.

[27] Volodymyr Mnih and Geoffrey E. Hinton. Learning to detect roads in high-resolution aerial images. In *ECCV*, 2010.

[28] Limeng Qiao, Wenjie Ding, Xi Qiu, and Chi Zhang. End-to-End Vectorized HD-map Construction with Piecewise Bézier Curve. In *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 13218–13228. IEEE, 2023.

[29] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. SAM 2: Segment Anything in Images and Videos, 2024.

[30] Pradyumna Reddy, Michael Gharbi, Michal Lukac, and Niloy J Mitra. Im2vec: Synthesizing vector graphics without vector supervision. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 7342–7351, 2021.

[31] Suprosanna Shit, Rajat Koner, Bastian Wittmann, Johannes Paetzold, Ivan Ezhov, Hongwei Li, Jiazhen Pan, Sahand Sharifzadeh, Georgios Kaissis, Volker Tresp, and Bjoern Menze. Relationformer: A Unified Framework for Image-to-Graph Generation. In *Computer Vision – ECCV 2022*, pages 422–439. Springer Nature Switzerland, 2022.

[32] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurélien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 2443–2451, 2020.

[33] Yong-Qiang Tan, Shang-Hua Gao, Xuan-Yi Li, Ming-Ming Cheng, and Bo Ren. VecRoad: Point-Based Iterative Graph Exploration for Road Graphs Extraction. In *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 8907–8915. IEEE, 2020.

[34] Zheng Tang, Milind Naphade, Ming-Yu Liu, Xiaodong Yang, Stan Birchfield, Shuo Wang, Ratnesh Kumar, David Anastasiu, and Jenq-Neng Hwang. CityFlow: A city-scale benchmark for multi-target multi-camera vehicle tracking and re-identification. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 8789–8798, 2019.

[35] Yael Vinker, Ehsan Pajouheshgar, Jessica Y. Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, and Ariel Shamir. CLIPasso: Semantically-aware object sketching. *ACM Trans. Graph.*, 41(4), 2022.

[36] Yael Vinker, Yuval Alaluf, Daniel Cohen-Or, and Ariel Shamir. CLIPascene: Scene sketching with different types and levels of abstraction. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 4146–4156, 2023.

[37] Lei Wang, Min Dai, Jianan He, and Jingwei Huang. Regularized Primitive Graph Learning for Unified Vector Mapping. In *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 16771–16780. IEEE, 2023.

[38] Shenlong Wang, Min Bai, Gellert Mattyus, Hang Chu, Wenjie Luo, Bin Yang, Justin Liang, Joel Cheverie, Sanja Fidler, and Raquel Urtasun. TorontoCity: Seeing the world with a million eyes. In *2017 IEEE International Conference on Computer Vision (ICCV)*, pages 3028–3036, 2017.

[39] Yizhi Wang and Zhouhui Lian. Deepvecfont: synthesizing high-quality vector fonts via dual-modality learning. *ACM Transactions on Graphics (TOG)*, 40(6):1–15, 2021.

[40] Yuqing Wang, Yizhi Wang, Longhui Yu, Yuesheng Zhu, and Zhouhui Lian. Deepvecfont-v2: Exploiting transformers to synthesize vector fonts with higher quality. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 18320–18328, 2023.

[41] Syed Waqas Zamir, Aditya Arora, Akshita Gupta, Salman Khan, Guolei Sun, Fahad Shahbaz Khan, Fan Zhu, Ling Shao, Gui-Song Xia, and Xiang Bai. iSAID: A large-scale dataset for instance segmentation in aerial images. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops*, pages 28–37, 2019.

[42] Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, and Ziwei Liu. Compositional generative model of unbounded 4D cities. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2025.

[43] XiMing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, and Dong Xu. DiffSketcher: Text guided vector sketch synthesis through latent diffusion models. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.

[44] Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. SVGDreamer: Text guided SVG generation with diffusion model. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4546–4555, 2024.

[45] Jiakun Xu, Bowen Xu, Gui-Song Xia, Liang Dong, and Nan Xue. Patched line segment learning for vector road mapping. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(6):6288–6296, 2024.

[46] Zhenhua Xu, Yuxuan Liu, Lu Gan, Yuxiang Sun, Xinyu Wu, Ming Liu, and Lujia Wang. RNGDet: Road Network Graph Detection by Transformer in Aerial Images. *IEEE Transactions on Geoscience and Remote Sensing*, 60:1–12, 2022.

[47] Zhenhua Xu, Yuxuan Liu, Yuxiang Sun, Ming Liu, and Lujia Wang. RNGDet++: Road Network Graph Detection by Transformer With Instance Segmentation and Multi-Scale Features Enhancement. *IEEE Robotics and Automation Letters*, 8(5):2991–2998, 2023.

[48] Pan Yin, Kaiyu Li, Xiangyong Cao, Jing Yao, Lei Liu, Xueru Bai, Feng Zhou, and Deyu Meng. Towards satellite image road graph extraction: A global-scale dataset and a novel method. In *Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)*, pages 1527–1537, 2025.

[49] Yifan Zao, Zhengxia Zou, and Zhenwei Shi. Road Graph Extraction via Transformer and Topological Representation. *IEEE Geoscience and Remote Sensing Letters*, 21:1–5, 2024.

[50] Yifan Zao, Zhengxia Zou, and Zhenwei Shi. Topology-Guided Road Graph Extraction From Remote Sensing Images. *IEEE Transactions on Geoscience and Remote Sensing*, 62:1–14, 2024.

[51] T. Y. Zhang and C. Y. Suen. A fast parallel algorithm for thinning digital patterns. *Communications of the ACM*, 27(3):236–239, 1984.