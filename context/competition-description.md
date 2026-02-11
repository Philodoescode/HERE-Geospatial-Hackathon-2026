# HERE Cairo Hackathon 2026

**Features Extraction Using GIS Data!**
Hosted by Geospatial Community of Practice

## The Challenge – Geometry hackathon Cairo

**Centerline creation from wide available sources (VPD, PROBE)**

### Sources – VPD and Probe

#### VPD (Vehicle Path Data)

* VPD provides localized road-level observations that are typically more precise than probe GPS data but limited to
  where sensor-equipped vehicles have driven.
* Unlike probe data, VPD is **vehicle-only** and more structured.
* Delivered as **WKT LINESTRING**.
* Coordinates in **WGS84 (EPSG:4326)**.
* We use only **Fused True** drives (If True, path contains fused path. Else, it contains raw GPS path).

#### Attributes

* Date, Hour
* Construction Percent
* Altitudes
* Crosswalk types
* Traffic signal count
* Direction of travel (implicit on drive direction)

#### Geometry Format

* Delivered as **WKT LINESTRING**
* Coordinates in **WGS84**

| path                                                                 | altitudes                |
|:---------------------------------------------------------------------|:-------------------------|
| LINESTRING (150.78007542377264 -33.73787582125322, 150.7800561600... | [ 59.38303213752806,...  |
| LINESTRING (55.331326034295174 25.227238038699667, 55.33135079199... | [ -31.184026181697845... |
| LINESTRING (-89.59045854004647 34.3578389607912, -89.59045827488...  | [ 87.3300636895001, ...  |

---

### VPD Visuals

* **Fused True and False:** Shows raw, noisy path data.
* **Fused True only:** Shows cleaner, more aligned road path observations.
* **Examples:** Potential parking lot, well-defined roads with adjacent single drives, roads at different levels.

---

### Probe

* Probe data represents location traces collected from moving entities, such as vehicles, mobile devices, or other
  connected sensors. Probes are not limited to cars and may include mixed travel modes.
* Probe data is typically:
    * Noisy (GPS drift, variable accuracy)
    * Sparse or unevenly sampled
    * Collected at different speeds and sampling rates
* Each probe trace approximates a path taken through the road network but does not directly represent a road centerline.
* Probe is nevertheless widely available.

#### Attributes

* Timestamp
* Latitude / Longitude
* Speed (optional)
* Heading / bearing (optional)
* Probe source or type (optional)

#### Geometry Format

* Delivered as **WKT LINESTRING** (individual traces) or **MULTILINESTRING** (grouped traces)
* Coordinates in **WGS84**

---

### Problem 1 – Centerline Generation from Probe & VPD Data

**Goal:** Design an algorithm or pipeline to convert raw probe traces and/or VPD detections into smooth, continuous, and
topologically correct road centerlines suitable for ingestion into a road network.

#### Key focus areas:

* Handling sparse, noisy, and heterogeneous inputs
* Clustering and alignment of output
* Generating smooth and connected centerlines
* Preserving intersections and road continuity

#### Key Deliverables:

* Algorithm or model description
* Generated centerline examples
* Quality evaluation metrics

---

### Problem 2 – Road Geometry Filtering & Classification

**Goal:** Build a system to classify detected road geometries and decide whether they should be ingested into the road
network.

#### Key focus areas:

* Use of probe and VPD signals (density, directionality, temporal consistency)
* Distinguishing navigable roads from low-value or non-navigable\routable paths
* Handling ambiguous cases and minimizing false positives

#### Key Deliverables:

* Classification approach* or model description (ML-based, rule-based,..)
* Definition of input features and class labels
* Classification results with performance metrics
* Error analysis and examples of misclassified geometries

***Example of possible classes:**

* Public navigable roads
* Restricted or residential roads
* Parking entrances, service roads

---

## Judging Criteria

| Criterion                    | Question                                                                                  |
|:-----------------------------|:------------------------------------------------------------------------------------------|
| **Innovation**               | Is the solution unique and creative?                                                      |
| **Technical Implementation** | How effectively does the solution use HERE tech & relevant tech (e.g., databases, AI/ML)? |
| **Functionality**            | Does it work as intended?                                                                 |
| **Global Scalability**       | Can the solution be implemented across different countries?                               |
| **Impact**                   | Is it feasible to implement?                                                              |
| **Presentation**             | Is the solution clearly explained?                                                        |