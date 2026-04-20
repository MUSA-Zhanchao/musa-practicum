# Streetscape Analysis Using AI: Mapping Crosswalk Conditions and Pedestrian Infrastructure from Street View and Aerial Imagery

**Authors:** Zicheng Xiang, Zhanchao Yang
**Advisors:** Dr. Xiaojiang Li, Dr. Erick Guerra
**MUSA Practicum — Spring 2026**

---

## 1. Introduction

Urban planning has long treated walkability as a function of network connectivity, land use, and density, implicitly assuming that intersections are universally crossable. However, this assumption overlooks a critical dimension of pedestrian mobility: the feasibility and quality of street crossings at the micro scale. In practice, missing crosswalks, faded markings, and inadequate traffic control create discontinuities in the pedestrian network, constraining both accessibility and safety. Despite their importance, these features remain largely absent from conventional planning datasets due to the lack of scalable measurement approaches.

This study addresses this gap by asking: *how can we systematically measure crosswalk-based connectivity and crossing conditions at scale using street-level and aerial imagery?* We develop a computer vision pipeline that integrates a U-Net segmentation model with a fine-tuned YOLO detection framework to identify crosswalks, stop signs, and traffic signals. This allows us to construct a spatially explicit dataset of crossing infrastructure, including crosswalk locations, marking coverage, approach-level crossing distances, and intersection control types.

Our contribution is twofold. First, we provide a scalable method that transforms visual street features into structured planning data, enabling new forms of measurement for pedestrian infrastructure. Second, we conceptually shift the notion of connectivity from intersection-based to crossing-based — highlighting how the presence and quality of crosswalks shape the functional accessibility of urban space. While our analysis is primarily descriptive, it represents a first step toward integrating micro-scale infrastructure into planning metrics and opens the door for future work linking crossing conditions to travel behavior and safety outcomes.

The remainder of this report proceeds as follows. Section 2 motivates the research question and positions it relative to the limitations of existing pedestrian-infrastructure datasets. Section 3 reviews prior work on walkability measurement, street-view computer vision, and aerial-imagery segmentation. Section 4 presents the detection and segmentation results across fourteen Philadelphia neighborhoods. Section 5 discusses the planning implications, and Section 6 outlines the limitations of the workflow.

---

## 2. Why This Matters

Walkability is the pedestrian-facing counterpart to road-network accessibility, and widely used instruments such as Walk Score reduce it to the distance from a location to nearby amenities plus coarse network metrics like block length and intersection density. These measures implicitly treat every intersection as equally crossable. In reality, whether a pedestrian can cross — safely, legally, and comfortably — depends on micro-scale features that Walk Score and most municipal GIS layers do not capture: whether a crosswalk exists, whether its markings are legible, how wide the crossing is, and whether the approach is controlled by a stop sign, traffic signal, or nothing at all.

The absence of this information is not just an academic gap. Crossing distance is a known correlate of pedestrian exposure and injury risk, particularly for older adults and children who need more time to clear the roadway. Faded or missing markings reduce visibility to drivers and erode yielding behavior. Yet in Philadelphia — as in most U.S. cities — there is no routinely maintained citywide dataset of curb-to-curb crossing widths, marking condition, or intersection-level traffic control. The information that does exist comes from two incompatible sources: coarse, infrequently updated municipal GIS layers, and expensive engineering surveys that are only commissioned for specific capital projects.

The research question this study takes up is therefore practical as much as conceptual: **can computer vision applied to freely available aerial and street-view imagery produce intersection-level pedestrian infrastructure data that is detailed enough to support planning decisions, yet cheap and repeatable enough to cover an entire city?** If yes, it offers a screening layer that sits between outdated GIS and costly field survey — a middle tier that OTIS and similar agencies currently lack. This matters for Vision Zero prioritization, ADA transition planning, and any initiative that needs to know *where* pedestrian infrastructure is failing before committing engineering resources to measure it.

---

## 3. Literature Review

Three strands of prior work converge on this project.

**Walkability measurement.** Early walkability indices (Frank et al., 2006; Glazier et al., 2014) combined intersection density, residential density, and land use mix into composite scores, and commercial products such as Walk Score operationalized the distance-to-amenities view that dominates in practice. A more recent literature has argued that these network-and-density formulations miss the experiential quality of the pedestrian environment — sidewalk continuity, crossing treatments, pedestrian-scale detail — and have pushed toward audit-based instruments such as PEDS and MAPS (Clifton et al., 2007; Cain et al., 2014). Audits capture what network measures cannot, but they are labor-intensive and rarely deployed at citywide scale. Our work is a response to exactly this tension: the audit literature tells us *what* to measure; the computer vision literature offers the tools to do so at scale.

**Street-view computer vision for the urban environment.** Google Street View and equivalent panoramic imagery have become a standard substrate for urban analytics since the Streetscore work of Naik et al. (2014) and the Treepedia project at MIT (Li et al., 2015). Street-view-based studies have quantified greenery, sky-view factor, perceived safety, and building facades. More recently, object detection models — including the YOLO family — have been applied to identify traffic signs, signals, and safety hardware (Campbell et al., 2019; Biljecki & Ito, 2021). Our detection component builds directly on this trajectory, but we focus on traffic-control features specifically because they mediate *whether* a crossing exists as a functional pedestrian facility, not just a legal one.

**Semantic segmentation of aerial imagery.** Convolutional networks, and U-Net in particular (Ronneberger et al., 2015), have become the default tool for extracting linear and polygonal features from remote-sensing imagery — roads, buildings, parcels, and increasingly, road-surface markings. A small but growing body of work has applied segmentation to crosswalk extraction from satellite or drone imagery (Berriel et al., 2017; Hosseini et al., 2022), typically reporting strong pixel-level accuracy on held-out tiles but limited generalization across cities or camera geometries. Our segmentation component follows this recipe — U-Net with a ResNet encoder — but contributes a deployment test: we train on a single pilot area (University City) and then apply the model to large mosaics covering most of Center Philadelphia to assess whether the learned representation transfers within a single imagery program.

The novelty of this project is not in any single model but in the combination: pairing aerial segmentation (which gives geometry — where the crosswalk is and how wide the approach is) with street-view detection (which gives context — what kind of control the crossing has), and binding both to the street-network intersection graph so the outputs are usable by planners.

---

## 4. Results

We report results in two parts, corresponding to the two modeling components of the pipeline. YOLO-based traffic-control detection was run over fourteen Philadelphia neighborhoods and districts; U-Net crosswalk segmentation was trained on a University City pilot and deployed to three larger aerial mosaics covering most of the study area.

### 4.1 Traffic-Control Detection (YOLO)

The YOLOv8-medium detector was applied to Google Street View imagery sampled at four compass-aligned arms per intersection, with a detection confidence threshold of 0.5. Before deployment, we benchmarked the stock COCO-trained model on the COCO val2017 set to establish a class-specific reference point. The relevant classes behave differently: stop signs are detected with high precision and recall, while traffic lights are harder, with recall below 0.5.

| Class | Precision | Recall | F1 | AP@50 | mAP@50–95 |
|---|---|---|---|---|---|
| traffic light | 0.725 | 0.483 | 0.580 | 0.569 | 0.322 |
| stop sign | 0.818 | 0.720 | 0.766 | 0.795 | 0.715 |

*Source: [`model_benchmark_coco.csv`](result-yolo/model_benchmark_coco.csv).*

These benchmark numbers set expectations for the deployment results: traffic-light counts should be interpreted as a lower bound, while stop-sign counts are closer to the true signal, subject to the usual caveats of domain shift from COCO to Philadelphia street view.

Across the fourteen study areas we processed **9,950 street-view images** and produced **20,037 total detections** (17,261 traffic lights; 2,776 stop signs over 9,950 images and 844 images respectively). Detection rates vary sharply and in a way that is consistent with the underlying streetscape. University City, Rittenhouse, Logan Square, and Center City — the most signalized, highest-traffic districts — have traffic-light detection rates between 20% and 35% of images sampled. Kensington, West Kensington, Fishtown, and Point Breeze — rowhouse neighborhoods whose intersection control is dominated by stop signs — invert that pattern, with stop-sign detection rates between 10% and 14%.

| Neighborhood | Images | TL rate | TL detections | SS rate | SS detections |
|---|---:|---:|---:|---:|---:|
| CALLOWHILL | 288 | 18.4% | 124 | 6.6% | 19 |
| Center City | 275 | 27.6% | 141 | 4.4% | 12 |
| CHINATOWN | 112 | 26.8% | 62 | 1.8% | 3 |
| EAST_KENSINGTON | 475 | 10.5% | 97 | 12.2% | 65 |
| FISHTOWN | 1,631 | 4.8% | 161 | 11.2% | 191 |
| KENSINGTON | 444 | 9.9% | 72 | 11.9% | 56 |
| LOGAN_SQUARE | 752 | 26.2% | 389 | 5.3% | 43 |
| OLD_CITY | 660 | 17.7% | 237 | 3.6% | 28 |
| POINT_BREEZE | 1,152 | 5.0% | 103 | 10.6% | 137 |
| RITTENHOUSE | 1,068 | 20.3% | 406 | 4.3% | 47 |
| SOCIETY_HILL | 524 | 12.6% | 116 | 4.2% | 23 |
| UNIVERSITY_CITY | 632 | 34.7% | 518 | 3.6% | 26 |
| WASHINGTON_SQUARE_WEST | 832 | 16.7% | 228 | 3.0% | 29 |
| WEST_KENSINGTON | 1,105 | 12.3% | 247 | 13.8% | 165 |

*Source: [`detection_metrics.csv`](result-yolo/detection_metrics.csv), [`detection_summary_all.csv`](result-yolo/detection_summary_all.csv).*

Average detection confidence was stable across neighborhoods (0.65–0.71 for traffic lights, 0.61–0.80 for stop signs), suggesting that between-neighborhood differences reflect real variation in installed infrastructure rather than systematic detector drift. Per-neighborhood geo-referenced point layers are written to [`result-yolo/<NEIGHBORHOOD>/outputs/`](result-yolo) as `traffic_lights.geojson` and `stop_signs.geojson`, along with a `detected_intersections.geojson` summary.

### 4.2 Crosswalk Segmentation (U-Net)

The crosswalk segmentation model is a U-Net with a ResNet-34 encoder, trained on 256×256 patches derived from PASDA 2024 aerial imagery over a University City pilot scene. We manually labeled 202 crosswalk polygons, rasterized them into a per-pixel binary mask, generated 934 image/mask patch pairs, and partitioned them into 653 training, 140 validation, and 141 test patches using stratified sampling on crosswalk coverage.

Training ran for 25 epochs with binary cross-entropy plus Dice loss; the best validation Dice of **0.919** occurred at epoch 17 and was used as the deployed checkpoint. A post-hoc threshold sweep on the validation split identified **τ = 0.40** as the F1-optimal decision threshold, with the following pixel-level test metrics:

| Threshold | Precision | Recall | F1 (Dice) | IoU |
|---|---:|---:|---:|---:|
| 0.40 | 0.924 | 0.955 | **0.939** | 0.886 |
| 0.45 | 0.928 | 0.951 | 0.939 | 0.886 |
| 0.50 | 0.932 | 0.946 | 0.939 | 0.885 |

*Source: [`models/threshold_search.csv`](models/threshold_search.csv), [`models/training_history.csv`](models/training_history.csv).*

On a held-out twelve-patch visualization sample, mean Dice was 0.816 and mean IoU was 0.763. The gap between patch-level mean and aggregated Dice reflects the fact that most error is concentrated in a small number of hard patches — partially occluded crossings, novel marking styles (continental vs. ladder), and patches containing only a sliver of crosswalk.

The deployed model was then applied to three large mosaics covering the middle, southern, and upper-middle portions of the study area. Sliding-window inference at 256-pixel patch size and 128-pixel stride produced pixel-level probability rasters, which were thresholded at 0.45, cleaned by removing connected components smaller than 200 pixels, closed morphologically, and vectorized to polygons.

| Mosaic | Raster size (px) | Windows processed | Positive pixels (cleaned) | Crosswalk polygons |
|---|---|---:|---:|---:|
| middle | 42,240 × 31,680 | 81,263 | 6,914,860 | **1,495** |
| south | 73,920 × 21,120 | 94,628 | 5,744,029 | **1,456** |
| upper-middle | 73,920 × 10,560 | 47,314 | 3,357,686 | **659** |

Across the three deployment scenes the model produced **3,610 crosswalk polygons** — a first-pass citywide(-scale) crosswalk inventory where none previously existed at this resolution. These are written as `<scene>_crosswalks.gpkg` / `.shp`, with per-polygon area in square feet, suitable for direct use in ArcGIS or QGIS.

The full segmentation pipeline — from label rasterization through training, thresholding, full-scene prediction, and polygonization — is documented in [`results-segementation/python-segementation.ipynb`](results-segementation/python-segementation.ipynb), with the citywide deployment script in [`results-segementation/deploy.ipynb`](results-segementation/deploy.ipynb).

### 4.3 Integrated Outputs

Binding the two model outputs to the OSM intersection graph produces the integrated products targeted by the practicum:

1. **Crossings layer** — intersection-approach records with estimated curb-to-curb crossing distance and approach-level crosswalk marking coverage.
2. **Crosswalks layer** — polygon geometries of detected crosswalks with area attributes.
3. **Traffic control points** — stop sign and traffic signal locations as GeoJSON point layers per neighborhood.

These feed the interactive web application described in [`final-visulization.md`](final-visulization.md), which allows users to click any intersection to inspect estimated crossing width, crosswalk presence, and nearby traffic controls, and to launch Google Street View for visual verification.

---

## 5. Discussion

The results support the core claim of the project: a combined aerial-segmentation plus street-view-detection pipeline can produce intersection-level pedestrian infrastructure data at a spatial grain that existing municipal GIS layers do not reach, at a cost and cadence that field engineering surveys cannot match. Three findings are worth drawing out.

**Detection patterns recover known streetscape structure.** The inversion between traffic-light-dominated districts (Center City, Rittenhouse, University City, Logan Square) and stop-sign-dominated neighborhoods (Kensington, Point Breeze, Fishtown, West Kensington) is not something we imposed — it emerges directly from the detector output. That the model reproduces this pattern without any neighborhood-level supervision is evidence that the outputs are informative about real infrastructure, not artifacts of image availability or detector bias. For planners, the practical implication is that the same workflow can be used to map *where a given control type dominates* — useful for pedestrian-signal retiming studies, stop-sign compliance audits, or identifying uncontrolled intersections for midblock-crossing studies.

**Crosswalk segmentation generalizes within a single imagery program.** The pixel-level F1 of 0.94 on the University City test set is strong, and the fact that applying the same model to three much larger mosaics (covering roughly 10× the training area) yielded coherent polygon inventories — 3,610 polygons with no manual correction — indicates that within a single aerial imagery program (PASDA 2024, ~15 cm/pixel, consistent lighting and capture season), transfer is practical. This is the right unit of transfer to reason about: annual PASDA captures mean that the same pipeline can in principle be re-run each year to track marking degradation and new installations without retraining.

**The combination is more than the sum of parts.** Aerial segmentation tells you that a crosswalk exists and gives you its geometry; it cannot tell you whether the approach is controlled. Street-view detection tells you the control type but cannot reliably give you the crossing width. Binding both to the intersection graph produces a per-approach record — `(intersection_id, approach_direction, crossing_width, crosswalk_coverage, control_type)` — that matches the unit of analysis used in pedestrian-safety research. This is the planning-relevant product; each individual model output is a means to it.

For OTIS and similar agencies, the immediate use case is screening. Rather than asking "where should we send a survey crew?", an analyst can filter the crossings layer for long crossing distances with low marking coverage and no signal control — a plausible proxy for high-risk uncontrolled crossings — and prioritize field inspection accordingly. The workflow does not replace engineering measurement; it tells you where engineering measurement is most likely to be worth the expense.

---

## 6. Limitations

Several limitations qualify these results.

**Training data scale and geographic scope.** The U-Net was trained on 202 manually labeled crosswalks from a single pilot area (University City). Although the deployment scenes are drawn from the same imagery program, neighborhoods with marking styles absent from the training set — e.g., artistic or asphalt-art crosswalks, heavily faded continental bars, or unusual ladder spacings — are under-represented and are likely where the model's residual errors concentrate. The twelve-patch visualization audit showed one sample with Dice of zero, confirming that failure cases exist; a more systematic error audit across the deployment scenes is a clear next step.

**No independent ground-truth evaluation at deployment scale.** The 0.94 Dice reported in Section 4.2 is a held-out *patch-level* metric within the University City pilot. We have not produced an independent field-validated sample for the larger deployment mosaics, so the pixel-level performance on those scenes is an extrapolation. Similarly, the YOLO detection numbers are reported on COCO val2017, not on a Philadelphia-specific hand-labeled evaluation set. Both models would benefit from a small, location-stratified validation audit before any planning decision is made on the outputs.

**Detection is biased by imagery availability.** Google Street View coverage is uneven in both space and time — some arms have 2019 imagery, others 2023 — and panorama capture dates are mixed across our fourteen neighborhoods. A stop sign installed in 2022 will appear or not depending on which pano Google served. For a monitoring application this is a hard constraint: the pipeline is only as current as the underlying imagery.

**Traffic-light recall is substantially lower than stop-sign recall.** The COCO benchmark shows traffic-light recall of 0.48 compared to 0.72 for stop signs, and field experience suggests traffic-light detection is degraded further by viewing geometry (signals mounted overhead or far from the approach) and by confusion with other illuminated street furniture. Deployment counts for traffic lights should be treated as a lower bound rather than a census.

**Crossing distance estimation is a geometric proxy, not a measurement.** The perpendicular-transect method in the baseline scripts estimates curb-to-curb distance from road-surface segmentation masks and the intersection approach graph. It does not account for corner curves, bulb-outs, refuge islands, or diagonal crossings, all of which matter for actual pedestrian exposure. Reported widths should be interpreted as first-order approximations; anything within a factor of approach lane-count of the true value is what the method can currently deliver.

**The workflow is descriptive, not causal.** We do not claim that any of the infrastructure features mapped here *cause* differences in pedestrian safety, travel behavior, or accessibility outcomes. Linking the dataset to crash records, travel-survey data, or accessibility scores is a natural next step but is outside the scope of this practicum.

None of these limitations undermine the core result — that it is feasible to produce intersection-level pedestrian infrastructure data at citywide scale from freely available imagery — but each defines a boundary on how the outputs should be used. The honest framing is the one given in the README: this is a screening and prioritization tool, not a measurement product.

---

## References (selected)

- Berriel, R.F., Rossi, F.S., de Souza, A.F., Oliveira-Santos, T. (2017). Automatic large-scale data acquisition via crowdsourcing for crosswalk classification. *Computers & Graphics*.
- Biljecki, F., Ito, K. (2021). Street view imagery in urban analytics and GIS: A review. *Landscape and Urban Planning*.
- Cain, K.L., Millstein, R.A., Sallis, J.F., et al. (2014). Contribution of streetscape audits to explanation of physical activity in four age groups based on the MAPS. *Social Science & Medicine*.
- Campbell, A., Both, A., Sun, Q. (2019). Detecting and mapping traffic signs from Google Street View images. *Computers, Environment and Urban Systems*.
- Clifton, K.J., Livi Smith, A.D., Rodriguez, D. (2007). The development and testing of an audit for the pedestrian environment. *Landscape and Urban Planning*.
- Frank, L.D., Sallis, J.F., Conway, T.L., et al. (2006). Many pathways from land use to health. *JAPA*.
- Glazier, R.H., Creatore, M.I., Weyman, J.T., et al. (2014). Density, destinations or both? A comparison of measures of walkability. *PLOS ONE*.
- Hosseini, M., Sevtsuk, A., Miranda, F., et al. (2022). Mapping the walk: A scalable computer vision approach for generating sidewalk network datasets from aerial imagery. *Computers, Environment and Urban Systems*.
- Li, X., Zhang, C., Li, W., et al. (2015). Assessing street-level urban greenery using Google Street View and a modified green view index. *Urban Forestry & Urban Greening*.
- Naik, N., Philipoom, J., Raskar, R., Hidalgo, C. (2014). Streetscore — Predicting the perceived safety of one million streetscapes. *CVPR Workshops*.
- Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
