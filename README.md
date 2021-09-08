# Open-world metrics calculation for nuScenes

We modify the official SemanticKITTI api to calculate the closed-set mIoU and open-set metrics
including AURP, AUROC, and FPR95 in this repository.

To use this repository for calculating metrics, the closed-set prediction labels and uncertainty scores
for each points in the dataset should be generated and saved as:

### nuScenes
```
./
├── 
├── ...
└── lidarseg/
    ├──v1.0-trainval/
    ├──v1.0-mini/
    ├──v1.0-test/
    ├──nuscenes_infos_train.pkl
    ├──nuscenes_infos_val.pkl
    ├──nuscenes_infos_test.pkl
└── predictions/        
    ├── closed-set_prediction_results/	
        ├── 000000.label
        ├── 000001.label
        └── ...
    └── uncertainty_scores/ 
        ├── 000000.score
        ├── 000001.score
        └── ...
```
## Evaluation
### nuScenes

- Run the evaluation script to obtain the closed-set mIoU and open-set metrics including AUPR,
AUROC, and FPR95. The path is determined in the `evalute_semantics.sh` and line 164, 174 of `evaluate_semantics.py`.
```
./evaluate_semantics.sh
```
- The result is shown like:
```
********************************************************************************                          
INTERFACE:                                                                                                
Data:  /harddisk/nuScenes                                                                          
Predictions:  /harddisk/nuScenes/predictions                                                       
Backend:  numpy                                                                                           
Split:  valid                                                                                             
Config:  config/nuscenes.yaml                                                                             
Limit:  None                                                                                              
Codalab:  None                                                                                            
********************************************************************************                          
Opening data config file config/nuscenes.yaml                                                             
Ignoring xentropy class  0  in IoU evaluation                                                             
[IOU EVAL] IGNORE:  [0]                                                                                   
[IOU EVAL] INCLUDE:  [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]                                    
======       
Loading NuScenes tables for version v1.0-trainval...
Loading nuScenes-lidarseg...
Loading nuScenes-panoptic...
32 category,
8 attribute,
4 visibility,
64386 instance,
12 sensor,
10200 calibrated_sensor,
2631083 ego_pose,
68 log,
850 scene,
34149 sample,
2631083 sample_data,
1166187 sample_annotation,
4 map,
34149 lidarseg,
34149 panoptic,
Done loading in 39.290 seconds.
======
Reverse indexing ...
Done reverse indexing in 10.0 seconds.
======
labels:  6019
predictions:  6019
Evaluating sequences: 10% 20% 30% 40% 50% 60% 70% 80% 90% AUPR is:  0.21182361584888953
AUROC is:  0.8448526947283441
FPR95 is:  0.7141644584917942
Validation set:
Acc avg 0.923
IoU avg 0.568
IoU class 1 [barrier] = 0.000
IoU class 2 [bicycle] = 0.282
IoU class 3 [bus] = 0.921
IoU class 4 [car] = 0.854
IoU class 5 [construction_vehicle] = 0.000
IoU class 6 [motorcycle] = 0.689
IoU class 7 [pedestrian] = 0.755
IoU class 8 [traffic_cone] = 0.000
IoU class 9 [trailer] = 0.000
IoU class 10 [truck] = 0.752
IoU class 11 [driveable_surface] = 0.959
IoU class 12 [other_flat] = 0.677
IoU class 13 [sidewalk] = 0.735
IoU class 14 [terrain] = 0.740
IoU class 15 [manmade] = 0.865
IoU class 16 [vegetation] = 0.862
********************************************************************************
below can be copied straight for paper table
0.000,0.282,0.921,0.854,0.000,0.689,0.755,0.000,0.000,0.752,0.959,0.677,0.735,0.740,0.865,0.862,0.568,0.923


```