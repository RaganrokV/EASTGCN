2023.12.20
(1) Addition of ablation experiments and design of 3 variants as suggested by the reviewers 
(2) Addition of comparisons on public datasets, the public datasets used in this study were PEMS04 and PEMS08

# Variable-length traffic state prediction and applications for urban network with adaptive signal timing plan

## 1.What is variable length traffic flow？

Sample according to the phase period rather than at fixed intervals. In this case, not only the traffic flow interval of the same intersection in time is not fixed, but also the traffic flow of each intersection in space is difficult to be aligned. Therefore, the spatial and temporal information fragmentation is generated.

![image](https://github.com/RaganrokV/EASTGCN/assets/73992419/3bb58db1-8a08-45b2-b02f-7139cf3fd728)

## 2. Model structure
This paper proposes a novel predictive model called Embedding Attention Spatio-Temporal Graph Convolutional Neural Network (EASTGCN) specifically designed for PR prediction in urban networks. In the encoder part of EASTGCN, the diffusion of vehicles on the urban network is modeled as random walking, and diffusion convolution is employed to capture spatial relationships. Additionally, considering that historical traffic states are causally connected to future states, temporal causal convolution is used to capture temporal relationships. To capture the dynamic evolution of weights, an attention mechanism is adopted. The complete convolutional architecture of the encoder part ensures high computational efficiency. In the decoder part, a recurrent architecture is used to retain the sequence dependencies between outputs for multi-step prediction.

![image](https://github.com/RaganrokV/EASTGCN/assets/73992419/768db636-9fbb-432a-9992-dbb0e767a8aa)


## 3.Data section
ITSM realizes automatic collection of traffic flow and occupancy by deploying video, geomagnetic, and wide area radar detection. ITSM provides rich signal control schemes based on the collected traffic states, including coordinated control, bus priority control, and single-point adaptive control. The control scheme of the signal achieves dynamic optimization of traffic efficiency. Finally, ITSM continuously optimizes the control scheme based on the feedback received from the traffic state monitoring.

![image](https://github.com/RaganrokV/EASTGCN/assets/73992419/4e542cbe-17f5-4d17-9343-f0f9184e875a)

## 4. visualization of variable-length traffic

![image](https://github.com/RaganrokV/EASTGCN/assets/73992419/6140bf7b-e769-45c1-9dd3-a6f13cfebeb8)

## 5.Application Scenarios
### Scenarios 1: Speed inducement for electric vehicles

![image](https://github.com/RaganrokV/EASTGCN/assets/73992419/25ef00e8-5a3a-40f3-b8da-366d37a54927)
### Scenarios 2: Route planning for the connected vehicles

![image](https://github.com/RaganrokV/EASTGCN/assets/73992419/8e107a13-dc8f-4ff4-ab97-99af7134488e)

### Scenarios 3: Traffic congestion management for optimal scheduling

![image](https://github.com/RaganrokV/EASTGCN/assets/73992419/b522aca4-aa09-437b-84cb-47211382dee6)


## Experimental details and performance comparisons can be found in our original paper（ idk how to make table）, and you are welcome to cite our paper if you find it helpful.

@article{HUANG2024129566,
title = {Variable-length traffic state prediction and applications for urban network with adaptive signal timing plan},
journal = {Physica A: Statistical Mechanics and its Applications},
volume = {637},
pages = {129566},
year = {2024},
issn = {0378-4371},
doi = {https://doi.org/10.1016/j.physa.2024.129566},
url = {https://www.sciencedirect.com/science/article/pii/S0378437124000748},
author = {Hai-chao Huang and Hong-di He and Zhe Zhang and Qing-hai Ma and Xing-kuo Xue and Wen-xiu Zhang},
keywords = {Traffic states prediction, Variable-length, Deep learning, Adaptive signal timing, Application scenarios},
abstract = {In urban road networks, traffic states are characterized with complicated signal intervention and results in variable-length traffic states, especially under the adaptive signal timing condition. Due to the limitations in available real-world data and technological limitations, there have been limited investigations regarding the prediction of traffic states in road networks under adaptive signal timing. The purpose of this study is to provide a prediction method for variable-length traffic states and discuss the practical application of predictions within intelligent transportation systems. This study proposes an indicator called Phase flow Rate (PR) that integrates traffic states with signal timing information. We further introduce warping algorithm align the variable-length PR into shapes that can be accepted by the model. Finally, we propose an embedded attention spatio-temporal graph convolutional neural network (EASTGCN) for predicting PR in urban traffic systems. Base on two months of data from an urban network with an adaptive signal timing plan, experimental results demonstrate that our approach effectively addresses the challenge of variable-length data due to irregular sampling. Moreover, EASTGCN outperforms state-of-the-art models, showing an improvement in prediction performance ranging from 10.6% to 21.2%. Predicting PR rather than conventional traffic states offers distinct advantages in three application scenarios, including a 6.6% improvement in energy efficiency through speed inducement of electric vehicles, 16% travel time savings for route planning of connected vehicles, and real-time optimization for traffic congestion management.}
}


