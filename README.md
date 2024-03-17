### MIM-Reasoner: Learning with Theoretical Guarantees for Multiplex Influence Maximization (AISTATS 2024)

Nguyen Do, Tanmoy Chowdhury, Chen Ling, Liang Zhao, My T. Thai

Multiplex influence maximization (MIM) asks us to identify a set of seed users such as to maximize the expected number of influenced users in a multiplex network. MIM has been one of central research topics, especially in nowadays social networking landscape where users participate in multiple online social networks (OSNs) and their influences can propagate among several OSNs simultaneously. Although there exist a couple combinatorial algorithms to MIM, learning-based solutions have been desired due to its generalization ability to heterogeneous networks and their diversified propagation characteristics. In this paper, we introduce MIM-Reasoner, coupling reinforcement learning with probabilistic graphical model, which effectively captures the complex propagation process within and between layers of a given multiplex network, thereby tackling the most challenging problem in MIM. We establish a theoretical guarantee for MIM-Reasoner as well as conduct extensive analyses on both synthetic and real-world datasets to validate our MIM-Reasoner's performance.

Link paper: https://arxiv.org/abs/2402.16898 

Published in International Conference on Artificial Intelligence and Statistics (AISTATS) 2024

`@article{do2024mimreasoner,
  title   = {MIM-Reasoner: Learning with Theoretical Guarantees for Multiplex Influence Maximization},
  author  = {Nguyen Do and Tanmoy Chowdhury and Chen Ling and Liang Zhao and My T. Thai},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2402.16898}
}`

Please CITE our paper whenever our model implementation is used to help produce published results or incorporated into other software.

### 1. Install necessary libraries

`pip install -r requirements.txt`


### 2. Run the ISF algorithm
`cd ISF`
`python main_isf.py`

### 3. Run the KSN algorithm
`cd KSN`
`python main_ksn.py`

### 3. Creating Synthetic Multiplex Network
To create a synthetic multiplex network, you can navigate to the "createData.py" file in the "Synthetic_Dataset" directory and follow the instructions provided in the command line. If you wish to change the percentage of overlapping users in the synthetic network, you can modify the following code:

`parser.add_argument("-o", "--overlapping_user", default=0.3, type=float, help="one of: {}".format(", ".join(str(sorted(overlap)))))`

In this code snippet, the "overlapping_user" variable represents the default percentage of overlapping users, which is set to 0.3. You can change this value to your desired percentage by modifying the default parameter.

### 4. Real Multiplex Network

There are some real datasets available in the "Real_Dataset" folder. The MIM-Reasoner is currently being tested on Xenopus. You can find other multiplex network datasets by visiting the following link: https://manliodedomenico.com/data.php. To download a dataset, you can access the link provided and reformat it to match our desired format, such as an adjacency matrix. Then the code should work correctly.


### 4. Run the MIM-Reasoner 
`cd RL_MIM`
`python main_rl.py`

** To adjust the budget size, you can modify the following lines of code:

`seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=5, type=int, help="one of: {}".format(", ".join(str(sorted(seed_rate)))))`

In this code snippet, the seed_rate variable represents the available options for the budget size. You can modify the list seed_rate to include the desired budget sizes that you want to experiment with.

** To configure whether you want to retrain the GCN to find a good node set, you can modify the following line of code:

`parser.add_argument("-tr", "--training", default=False, type=bool, help="Training Good Nodes")`

By changing the value of the default parameter from False to True, you can enable the retraining of the GCN to find a good node-set. This allows the algorithm to adapt and optimize the selection of nodes based on the specific problem and budget size.

** You can also modify the propagation model type by adjusting the hyperparameter in the following code:

`diffusion = ['IC', 'LT', 'SIS']
parser.add_argument("-dm", "--diffusion_model", default="LT", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))`
                    
In the code snippet above, the diffusion_model parameter can be set to one of the following options: "IC" (Independent Cascade), "LT" (Linear Threshold), or "SIS" (Susceptible-Infected-Susceptible). You can change the default value to the desired diffusion model type or provide it as a command-line argument when running the code.

### Where are the results? 

After running each algorithm with the same dataset, you can find the results in the "Output" folder. For example, if you used the dataset "Xenopus_mean_LT50" with the Linear Threshold model, you would obtain the result in the file "Output/Xenopus_mean_LT50.csv".

The result file would have the following format:

`algorithm,Nodes,Edges,Layer,Budget,Diffusion,Seed set,Spread,Time
ISF,461,584,5,5,LT,"[329, 1, 61, 378, 86, 435, 25, 26, 85, 139, 158, 3, 246, 36, 79, 322, 451, 105, 203, 208, 268, 317, 95]",180.0,7.30393123626709
KSN,461,584,5,5,LT,"[1, 3, 1, 97, 329, 1, 61, 378, 86, 435, 25, 26, 85, 139, 158, 36, 79, 322, 1, 86, 26, 25, 139]",145.0,13.688209533691406
RL_MIM,461,584,5,5,LT,"[1, 3, 97, 63, 329, 378, 61, 86, 85, 26, 435, 139, 25, 158, 36, 79, 451, 322, 203, 208, 105, 365, 34]",176.0,54.20702815055847`

In the result file, each row represents the result of a specific algorithm run. The columns provide details such as the algorithm name, number of nodes, number of edges, layer information, budget, diffusion model, seed set used, spread achieved, and the execution time.
 

