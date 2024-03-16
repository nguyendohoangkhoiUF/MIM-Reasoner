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
