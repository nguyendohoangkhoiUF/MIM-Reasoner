### 1. Install necessary libraries

`pip install -r requirements.txt`


### 2. Run the ISF algorithm
`cd ISF`
`python main_isf.py`

### 3. Run the KSN algorithm
`cd KSN`
`python main_ksn.py`

### 4. Run the MIM-Reasoner 
`cd RL_MIM`
`python main_rl.py`

To adjust the budget size, you can modify the following lines of code:

seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=5, type=int, help="one of: {}".format(", ".join(str(sorted(seed_rate)))))

In this code snippet, the seed_rate variable represents the available options for the budget size. You can modify the list seed_rate to include the desired budget sizes that you want to experiment with.

To configure whether you want to retrain the GCN to find a good node set, you can modify the following line of code:

parser.add_argument("-tr", "--training", default=False, type=bool, help="Training Good Nodes")

By changing the value of the default parameter from False to True, you can enable the retraining of the GCN to find a good node set. This allows the algorithm to adapt and optimize the selection of nodes based on the specific problem and budget size.
