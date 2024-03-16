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

If you want to change budget size, you can change in here:
`seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=5, type=int, help="one of: {}".format(", ".join(str(sorted(seed_rate)))))`
