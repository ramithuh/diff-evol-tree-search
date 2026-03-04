import subprocess
import os
import math
        
print("Which GPU do you want to restrict to?")
id = input()
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)

# Define the list of seeds you want to use
seeds = [x+42 for x in range(10)]

# Define the leaf complexity you want to use
print("which leaf complexity do you want to use?")
print("start value: ?")
s = int(input())
print("end value: ?")
e = int(input())

leaves = [2**x for x in range(int(math.log2(s)), int(math.log2(e))+1)]
print("ok will run for the following leaf complexities:")
print(leaves)

# Common command and arguments
base = ["python", "train_small_parsimony.py", "-nl", "20", "-m", "50", "-sl", "256", "-tLs", "[0,0.01,100,3]", "-lr", "0.1","-lr_seq", "0.01", "-n", "Small Parsimony Run", "-ft", "-t", "small_parsimony", "-p","Batch-Run-Maximum-Parsimony-A100-META-optimized", "-e", "5000", "-ic", "10"]
base = base + ["-g", "0"]

for l in leaves:
    cmd = base + ["-l", str(l)]
    
    print(f"Running for {l} leaves")
    
    for seed in seeds:
        # Add the seed argument to the command
        cmd_with_seed = cmd + ["-s", str(seed)]

        # Run the command with the modified seed
        print(f"-> Running for seed {seed}: ", end =", ")
        
        #Use Popen instead of run and store the process object
        process = subprocess.Popen(cmd_with_seed, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        #Print the PID
        print(f"PID: {process.pid}")
        print(f"{process.stderr.read()}")
        
        process.wait()
