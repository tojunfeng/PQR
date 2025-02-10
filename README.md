### Step 1: Prepare Data  

Download the required dataset from [https://github.com/beir-cellar/beir](https://github.com/beir-cellar/beir).  

### Step 2: Generate Potential Queries  

1. Deploy the Llama 8B model using SGlang or vLLM.  
2. Modify the dataset path in the source code to the dataset that was just downloaded.
3. Run the following commands:  
   ```bash
   python generate_zs.py
   python generate_mg.py
   python generate_ta.py
   ```

### Step 3: Model Evaluation  

Run the following command to evaluate the model:  
```bash
python infer.py
```