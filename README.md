
# BAP: Binding Affinity Prediction

Predict protein–ligand binding affinities using interpretable feature‐driven models (IMCP, IMC, IMCEP, IMCPiEB, IMCEP2) and deep sequence embeddings (ESM).

---

## 🔧 Requirements

- **OS:** Linux  
- **Python:** ≥ 3.7  

**Main dependencies** (install via `pip install …` or `conda install …`):

| Package         | URL                                                     |
| --------------- | ------------------------------------------------------- |
| DeepChem        | https://github.com/deepchem/deepchem                    |
| RDKit           | https://www.rdkit.org/                                  |
| NumPy           | https://numpy.org/                                      |
| pandas          | https://pandas.pydata.org/                              |
| scikit-learn    | https://scikit-learn.org/stable/                        |
| TensorFlow      | https://www.tensorflow.org/                             |
| biopandas       | http://rasbt.github.io/biopandas/                       |
| SciPy           | https://www.scipy.org/                                  |
| multiprocessing | part of Python standard library                        |




## 📦 Data Preparation

1. **Download PDBbind refined set v2020**

   * Visit [http://www.pdbbind.org.cn/](http://www.pdbbind.org.cn/) and download the “refined-set” data folder.
2. **Download CASF-2016 benchmark set**

   * Download “casf2016” from PDBbind and place it under `data/casf2016/`.
3. Merge the 2 dataset by removing the duplicate complexes. Place the dataset in folder named data in PDBbind folder
4. **Convert ligands from MOL2 to PDB**

   ```bash
   bash ./1.sh 
   ```

   > *Requires UCSF Chimera installed and on your PATH.*

---

## 🛠️ Feature Extraction

Generate and cache per-complex descriptors (“SIFT objects”):

This will pickle each complex’s descriptor object to avoid recomputation. 

Run feature.py

---

## 🎓 Usage

1. **Run feature-driven models**
   Open `notebooks/BAP.ipynb` to train and evaluate:

   * IMCP, IMC, IMCEP
   * IMCPiEB, IMCEP2

2. **Run deep‐sequence model**
   Open `notebooks/ESM_model.ipynb` to train/evaluate the ESM-based predictor.

---

## 📖 Report

See the PDF in the repository root (e.g., `CS6024_Project_Report.pdf`) for methodology, results, and discussion.



```
```
