In this respository, codes for analyzing longitudinal peripheral blood mononuclear cell (PBMC) transcriptomic profiles from corneal transplant recipients are provided. These codes implement nine featrue ranking algorithms: last absolute shrinkage and selection operator (LASSO), ridge regression,  random forest (RF), Adaptive Boosting (AdaBoost), Categorical Boosting (CATBoost), Extremely Randomized Trees (ExtraTrees), light gradient boosting machine (LightGBM),  eXtreme Gradient Boosting (XGBoost), and SelectKBest (SKB), and two classification algiorithms: decision tree (DT) and random forest (RF).
Python environment installation

Step 1 Download the software by git clone https://github.com/*******/*******

Step 2 Installation has been tested in Linux and Windows with Python 3 (Recommend to use Python 3.10 or miniconda 3 platform). Since the package is written in Python 3, Python 3 with the pip tool must be installed first. The tools use the following dependencies: numpy, scipy, pandas, scikit-learn You can install these packages first, by the following commands: pip install pandas pip install numpy pip install scipy pip install scikit-learn

Step 3 cd to the software folder and run the installation command: python *************

Guide for codes 

Step 1: The transcriptomic profiles from corneal transplant recipients was fed into the codes of one of the feature ranking algorithm, which can produce a feature list. 

Step 2: With the feature list in Step 1, using codes in folder csv_make to generate csv files, which contain samples represented by some top features in the list. 

Step 3: For each generated csv file, using the codes of one classification algorithm to obtain the cross-validation results.

Other notes

In each folder, the file "comd" contains the codes to run the corresponding package.
