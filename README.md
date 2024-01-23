# MIT_masters_thesis

Welcome to my master's thesis repository! This repository contains the source code for my MIT-WHOI Master's of Science thesis, titled: Energetics and Similarity Theory in the Wave-Affected Boundary Layer.

## Code Structure

You may store your raw data in a folder of your choice. Then to run the code, add in the folder path of the raw data where applicable. The output of the code that works with the raw data (masters_level00_code_pipeline.py) will then output files into a sub-folder within a master folder called "code_pipeline"; you will need to create the master folder and the subfolders within it, as outlined below: 
### code_pipeline
#### Level1_errorLinesRemoved
#### Level1_align-interp
#### Level2

## Getting Started

To get started with the project, Run the following code in this order:

1. masters_level00_code_pipeline.py
2. masters_level0_p1.py through masters_level0_p7.py
3. masters_despikeSonicOutputs.py
4. masters_jd_date_conversion.py
5. masters_determineBadWindDirs.py
6. masters_ctd_Z_conversion.py
7. masters_metAverages.py
8. masters_paros_avg_code.py
9. masters_rhoCalculation.py
10. masters_virtualPotentialTemp_conversion.py
11. masters_makePW_fileMatchSelectedDates.py
12. masters_productionTerm.py
13. masters_productionTerm.py
14. masters_dissipationTermU.py
15. masters_dissipationTermW.py
16. masters_eps_comparison_PuuPww
17. masters_uStar_calcs.py
18. masters_ZoverL_calcsAndPlots.py


Then, you can explore the other files, particularly the masters_UniversalFunctions.py and masters_PversusE_PW.py files.

At the beginning of each code, there is a short description of the code purpose as well as input files/location and output files/locations that correspond to the code_pipeline folder, but in which you will need to rename to your exact filepath.


