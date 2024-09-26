## What is FG-LAED?
FG-LAED (Feature Generator based on Local Atomic Environment Descriptor) is a Python-based toolkit designed to calculate atomic center symmetric function values by reading CIF structure files and automatically extracting inherent properties of coordinating elements. This toolkit facilitates customized feature generation for various types of catalysts, enabling automated catalyst design across a wide range of materials while effectively reducing costs.
## Requirements
```
ase (atomic simulation environment)
openpyxl (library for reading and writing Excel files)
shutil (library for file operations)
numpy (library for numerical computing)
pandas (library for data manipulation and analysis)
re (regular expressions library)
sys (system-specific parameters and functions)
utils.cif2input (custom utility for converting CIF files)
utils.ACSF (custom utility for Atomic Cluster Symmetry Function)
```

## Example
Extract features from the data in `input`， as shown in the example below:
```python
    form run import run
    configpath = r'template/config1.data'
    inputpath = r'input/'
    savepath = r'temp/out'
    def main():
    split_and_save_files('temp/out', output_folder)
    copy_and_rename_files(output_folder)    
    # Define electronegativity values
    element_values = {
        'H': 2.20, 'He': None,
        'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        # Add other elements as needed
    }

    process_data_files(output_folder, element_values)

```
	
    After the execution is complete, an out folder will be automatically created within the temp directory. This out folder contains feature table files, and the Sheet3 in the table corresponds to the extracted features for each catalyst. 
	
## Parameter introduction

```
with_energy_and_forces: specifies whether to include energy and forces in the output
configpath: path to the configuration file
calculate_gfun: calculates symmetry functions based on the configuration
sysatoms: atoms object representing the system
inputpath: path to the input data file
savepath: output file path for the results
get_gfunction: reads structure and computes symmetry functions, saving results to a file
process_files：processes all relevant input files and generates corresponding output files
find_first_column_of_Uiso: extracts the first column related to Uiso values for elements
process_cif_files: reads an Excel file and matches elements with Uiso data in CIF files
copy_data_files: copies specified lines from data files to new output files
file_idx_dict: dictionary of file names and their corresponding indices to copy
create_excel_from_data_files: combines data into an Excel format for easier analysis

```
## Notes

```
The script uses external libraries utils/ACSF.so, utils/cutoff.so and utils/cif2input.so for extracting CIF structures and computing symmetry functions.
Users should ensure that the parameters in the config1.data file are set before running the script, and that their raw data is entered into the input.xlsx spreadsheet. Additionally, CIF files should be imported into the input folder.
```
