#!/bin/bash
# filename: all_uncertainty_input_data.sh

echo ====================================================================
echo Executing \"UncertaintyInputData.py\" for all PDB-files in \"../cache\":

for file in $( find ../cache/*.pdb ); do
    
    id=${file:9:4}
    # if [ ! -f "../cache/$id.uid" ] 
    # then
        echo ==================================================================== | tee -a -i all_uncertainty_input_data.log
        echo Processing PDB-ID: $id                                               | tee -a -i all_uncertainty_input_data.log
        echo ==================================================================== | tee -a -i all_uncertainty_input_data.log
        python3.5 UncertaintyInputData.py -o $id 2>&1                             | tee -a -i all_uncertainty_input_data.log
    # fi
done

