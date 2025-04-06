#!/bin/bash
# This script downloads user desired trained models from Zenodo.

#########################################
# Get user input
#########################################
# Get user desired test
declare -A test_to_modelname=(
    ["airfoil"]="AirfoilWrapper_2D_Airfoil"
    ["allen"]="2D_AllenCahn"
    ["cont_tran"]="2D_ContTranslation"
    ["darcy"]="2D_Darcy"
    ["disc_tran"]="2D_DiscContTranslation"
    ["poisson"]="2D_SinFrequency"
    ["shear_layer"]="2D_ShearLayer"
    ["wave_0_5"]="2D_WaveEquation"
)

printf "\nDigit the number of the test you want to download the model for and then press enter. Available models:\n"
select selected_test in "${!test_to_modelname[@]}"; do
    echo "selected: $selected_test"
    break
done

# Get user desired architecture
declare -A model_available=(
    ["Fourier Neural Operator"]="FNO"
    ["Convolutional Neural Operator"]="CNO"
)

printf "\nDigit the number of the architecture you want to download the model for and then press enter. Available architectures:\n"
select selected_model in "${!model_available[@]}"; do
    echo "selected: $selected_model"
    break
done

# Get user desired mode for the model
declare -A mode_available=(
    ["model with default hyperameter configuration"]="default"
    ["model with our best hyperparameter configuration"]="best"
    ["model with our best hyperparameter configuration, same number of parameters as default"]="bestsamedofs"
)

if ["$selected_test" = "cont_tran" && "$selected_model" = "Fourier Neural Operator"]; then
    mode_available["Best continuous transport for FNO with 500k parameters"]="best500k"
    mode_available["Best continuous transport for FNO with 50M parameters"]="best50M"
    mode_available["Best continuous transport for FNO with 150M parameters"]="best150M"
fi

printf "\nDigit the mode of the architecture you want to download the model for and then press enter. Available modality:\n"
select selected_mode in "${!mode_available[@]}"; do
    echo "selected: $selected_mode"
    break
done

# For the moment all the tests are with the L1 norm
loss="L1" 

#########################################
# Download process
#########################################
# Define the download link
download_link="https://zenodo.org/records/15055547/files/model_${model_available[$selected_model]}_${mode_available[$selected_mode]}_${test_to_modelname[$selected_test]}"

# Define the destination folder
dest_folder="neural_operators/tests/${model_available[$selected_model]}/$selected_test/loss_${loss}_mode_${mode_available[$selected_mode]}"
mkdir -p "$dest_folder"
printf "\nDestination folder: $dest_folder\n"

# Download the file using curl
printf "\nDownloading model...\n"
printf "\nDownload link: $download_link\n"
curl --output "${dest_folder}/model_${model_available[$selected_model]}_${test_to_modelname[$selected_test]}" "$download_link"

if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Error occurred during download. Please check if the model exists."
fi