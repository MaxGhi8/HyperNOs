#!/bin/bash

# Associate the name of the test to the name of the model file
declare -A test_to_modelname=(
    ["airfoil"]="model_AirfoilWrapper_2D_Airfoil"
    ["allen"]="model_FNO_2D_AllenCahn"
    ["cont_tran"]="model_FNO_2D_ContTranslation"
    ["darcy"]="model_FNO_2D_Darcy"
    ["disc_tran"]="model_FNO_2D_DiscContTranslation"
    ["poisson"]="model_FNO_2D_SinFrequency"
    ["shear_layer"]="todo"
    ["wave_0_5"]="model_FNO_2D_WaveEquation"
)

# Get user desired test
printf "\nDigit the number of the test you want to download the model for and then press enter. Available models:\n"
select selected_test in "${!test_to_modelname[@]}"; do
    echo "selected: $selected_test"
    break
done

# Select the name of the model
declare -A model_available=(
    ["Fourier Neural Operator"]="FNO"
    ["Convolutional Neural Operator"]="CNO"
)

# Get user desired architecture
printf "\nDigit the number of the architecture you want to download the model for and then press enter. Available architectures:\n"
select selected_model in "${!model_available[@]}"; do
    echo "selected: $selected_model"
    break
done

# Select the mode for the model
declare -A mode_available=(
    ["model with default hyperameter configuration"]="default"
    ["model with our best hyperparameter configuration"]="best"
    ["model with our best hyperparameter configuration, same number of parameters as default"]="best_samedofs"
    ["Best continuous transport for FNO with 500k parameters"]="best_500k"
    ["Best continuous transport for FNO with 50M parameters"]="best_50M"
    ["Best continuous transport for FNO with 150M parameters"]="best_150M"
)

# Get user desired architecture
printf "\nDigit the mode of the architecture you want to download the model for and then press enter. Available modality:\n"
select selected_mode in "${!mode_available[@]}"; do
    echo "selected: $selected_mode"
    break
done

# For the moment all the tests are with the L1 norm
loss="L1" 

# Get link and folder for the selected model
download_link="https://zenodo.org/uploads/14860202/${test_to_modelname[$selected_test]}_${mode_available[$selected_mode]}_${model_available[$selected_model]}"

# Define the destination folder
dest_folder="neural_operators/tests/${model_available[$selected_model]}/$selected_test/loss_${loss}_mode_${mode_available[$selected_mode]}"
# mkdir -p "$dest_folder"
printf "\nDestination folder: $dest_folder\n"

# Get filename from the download link (you can customize this)
filename=${test_to_modelname[$selected_test]}

# Download the file using curl
printf "\nDownloading model...\n"
curl --output "${dest_folder}/${filename}" "$download_link"

if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Error occurred during download. Please check if the model exists."
fi
