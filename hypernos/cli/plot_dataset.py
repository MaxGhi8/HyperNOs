import argparse
import os

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from hypernos.datasets import NO_load_data_model

def get_parser():
    parser = argparse.ArgumentParser(description="Load a dataset and plot its input and output to a PDF.")
    parser.add_argument("example", type=str, help="Name of the example (e.g., poisson, wave_0_5, darcy, allen, shear_layer, cont_tran, disc_tran, airfoil)")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to plot.")
    parser.add_argument("--filename", type=str, default=None, help="Specific filename to load if needed.")
    parser.add_argument("--out", type=str, default="dataset_samples.pdf", help="Output PDF filename.")
    return parser

def denormalize(input_tensor, output_tensor, example, which_example):
    """Denormalizes the tensors based on the example specifics."""
    match which_example:
        case "fhn":
            input_tensor = example.a_normalizer.decode(input_tensor)
            output_tensor[:, :, [0]] = example.v_normalizer.decode(output_tensor[:, :, [0]])
            output_tensor[:, :, [1]] = example.w_normalizer.decode(output_tensor[:, :, [1]])

        case "hh":
            input_tensor = example.a_normalizer.decode(input_tensor)
            output_tensor[:, :, [0]] = example.v_normalizer.decode(output_tensor[:, :, [0]])
            output_tensor[:, :, [1]] = example.m_normalizer.decode(output_tensor[:, :, [1]])
            output_tensor[:, :, [2]] = example.h_normalizer.decode(output_tensor[:, :, [2]])
            output_tensor[:, :, [3]] = example.n_normalizer.decode(output_tensor[:, :, [3]])

        case "crosstruss":
            output_tensor[:, :, :, 0] = (example.max_x - example.min_x) * output_tensor[:, :, :, 0] + example.min_x
            output_tensor[:, :, :, 1] = (example.max_y - example.min_y) * output_tensor[:, :, :, 1] + example.min_y
            output_tensor[:, :, :, [0]] = output_tensor[:, :, :, [0]] * input_tensor[:, :, :]
            output_tensor[:, :, :, [1]] = output_tensor[:, :, :, [1]] * input_tensor[:, :, :]

        case "poisson" | "wave_0_5" | "allen" | "shear_layer" | "darcy":
            input_tensor = input_tensor * (example.max_data - example.min_data) + example.min_data
            output_tensor = (example.max_model - example.min_model) * output_tensor + example.min_model

        case "eig":
            input_tensor = example.input_normalizer.decode(input_tensor)
            output_tensor = example.output_normalizer.decode(output_tensor)

        case "darcy_zongyi":
            input_tensor = example.a_normalizer.decode(input_tensor)
            output_tensor = example.u_normalizer.decode(output_tensor)

        case "bampno_8_domain" | "bampno_continuation":
            input_tensor = example.input_normalizer.decode(input_tensor)

    return input_tensor, output_tensor

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    print(f"Loading dataset for '{args.example}'...")
    
    # Specific check for BAMPNO/Darcy which requires a filename
    if "bampno" in args.example.lower() and args.filename is None:
        print("Error: 'bampno' datasets require a --filename (e.g., --filename Darcy_8_chebyshev_60pts.mat)")
        return

    try:
        # Default initialization based on common settings.
        # We use a minimum of 20 training samples to avoid NaN in normalization (standard deviation).
        dataset_loader = NO_load_data_model(
            which_example=args.example,
            no_architecture={'FourierF': 0, 'retrain': 4},
            batch_size=args.samples,
            training_samples=max(args.samples, 20),
            filename=args.filename
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    loader = dataset_loader.train_loader
    input_batch, output_batch = next(iter(loader))
    
    # Handle tuples for models like DON
    if isinstance(input_batch, (list, tuple)):
        input_batch = input_batch[0]
        
    # Denormalize to map back to physics variables
    input_tensor, output_tensor = denormalize(input_batch, output_batch, dataset_loader, args.example)
    
    input_np = input_tensor.cpu().numpy()
    output_np = output_tensor.cpu().numpy()
    
    print(f"Input shape: {input_np.shape}")
    print(f"Output shape: {output_np.shape}")
    
    n_samples = min(args.samples, input_np.shape[0])
    
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    base_name, ext = os.path.splitext(args.out)
    if not ext:
        ext = ".pdf"
        
    out_in_path = os.path.join(out_dir, f"{base_name}_input{ext}")
    out_out_path = os.path.join(out_dir, f"{base_name}_output{ext}")

    # --- Plot Inputs ---
    fig_in, axs_in = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4), squeeze=False)
    for i in range(n_samples):
        inp = input_np[i]
        if inp.ndim >= 3:
            inp = inp[..., 0] # take first channel if multiple
        inp_sq = inp.squeeze()
        
        if inp_sq.ndim == 1:
            axs_in[0, i].plot(inp_sq)
            axs_in[0, i].set_title("Diffusion coefficient", fontweight="bold")
        else:
            if inp_sq.ndim == 3 and hasattr(dataset_loader, "X_phys"):
                # Multi-patch full domain plotting (e.g. BAMPNO)
                X = dataset_loader.X_phys.cpu().numpy()
                Y = dataset_loader.Y_phys.cpu().numpy()
                vmin_i, vmax_i = inp_sq.min(), inp_sq.max()
                for patch in range(X.shape[0]):
                    im_in = axs_in[0, i].pcolormesh(
                        X[patch], Y[patch], inp_sq[patch],
                        vmin=vmin_i, vmax=vmax_i, shading="auto",
                        rasterized=True
                    )
                axs_in[0, i].set_aspect("equal")
                axs_in[0, i].set_title("Diffusion coefficient", fontweight="bold")
            else:
                im_in = axs_in[0, i].imshow(inp_sq)
                axs_in[0, i].set_title("Diffusion coefficient", fontweight="bold")
                
            divider_in = make_axes_locatable(axs_in[0, i])
            cax_in = divider_in.append_axes("right", size="5%", pad=0.05)
            fig_in.colorbar(im_in, cax=cax_in)
        
    # fig_in.suptitle(f"Dataset Inputs: {args.example}", fontsize=16)
    fig_in.tight_layout()
    fig_in.savefig(out_in_path, format="pdf", bbox_inches="tight", dpi=600)
    plt.close(fig_in)
    
    # --- Plot Outputs ---
    fig_out, axs_out = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4), squeeze=False)
    for i in range(n_samples):
        out = output_np[i]
        if out.ndim >= 3:
            out = out[..., 0] # take first channel if multiple
        out_sq = out.squeeze()
        
        if out_sq.ndim == 1:
            axs_out[0, i].plot(out_sq)
            axs_out[0, i].set_title("Solution", fontweight="bold")
        else:
            if out_sq.ndim == 3 and hasattr(dataset_loader, "X_phys"):
                # Multi-patch full domain plotting (e.g. BAMPNO)
                X = dataset_loader.X_phys.cpu().numpy()
                Y = dataset_loader.Y_phys.cpu().numpy()
                vmin_i, vmax_i = out_sq.min(), out_sq.max()
                for patch in range(X.shape[0]):
                    im_out = axs_out[0, i].pcolormesh(
                        X[patch], Y[patch], out_sq[patch],
                        vmin=vmin_i, vmax=vmax_i, shading="auto",
                        rasterized=True
                    )
                axs_out[0, i].set_aspect("equal")
                axs_out[0, i].set_title("Solution", fontweight="bold")
            else:
                im_out = axs_out[0, i].imshow(out_sq)
                axs_out[0, i].set_title("Solution", fontweight="bold")
                
            divider_out = make_axes_locatable(axs_out[0, i])
            cax_out = divider_out.append_axes("right", size="5%", pad=0.05)
            fig_out.colorbar(im_out, cax=cax_out)
        
    # fig_out.suptitle(f"Dataset Outputs: {args.example}", fontsize=16)
    fig_out.tight_layout()
    fig_out.savefig(out_out_path, format="pdf", bbox_inches="tight", dpi=600)
    plt.close(fig_out)
    
    print(f"Successfully saved input plot to {out_in_path}")
    print(f"Successfully saved output plot to {out_out_path}")

if __name__ == "__main__":
    main()
