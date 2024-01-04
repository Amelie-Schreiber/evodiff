import torch
import numpy as np
from evodiff.pretrained import OA_DM_38M
from evodiff.conditional_generation_seq import inpaint_simple, generate_binder
from evodiff.utils import Tokenizer

def main(protein_1, protein_2):
    # Initialize the model
    checkpoint = OA_DM_38M()
    model, collater, tokenizer, scheme = checkpoint

    # Determine the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()  # Set the model to evaluation mode

    # Connect the protein sequences with a G-linker
    g_linker = "GGGGGGGGGGGGGGGGGGGG"
    connected_proteins = protein_1 + g_linker + protein_2

    # Inpaint an IDR in the connected sequence
    idr_start_idx = len(protein_1)  # Start of G-linker
    idr_end_idx = idr_start_idx + len(g_linker)  # End of G-linker
    _, generated_idr_sequence, generated_idr = inpaint_simple(model, connected_proteins, idr_start_idx, idr_end_idx, tokenizer, device=device)

    # Discard the second protein sequence
    binder_template = protein_1 + generated_idr

    # Generate a binder for the binder_template
    generated_binder = generate_binder(model, binder_template, binder_length=100, tokenizer=tokenizer, device=device)

    # Form the generated scaffold
    generated_scaffold = binder_template + generated_binder

    # Print results
    print("Generated IDR Linker:", generated_idr, "Length:", len(generated_idr))
    print("Binder Template:", binder_template, "Length:", len(binder_template))
    print("Generated Binder:", generated_binder, "Length:", len(generated_binder))
    print("Generated Scaffold:", generated_scaffold, "Length:", len(generated_scaffold))

# Example usage
protein_1 = "MEDYQAAEETAFVVDEVSNIVKEAIESAIGGNAYQHSKVNQWTTNVVEQTLSQLTKLGKPFKYIVTCVIMQKNGAGLHTASSCFWDSSTDGSCTVRWENKTMYCIVSAFGLSI"
protein_2 = "MATSIGVSFSVGDGVPEAEKNAGEPENTYILRPVFQQRFRPSVVKDCIHAVLKEELANAEYSPEEMPQLTKHLSENIKDKLKEMGFDRYKMVVQVVIGEQRGEGVFMASRCFWDADTDNYTHDVFMNDSLFCVVAAFGCFYY"
main(protein_1, protein_2)

