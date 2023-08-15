import os
import csv
from tqdm import tqdm

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.PubChem110m")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class PubChem110m(data.MoleculeDataset):
    """
    PubChem.
    This dataset doesn't contain any label information.

    Statistics:
        - #Molecule:

    Parameters:
        path (str):
        verbose (int, optional): output verbose level
        **kwargs
    """
    # TODO: download path & md5. Is it the statistics right?

    target_fields = []

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        smiles_file = os.path.join(path, "CID-SMILES")

        with open(smiles_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            if verbose:
                reader = iter(
                    tqdm(
                        reader,
                        f"Loading {path}",
                        utils.get_line_count(smiles_file),
                    )
                )
            smiles_list = [values[1] for values in reader]
        targets = {}
        self.load_smiles(smiles_list, targets, lazy=True, verbose=verbose, **kwargs)