from pathlib import Path

import obonet
import pandas as pd

def load_hpo(hpo_path: str | Path):
    """
    Load the Human Phenotype Ontology from an OBO file.

    Parameters
    ----------
    hpo_path:
        Paht to hp.obo.

    Returns
    -------
    networkx.MulitDiGraph
        HPO ontology graph.
    """
    hpo_path = Path(hpo_path)

    if not hpo_path.exists():
        raise FileNotFoundError(f"HPO file not found: {hpo_path}")
    
    return obonet.read_obo(hpo_path)


def load_hpo_annotations(hpoa_path: str | Path) -> pd.DataFrame:
    """
    Load disease-to-phenotype annotations from phenotype.hpoa.

    Parameters
    ----------
    hpoa_path:
        Path to phenotype.hpoa.

    Returns
    -------
    pandas.DataFrame
        Filtered HPO annotation table.
    """
    hpoa_path = Path(hpoa_path)

    if not hpoa_path.exists():
        raise FileNotFoundError(f"HPO annotation file not found: {hpoa_path}")

    hpoa = pd.read_csv(
        hpoa_path,
        sep="\t",
        comment="#",
        dtype=str
    )

    if "qualifier" in hpoa.columns:
        hpoa = hpoa[hpoa["qualifier"].fillna("") != "NOT"]

    if "aspect" in hpoa.columns:
        hpoa = hpoa[hpoa["aspect"] == "P"]

    return hpoa.reset_index(drop=True)


def get_hpo_labels(hpo) -> dict[str, str]:
    """
    Extract readable HPO labels.

    Parameters
    ----------
    hpo:
        HPO ontology graph.

    Returns
    -------
    dict[str, str]
        Mapping from HPO ID to HPO label.
    """
    return {
        node_id: data.get("name", node_id)
        for node_id, data in hpo.nodes(data=True)
    }
