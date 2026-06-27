from __future__ import annotations

from pathlib import Path

import networkx as nx
import obonet
import pandas as pd

def load_filtered_hpo_annotations(hpoa_filtered_path: str | Path) -> pd.DataFrame: 
    """
    Load the filtered disease-to-HPO annotation table.

    Parameters
    ----------
    hpoa_filtered_path:
        Path to data/processed/hpoa_filtered.csv.

    Rerturns
    ---------
    pandas.DataFrame
        Filtered HPO annotation table.
    """
    hpoa_filtered_path = Path(hpoa_filtered_path)

    if not hpoa_filtered_path.exists():
        raise FileNotFoundError(f"Filtered HPO annotation file not found: {hpoa_filtered_path}")
    
    hpoa = pd.read_csv(hpoa_filtered_path, dtype=str)

    required_columns = {"database_id", "disease_name", "hpo_id"}
    missing = required_columns - set(hpoa.columns)

    if missing:
        raise ValueError(f"Missing required columns in hpoa_filtered: {missing}")
    
    return hpoa


def build_disease_hpo_triples(
    hpoa_filtered: pd.DataFrame,
    hpo: nx.multidigraph,
    include_hpo_hierarchy: bool = True,
    include_only_relevant_hpo_edges: bool = True,
) -> pd.DataFrame:
    """
    Build knowledge graph triples for PykEEN.

    Parameters
    ----------
    hpoa_filtered:
        Filtered disease-to-phenotype annotations.
    hpo:
        HPO ontology graph loaded from hp.obo.
    include_hpo_hierarchy:
        If True, include HPO child -> parent is_a triples.
    include_only_relevant_hpo_edges:
        If Ture, include only HPO hierarchy edges connected to HPO terms that appear in the filtered disease annoations. This keeps the graph smaller.
    
    Returns
    -------
    pandas.DataFrame
        Dataframe with columsn:
        head, rrelation, tail
    
    Triple types
    ------------
    Disease-to-phenotypes:
        disease_id, has_phenotype, hpo_id
    
    HPO hierarchy:
        hpo_child, is_a, hpo_parent
    """
    triples = []

    # 1. Disease -> phenotpye triples
    for _, row in hpoa_filtered.iterrows():
        disease_id = row["database_id"]
        hpo_id = row["hpo_id"]

        triples.append(
            {
                "head": disease_id,
                "relation": "has_phenotype",
                "tail": hpo_id,
            }
        )
    
    # 2. HPO child -> parent triples
    if include_hpo_hierarchy:
        used_hpo_terms = set(hpoa_filtered["hpo_id"].dropna().unique())

        for child, parent, _ in hpo.edges(data=True):
            if include_only_relevant_hpo_edges:
                if child not in used_hpo_terms and parent not in used_hpo_terms:
                    continue
            
            triples.append(
                {
                    "head": child,
                    "relation": "is_a",
                    "tail": parent,
                }
            )
    
    triples_df = pd.DataFrame(triples)

    # Remove duplicates becaues disease-HPO pair can occure more than once
    triples_df = triples_df.drop_duplicates().reset_index(drop=True)

    return triples_df


def build_entity_metadata(
    hpoa_filtered: pd.DataFrame,
    hpo: nx.MultiDiGraph,
    triples: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build metadata fro entities in the triples.
    It is useful later for interpretation, visualization, and filtering.

    Parameters
    ----------
    hpoa_filtered:
        Filtered disease-to-phenotype annotaion dataframe.
    hpo:
        HPO ontology graph.
    triples:
        Triple dataframe with columsn head,relation, tail.
    
    Returns
    -------
    pandas.DataFrame
        Entity metadata with columsn: entitiy_id, entitiy_type, label
    """
    all_entities = set(triples["head"]) | set(triples["tail"])

    disease_names = (
        hpoa_filtered[["database_id", "disease_name"]]
        .drop_duplicates()
        .set_index("database_id")["disease_name"]
        .to_dict()
    )

    rows = []

    for entity_id in sorted(all_entities):
        if entity_id.startswith("HP:"):
            entity_type = "phenotype"
            label = hpo.nodes[entity_id].get("name", entity_id) if entity_id in hpo else entity_id
        else:
            entity_type = "disease"
            label = disease_names.get(entity_id, entity_id)

        rows.append(
            {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "label": label,
            }
        )
    
    return pd.DataFrame(rows)


def save_triples(
    triples: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save triples as a tab-separated file.

    Useful for inspection.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    triples.to_csv(output_path, sep="\t", index=False, header=False)


def save_entity_metadata(
    entity_metadata: pd.DataFrame,
    ouput_path: str | Path,
) -> None:
    """
    Save entity metadata as CSV.
    """
    ouput_path = Path(ouput_path)
    ouput_path.parent.mkdir(parents=True, exist_ok=True)

    entity_metadata.to_csv(ouput_path, index=False)


def get_triple_statistics(triples: pd.DataFrame) -> dict[str, int]:
    """
    Compute simple statistics for the triple set.
    """
    entities = set(triples["head"]) | set(triples["tail"])

    stats = {
        "triples_total": len(triples),
        "entities_total": len(entities),
        "relations_total": triples["relation"].nunique(),
    }

    relation_counts = triples["relation"].value_counts().to_dict()

    for relation, count in relation_counts.items():
        stats[f"relation_{relation}"] = int(count)
    
    return stats