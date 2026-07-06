from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


def create_triples_factory(
    triples: pd.DataFrame,
    create_inverse_triples: bool = True
) -> TriplesFactory:
    """
    Create a PyKEEN TriplesFactory froma triples dataframe.
    PyKEEN expects triples as a NumPy array of strings with shape: (n_triples, 3)

    Parameters
    ----------
    triples:
        Dataframe with columns: head, relation, tail

    create_inverse_triples:
        If True, PyKEEN internally creates inverse relations for training.
        This can help modles learn from directed graphs by also exposing reverse relation patterns.

    Returns
    -------
    pykeen.triples.TripleFactory
        PyKEEN triples factory.
    """
    required_columns = {"head", "relation", "tail"}
    missing = required_columns - set(triples.columns)

    if missing:
        raise ValueError(f"Missing required triple columns: {missing}")
    
    labeled_triples = triples[["head", "relation", "tail"]].astype(str).values

    return TriplesFactory.from_labeled_triples(
        triples=labeled_triples,
        create_inverse_triples=create_inverse_triples,
    )



def split_triples_factory(
    triples_factory: TriplesFactory,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_state: int = 5,
):
    """
    Split triples into train/validation/test sets.

    Parameters
    ----------
    triples_factory:
        Full triples factory.
    ratios:
        Train/validation/test split ratios.
    random_state:
        Random seed.

    Returns
    --------
    tuple
        traing, validation, testing triples factories.

    """

    training, validation, testing = triples_factory.split(
        ratios=ratios,
        random_state=random_state,
    )

    return training, validation, testing



def train_transe_model(
    training,
    validation,
    testing,
    embedding_dim: int = 64,
    num_epochs: int = 100,
    batch_size: int = 256,
    learing_rate: float = 0.001,
    random_seed: int = 5,
    model_output_directory: str | Path | None = None,
):
    """
    Train a TransE model with PyKEEN.
    TransE learns vectors so that: head + relation ≈ tail

    Parameters
    ----------
    training:
        Training triples factory.
    validation:
        Validation triples factory.
    testing:
        Testing triples factory.
    embedding_dim:
        Number of dimensions for entity and relation embeddings.
    num_epochs:
        Number of training epochs.
    batch_size:
        Training batch size.
    learning_rate:
        Optimizer learning rate.
    random_seed:
        Random seed for reproducibility.
    model_output_directory:
        Optional path where the trained pipeline result is saved.

    Returns
    -------
    pykeen.pipeline.PipelineResult
        Result object containing the trained model, metrics, and metadata.
    """
    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model="TransE",
        model_kwargs={
            "embedding_dim": embedding_dim,
            "scoring_fct_norm": 1,
        },
        optimizer="Adam",
        optimizer_kwargs={
            "lr":learing_rate,
        },
        training_kwargs={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
        },
        random_seed=random_seed,
        device="cpu",
    )

    if model_output_directory is not None:
        model_output_directory = Path(model_output_directory)
        model_output_directory.mkdir(parents=True, exist_ok=True)
        result.save_to_directory(model_output_directory)

    return result


def extract_entity_embeddings(
    result,
    entity_ids: list[str],
) -> pd.DataFrame:
    """
    Extract learned entitiy embeddings from a trained PyKEEN model.

    Paraemters
    ----------
    result:
        PyKEEN PipelineResult returned by train_transe_model().
    entity_ids:
        List of entity labels to extract.
    
    Returns
    -------
    pandas.DataFrame
        One row per entity: entity_id, dim_0, dim_1, ...
    """
    model = result.model
    entity_to_id = result.training.entity_to_id

    valid_entity_ids = [
        entity_id
        for entity_id in entity_ids
        if entity_id in entity_to_id
    ]

    if not valid_entity_ids:
        raise ValueError("None of the requested entity IDs exist in the PyKEEN model.")

    pykeen_ids = [entity_to_id[entity_id] for entity_id in valid_entity_ids]

    device = model.device
    indices = torch.as_tensor(pykeen_ids, dtype=torch.long, device=device)

    model.eval()

    with torch.no_grad():
        embedding_tensor = model.entity_representations[0](indices=indices)
        embedding_array = embedding_tensor.detach().cpu().numpy()

    rows = []

    for entity_id, vector in zip(valid_entity_ids, embedding_array):
        row = {"entity_id": entity_id}

        for i, value in enumerate(vector):
            row[f"dim_{i}"] = float(value)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_disease_embeddings(
    result,
    entitiy_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract embeddings only for disease entities.

    Parameters
    ----------
    result:
        PyKEEN PipelineResult.
    entity_metadata:
        Dataframe with columns: entity_id, entity_type, label
    """
    required_columns = {"entity_id", "entity_type", "label"}
    missing = required_columns - set(entitiy_metadata.columns)

    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    
    disease_metadata = entitiy_metadata[
        entitiy_metadata["entity_type"] == "disease"
    ].copy()

    disease_ids = disease_metadata["entity_id"].tolist()

    disease_embeddings = extract_entity_embeddings(
        result=result,
        entity_ids=disease_ids,
    )

    disease_embeddings = disease_embeddings.merge(
        disease_metadata,
        on="entity_id",
        how="left",
    )

    return disease_embeddings



def save_embedding(
    embeddings: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save embeddings to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings.to_csv(output_path, index=False)


def save_metrics(
    result,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Save PyKEEN evaluation metrics to CSV.

    Parameters
    ----------
    result:
        PyKEEN PipelineResult.
    output_path:
        Path to output CSV.

    Returns
    -------
    pandas.DataFrame
        Metrics dataframe.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_dict = result.metric_results.to_flat_dict()

    metrics_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in metrics_dict.items()
        ]
    )

    metrics_df.to_csv(output_path, index=False)

    return metrics_df


def load_triples_from_tsv(triples_path: str | Path) -> pd.DataFrame:
    """
    Load triples from a TSV file created by save_triples().
    """
    triples_path = Path(triples_path)

    if not triples_path.exists():
        raise FileNotFoundError(f"Triples file not found: {triples_path}")
    
    return pd.read_csv(
        triples_path,
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
        dtype=str,
    )