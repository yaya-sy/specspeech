"""Load a local PCS ONNX punctuator for the background draft worker."""

from pathlib import Path


def load_punctuator(path):
    """Load an ONNX file with config.yaml and one SentencePiece .model beside it."""
    from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX, PunctCapSegModelONNX

    path = Path(path).expanduser().resolve(strict=True)
    tokenizers = list(path.parent.glob("*.model"))
    if len(tokenizers) != 1:
        raise ValueError("Place exactly one matching SentencePiece .model beside the punctuator ONNX file")
    model = PunctCapSegModelONNX(
        PunctCapSegConfigONNX(
            directory=str(path.parent), model_filename=path.name, spe_filename=tokenizers[0].name,
        ),
        ort_providers=["CPUExecutionProvider"],
    )
    return lambda texts: [
        " ".join(sentences) for sentences in model.infer([text.lower() for text in texts], num_workers=0)
    ]
