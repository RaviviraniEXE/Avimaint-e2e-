from __future__ import annotations

import hashlib
import csv
import re
import importlib
import json
import os
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_ENTITY_TYPES = {
    "ABN_PROC",
    "ACTION",
    "FAULT",
    "LOC",
    "MAINT_ITEM",
    "OP_CTX",
    "OUTCOME",
    "REFERENCE",
    "TECH_OBS",
}

EXPECTED_RELATION_TYPES = {
    "ACTION_ADDRESSES_ISSUE",
    "ACTION_FOLLOWS_REFERENCE",
    "ACTION_INVESTIGATES_ISSUE",
    "ACTION_ON_ITEM",
    "ACTION_RESULTS_IN_OUTCOME",
    "ACTION_USES_ITEM",
    "HAS_LOCATION",
    "HAS_PART",
    "ISSUE_ON_ITEM",
    "OBSERVATION_OF_ITEM",
    "OCCURS_UNDER_CONTEXT",
}

REQUIRED_MODEL_FILES = (
    "config.json",
    "vocab.txt",
)

ASCII_CASE_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)


class SpERTRuntimeError(RuntimeError):
    """A configuration, compatibility, or inference failure."""


def _patch_transformers_tie_weights() -> None:
    """Compatibility shim for transformers >= ~4.50.

    Newer transformers refactored weight tying so that ``tie_weights`` references
    ``self.all_tied_weights_keys`` — an attribute the older Eberts-style SpERT
    model classes never set, which raises
    ``'SpERT' object has no attribute 'all_tied_weights_keys'`` during
    ``from_pretrained``. SpERT inference does not need weight tying, so we wrap
    ``tie_weights`` to guarantee the attribute exists and to swallow that one
    AttributeError. This lets the trained SpERT checkpoint load under the SAME
    modern transformers the dashboard/reranker use — no downgrade, no conflict.
    """
    try:
        import transformers.modeling_utils as _mu
        PM = _mu.PreTrainedModel
        if getattr(PM, "_avimaint_tie_patched", False):
            return
        _orig_tie = PM.tie_weights

        def _safe_tie(self, *args, **kwargs):
            if not hasattr(self, "all_tied_weights_keys"):
                try:
                    self.all_tied_weights_keys = []
                except Exception:
                    pass
            try:
                return _orig_tie(self, *args, **kwargs)
            except AttributeError:
                return None

        PM.tie_weights = _safe_tie
        PM._avimaint_tie_patched = True
    except Exception:
        # if transformers internals differ, leave them untouched
        pass


def normalise_query_for_model(text: str) -> str:
    """True-Raw SpERT must receive the original query representation unchanged."""
    return str(text)

RAW_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)+|[A-Za-z0-9]+|[^\w\s]")
def matched_raw_tokenize(text: str):
    return [(m.group(0), m.start(), m.end()) for m in RAW_TOKEN_RE.finditer(str(text))]


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    spert_root: Path
    model_path: Path
    types_path: Path


def _as_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser().resolve()


def _first_existing(candidates: Iterable[Path], predicate: Callable[[Path], bool]) -> Path | None:
    for candidate in candidates:
        if predicate(candidate):
            return candidate.resolve()
    return None


def _valid_spert_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "spert" / "models.py").is_file()
        and (path / "spert" / "input_reader.py").is_file()
        and (path / "spert" / "prediction.py").is_file()
    )


def _valid_model_path(path: Path) -> bool:
    has_weights = (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
    return (
        path.is_dir()
        and has_weights
        and all((path / name).is_file() for name in REQUIRED_MODEL_FILES)
    )


def _latest_final_model(project_root: Path) -> Path | None:
    save_root = project_root / "outputs" / "spert" / "save"
    if not save_root.is_dir():
        return None
    candidates = [path for path in save_root.glob("**/final_model") if _valid_model_path(path)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve()


def resolve_runtime_paths(
    project_root: str | os.PathLike[str],
    spert_root: str | os.PathLike[str] | None = None,
    model_path: str | os.PathLike[str] | None = None,
    types_path: str | os.PathLike[str] | None = None,
) -> RuntimePaths:
    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise SpERTRuntimeError(f"Project root does not exist: {project}")

    explicit_spert = _as_path(spert_root or os.getenv("AVIMAINT_SPERT_ROOT"))
    if explicit_spert:
        resolved_spert = explicit_spert if _valid_spert_root(explicit_spert) else None
    else:
        resolved_spert = _first_existing(
            (
                project.parents[1] / "external" / "spert",
                project.parent / "spert",
                project / "spert",
            ),
            _valid_spert_root,
        )
    if resolved_spert is None:
        attempted = explicit_spert or (project.parent / "spert")
        raise SpERTRuntimeError(
            "SpERT source was not found. Expected a folder containing "
            f"'spert\\models.py'. Checked: {attempted}"
        )

    explicit_model = _as_path(model_path or os.getenv("AVIMAINT_SPERT_MODEL"))
    repo_root = project.parents[1]
    registry_path = project / "outputs" / "reports" / "normalization_spert_matched_v2" / "MODEL_REGISTRY_V2.json"
    if explicit_model:
        resolved_model = explicit_model if _valid_model_path(explicit_model) else None
    else:
        resolved_model = None
        if registry_path.is_file():
            reg = json.loads(registry_path.read_text(encoding="utf-8"))
            rel = str((reg.get("raw") or {}).get("final_model_path") or "")
            candidate = repo_root / rel if rel else None
            if candidate and _valid_model_path(candidate): resolved_model = candidate.resolve()
    if resolved_model is None:
        raise SpERTRuntimeError(f"Corrected true-Raw model could not be resolved from {registry_path}. Historical outputs/spert is intentionally refused.")

    resolved_types = (
        _as_path(types_path or os.getenv("AVIMAINT_SPERT_TYPES"))
        or (project / "outputs" / "spert_normalized" / "raw" / "avimaint_types.json").resolve()
    )
    if not resolved_types.is_file():
        raise SpERTRuntimeError(
            f"The authoritative SpERT type file was not found: {resolved_types}"
        )

    return RuntimePaths(
        project_root=project,
        spert_root=resolved_spert,
        model_path=resolved_model,
        types_path=resolved_types,
    )


def load_and_validate_types(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpERTRuntimeError(f"Could not read the SpERT type file {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SpERTRuntimeError("The SpERT type file must contain one JSON object.")

    entities = raw.get("entities")
    relations = raw.get("relations")
    if not isinstance(entities, dict) or not isinstance(relations, dict):
        raise SpERTRuntimeError(
            "The SpERT type file must contain 'entities' and 'relations' objects."
        )

    entity_types = set(entities)
    relation_types = set(relations)
    if entity_types != EXPECTED_ENTITY_TYPES:
        missing = sorted(EXPECTED_ENTITY_TYPES - entity_types)
        extra = sorted(entity_types - EXPECTED_ENTITY_TYPES)
        raise SpERTRuntimeError(
            f"Entity labels do not match aviation_compact_v1. Missing={missing}; extra={extra}"
        )
    if relation_types != EXPECTED_RELATION_TYPES:
        missing = sorted(EXPECTED_RELATION_TYPES - relation_types)
        extra = sorted(relation_types - EXPECTED_RELATION_TYPES)
        raise SpERTRuntimeError(
            f"Relation labels do not match aviation_compact_v1. Missing={missing}; extra={extra}"
        )

    for label, value in entities.items():
        if not isinstance(value, dict) or not {
            "short",
            "verbose",
        }.issubset(value):
            raise SpERTRuntimeError(f"Entity type {label} is missing SpERT metadata.")
    for label, value in relations.items():
        if not isinstance(value, dict) or not {
            "short",
            "verbose",
            "symmetric",
        }.issubset(value):
            raise SpERTRuntimeError(f"Relation type {label} is missing SpERT metadata.")
    return raw


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_exact_tokenizer(project_root: Path) -> Callable[[str], Any]:
    return matched_raw_tokenize

def _validate_raw_tokenizer(project_root: Path, tokenizer: Callable[[str], Any]) -> None:
    repo_root = project_root.parents[1]
    source = repo_root / "outputs" / "normalization" / "full_corpus" / "raw.csv"
    gold_dir = project_root / "outputs" / "gold_variants" / "raw"
    if not source.is_file() or not gold_dir.is_dir():
        raise SpERTRuntimeError("Raw tokenizer parity artifacts are missing; refusing unverified query inference.")
    text_map={}
    with source.open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f): text_map[str(row["record_id"])]=str(row["prediction_text"])
    checked=0; bad=[]
    for path in sorted(gold_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            rec=json.loads(line); rid=str(rec.get("ident")); got=[x[0] for x in tokenizer(text_map[rid])]
            checked+=1
            if got!=rec.get("tokens") and len(bad)<3: bad.append(rid)
    if checked!=1600 or bad:
        raise SpERTRuntimeError(f"True-Raw tokenizer parity failed: checked={checked}, example mismatches={bad}")


def tokenize_with_offsets(
    text: str,
    tokenizer: Callable[[str], Any],
) -> tuple[list[str], list[tuple[int, int]]]:
    try:
        raw_tokens = list(tokenizer(text))
    except Exception as exc:
        raise SpERTRuntimeError(
            f"The project tokenizer could not process this query: {exc}"
        ) from exc

    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for index, raw in enumerate(raw_tokens):
        if not isinstance(raw, (tuple, list)) or len(raw) < 3:
            raise SpERTRuntimeError(
                "src.data.preannotate.tokenize must return "
                "(token, character_start, character_end) tuples."
            )
        token = str(raw[0])
        try:
            start, end = int(raw[1]), int(raw[2])
        except (TypeError, ValueError) as exc:
            raise SpERTRuntimeError(f"Invalid character offsets for token {index}: {raw}") from exc
        if not token or start < 0 or end <= start or end > len(text):
            raise SpERTRuntimeError(f"Invalid token or character offsets at token {index}: {raw}")
        tokens.append(token)
        offsets.append((start, end))
    return tokens, offsets


def _surface(
    text: str,
    tokens: list[str],
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> str:
    if 0 <= start < end <= len(offsets):
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        return text[char_start:char_end]
    return " ".join(tokens[start:end])


def format_predictions(
    *,
    text: str,
    tokens: list[str],
    offsets: list[tuple[int, int]],
    document: Any,
    predicted_entities: list[Any],
    predicted_relations: list[Any],
    get_span_tokens: Callable[[Any, Any], Any],
) -> dict[str, Any]:
    entity_records: list[dict[str, Any]] = []
    signatures: list[tuple[int, int, str]] = []

    for entity in predicted_entities:
        encoded_start, encoded_end, entity_type, score = entity
        span_tokens = get_span_tokens(document.tokens, (int(encoded_start), int(encoded_end)))
        if span_tokens is None or len(span_tokens) == 0:
            continue
        start = int(span_tokens[0].index)
        end = int(span_tokens[-1].index) + 1
        label = str(entity_type.identifier)
        entity_records.append(
            {
                "type": label,
                "start": start,
                "end": end,
                "text": _surface(text, tokens, offsets, start, end),
                "score": float(score),
            }
        )
        signatures.append((start, end, label))

    order = sorted(
        range(len(entity_records)),
        key=lambda index: (
            entity_records[index]["start"],
            entity_records[index]["end"],
            entity_records[index]["type"],
        ),
    )
    entity_records = [entity_records[index] for index in order]
    signatures = [signatures[index] for index in order]
    signature_to_index = {signature: index for index, signature in enumerate(signatures)}

    relation_records: list[dict[str, Any]] = []
    seen_relations: set[tuple[str, int, int]] = set()
    for relation in predicted_relations:
        head, tail, relation_type, score = relation

        def endpoint_signature(endpoint: Any) -> tuple[int, int, str] | None:
            encoded_start, encoded_end, entity_type = endpoint
            span_tokens = get_span_tokens(document.tokens, (int(encoded_start), int(encoded_end)))
            if span_tokens is None or len(span_tokens) == 0:
                return None
            return (
                int(span_tokens[0].index),
                int(span_tokens[-1].index) + 1,
                str(entity_type.identifier),
            )

        head_signature = endpoint_signature(head)
        tail_signature = endpoint_signature(tail)
        if head_signature is None or tail_signature is None:
            continue
        head_index = signature_to_index.get(head_signature)
        tail_index = signature_to_index.get(tail_signature)
        if head_index is None or tail_index is None:
            continue
        label = str(relation_type.identifier)
        signature = (label, head_index, tail_index)
        if signature in seen_relations:
            continue
        seen_relations.add(signature)
        relation_records.append(
            {
                "type": label,
                "head": head_index,
                "tail": tail_index,
                "score": float(score),
            }
        )

    relation_records.sort(key=lambda row: (row["head"], row["tail"], row["type"]))
    return {
        "text": text,
        "tokens": tokens,
        "entities": entity_records,
        "relations": relation_records,
    }


class SpERTRunner:
    """Load the trained model once and perform exact query-side inference."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        rel_filter_threshold: float = 0.4,
        max_span_size: int = 10,
        max_pairs: int = 1000,
        size_embedding: int = 25,
        prop_drop: float = 0.1,
        force_cpu: bool = False,
        max_query_tokens: int = 128,
    ):
        self.paths = paths
        self.rel_filter_threshold = float(rel_filter_threshold)
        self.max_span_size = int(max_span_size)
        self.max_pairs = int(max_pairs)
        self.size_embedding = int(size_embedding)
        self.prop_drop = float(prop_drop)
        self.max_query_tokens = int(max_query_tokens)
        self._lock = threading.Lock()
        self._tokenize = _import_exact_tokenizer(paths.project_root)
        _validate_raw_tokenizer(paths.project_root, self._tokenize)
        self._types = load_and_validate_types(paths.types_path)

        spert_text = str(paths.spert_root)
        if spert_text not in sys.path:
            sys.path.insert(0, spert_text)
        try:
            import torch
            from spert import models, prediction, sampling, util
            from spert.entities import Dataset
            from spert.input_reader import JsonPredictionInputReader
            from transformers import BertConfig, BertTokenizer
        except Exception as exc:
            raise SpERTRuntimeError(
                "The selected Python environment cannot import the trained "
                "SpERT runtime. It must contain torch, transformers, and the "
                "dependencies used for the original experiment."
            ) from exc

        # make the trained SpERT checkpoint loadable under modern transformers
        _patch_transformers_tie_weights()

        self._torch = torch
        self._prediction = prediction
        self._sampling = sampling
        self._util = util
        self._dataset_class = Dataset

        try:
            self._tokenizer = BertTokenizer.from_pretrained(
                str(paths.model_path),
                do_lower_case=False,
                local_files_only=True,
            )
            self._reader = JsonPredictionInputReader(
                str(paths.types_path),
                self._tokenizer,
                max_span_size=self.max_span_size,
                spacy_model=None,
            )
            model_class = models.get_model("spert")
            # --- transformers >= ~4.50 compatibility --------------------------
            # The modern loader expects `all_tied_weights_keys` (and friends) on
            # the model class. Older SpERT classes don't define them, which raises
            # "'SpERT' object has no attribute 'all_tied_weights_keys'". SpERT has
            # no tied weights that matter for inference, so we force these to empty
            # on the class itself — this shadows any base-class property and is
            # reached no matter where the loader accesses it.
            for _attr in ("all_tied_weights_keys", "_tied_weights_keys",
                          "_keys_to_ignore_on_load_missing"):
                try:
                    if getattr(model_class, _attr, None) is None:
                        setattr(model_class, _attr, [])
                except Exception:
                    pass
            config = BertConfig.from_pretrained(
                str(paths.model_path),
                local_files_only=True,
            )
            try:
                config.tie_word_embeddings = False
            except Exception:
                pass
            saved_version = str(getattr(config, "spert_version", "1.0"))
            if saved_version != str(model_class.VERSION):
                raise SpERTRuntimeError(
                    "Checkpoint/source mismatch: checkpoint SpERT version "
                    f"{saved_version}, source version {model_class.VERSION}."
                )
            self._model = model_class.from_pretrained(
                str(paths.model_path),
                config=config,
                cls_token=self._tokenizer.convert_tokens_to_ids("[CLS]"),
                relation_types=self._reader.relation_type_count - 1,
                entity_types=self._reader.entity_type_count,
                max_pairs=self.max_pairs,
                prop_drop=self.prop_drop,
                size_embedding=self.size_embedding,
                freeze_transformer=False,
                local_files_only=True,
            )
        except SpERTRuntimeError:
            raise
        except Exception as exc:
            raise SpERTRuntimeError(f"Could not load the final SpERT checkpoint: {exc}") from exc

        self._device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
        self._model.to(self._device)
        self._model.eval()
        self.model_version = str(getattr(config, "spert_version", "unknown"))
        self.max_position_embeddings = int(getattr(config, "max_position_embeddings", 512))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "engine": "SpERT API",
            "spert_version": self.model_version,
            "encoder": "bert-base-cased",
            "device": str(self._device),
            "entity_types": len(self._types["entities"]),
            "relation_types": len(self._types["relations"]),
            "relation_threshold": self.rel_filter_threshold,
            "max_span_size": self.max_span_size,
            "tokenization": "src.data.preannotate.tokenize",
            "query_case_normalization": "none_true_raw",
            "types_sha256": file_sha256(self.paths.types_path),
            "checkpoint": self.paths.model_path.name,
        }

    def predict(self, text: str) -> dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            raise SpERTRuntimeError("The query text is empty.")
        if len(text) > 5000:
            raise SpERTRuntimeError("The query is too long for a short maintenance-record model.")

        model_text = normalise_query_for_model(text)
        tokens, offsets = tokenize_with_offsets(model_text, self._tokenize)
        if not tokens:
            raise SpERTRuntimeError("The project tokenizer produced no tokens.")
        if len(tokens) > self.max_query_tokens:
            raise SpERTRuntimeError(
                f"The query has {len(tokens)} tokens; the service limit is {self.max_query_tokens}."
            )

        dataset = self._dataset_class(
            "query",
            self._reader._relation_types,
            self._reader._entity_types,
            None,
            None,
            self.max_span_size,
        )
        document = self._reader._parse_document({"tokens": tokens}, dataset)
        if len(document.encoding) > self.max_position_embeddings:
            raise SpERTRuntimeError("The WordPiece encoding exceeds the checkpoint position limit.")
        dataset.switch_mode(self._dataset_class.EVAL_MODE)
        batch = self._sampling.collate_fn_padding([dataset[0]])
        batch = self._util.to_device(batch, self._device)

        with self._lock, self._torch.no_grad():
            self._model.eval()
            entity_clf, rel_clf, rels = self._model(
                encodings=batch["encodings"],
                context_masks=batch["context_masks"],
                entity_masks=batch["entity_masks"],
                entity_sizes=batch["entity_sizes"],
                entity_spans=batch["entity_spans"],
                entity_sample_masks=batch["entity_sample_masks"],
                inference=True,
            )
            batch_entities, batch_relations = self._prediction.convert_predictions(
                entity_clf,
                rel_clf,
                rels,
                batch,
                self.rel_filter_threshold,
                self._reader,
                no_overlapping=False,
            )

        result = format_predictions(
            text=text,
            tokens=tokens,
            offsets=offsets,
            document=document,
            predicted_entities=batch_entities[0],
            predicted_relations=batch_relations[0],
            get_span_tokens=self._util.get_span_tokens,
        )
        result["runtime"] = {
            "engine": "SpERT API",
            "spert_version": self.model_version,
            "device": str(self._device),
            "relation_threshold": self.rel_filter_threshold,
            "query_case_normalization": "none_true_raw",
            "query_case_changed": model_text != text,
        }
        return result
