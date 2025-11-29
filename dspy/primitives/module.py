import inspect
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4
import asyncio

import litellm
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.dsp.utils.settings import settings
from dspy.predict.parallel import Parallel
from dspy.primitives.base_module import BaseModule
from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction
from dspy.utils import magicattr
from dspy.utils.callback import with_callbacks
from dspy.utils.inspect_history import pretty_print_history
from dspy.utils.saving import get_dependency_versions
from dspy.utils.usage_tracker import track_usage

logger = logging.getLogger(__name__)


class ProgramMeta(type):
    """Metaclass ensuring every ``dspy.Module`` instance is properly initialised."""

    def __call__(cls, *args, **kwargs):
        # Create the instance without invoking ``__init__`` so we can inject
        # the base initialization beforehand.
        obj = cls.__new__(cls, *args, **kwargs)
        if isinstance(obj, cls):
            # ``_base_init`` sets attributes that should exist on all modules
            # even when a subclass forgets to call ``super().__init__``.
            Module._base_init(obj)
            cls.__init__(obj, *args, **kwargs)

            # Guarantee existence of critical attributes if ``__init__`` didn't
            # create them.
            if not hasattr(obj, "callbacks"):
                obj.callbacks = []
            if not hasattr(obj, "history"):
                obj.history = []
        return obj


class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False
        self.callbacks = []
        self.history = []

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
        self._compiled = False
        # LM calling history of the module.
        self.history = []

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("history", None)
        state.pop("callbacks", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "history"):
            self.history = []
        if not hasattr(self, "callbacks"):
            self.callbacks = []

    @with_callbacks
    def __call__(self, *args, **kwargs) -> Prediction:
        from dspy.dsp.utils.settings import thread_local_overrides

        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        with settings.context(caller_modules=caller_modules):
            if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
                with track_usage() as usage_tracker:
                    output = self.forward(*args, **kwargs)
                tokens = usage_tracker.get_total_tokens()
                self._set_lm_usage(tokens, output)

                return output

            return self.forward(*args, **kwargs)

    @with_callbacks
    async def acall(self, *args, **kwargs) -> Prediction:
        from dspy.dsp.utils.settings import thread_local_overrides

        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        with settings.context(caller_modules=caller_modules):
            if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
                with track_usage() as usage_tracker:
                    output = await self.aforward(*args, **kwargs)
                    tokens = usage_tracker.get_total_tokens()
                    self._set_lm_usage(tokens, output)

                    return output

            return await self.aforward(*args, **kwargs)

    def named_predictors(self):
        from dspy.predict.predict import Predict

        return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]

    def predictors(self):
        return [param for _, param in self.named_predictors()]

    def _get_single_predictor_for_batch(self):
        predictors = self.predictors()
        if len(predictors) != 1:
            raise NotImplementedError(
                "Asynchronous batch creation currently supports modules with a single DSPy Predict instance. "
                f"Detected {len(predictors)} predictors."
            )
        return predictors[0]

    def set_lm(self, lm):
        for _, param in self.named_predictors():
            param.lm = lm

    def get_lm(self):
        all_used_lms = [param.lm for _, param in self.named_predictors()]

        if len(set(all_used_lms)) == 1:
            return all_used_lms[0]

        raise ValueError("Multiple LMs are being used in the module. There's no unique LM to return.")

    def __repr__(self):
        s = []

        for name, param in self.named_predictors():
            s.append(f"{name} = {param}")

        return "\n".join(s)

    def map_named_predictors(self, func):
        """Applies a function to all named predictors."""
        for name, predictor in self.named_predictors():
            set_attribute_by_name(self, name, func(predictor))
        return self

    def inspect_history(self, n: int = 1):
        return pretty_print_history(self.history, n)

    def batch(
        self,
        examples: list[Example],
        num_threads: int | None = None,
        max_errors: int | None = None,
        return_failed_examples: bool = False,
        provide_traceback: bool | None = None,
        disable_progress_bar: bool = False,
    ) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]:
        """
        Processes a list of dspy.Example instances in parallel using the Parallel module.

        Args:
            examples: List of dspy.Example instances to process.
            num_threads: Number of threads to use for parallel processing.
            max_errors: Maximum number of errors allowed before stopping execution.
                If ``None``, inherits from ``dspy.settings.max_errors``.
            return_failed_examples: Whether to return failed examples and exceptions.
            provide_traceback: Whether to include traceback information in error logs.
            disable_progress_bar: Whether to display the progress bar.

        Returns:
            List of results, and optionally failed examples and exceptions.
        """
        # Create a list of execution pairs (self, example)
        exec_pairs = [(self, example.inputs()) for example in examples]

        # Create an instance of Parallel
        parallel_executor = Parallel(
            num_threads=num_threads,
            max_errors=max_errors,
            return_failed_examples=return_failed_examples,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
        )

        # Execute the forward method of Parallel
        if return_failed_examples:
            results, failed_examples, exceptions = parallel_executor.forward(exec_pairs)
            return results, failed_examples, exceptions
        else:
            results = parallel_executor.forward(exec_pairs)
            return results

    def create_batch_file(
        self,
        examples: Sequence[Example | dict[str, Any]],
        *,
        endpoint: str | None = None,
        input_file_path: str | Path | None = None,
        custom_id_prefix: str | None = None,
    ) -> "BatchRequestArtifacts":
        if not examples:
            raise ValueError("`examples` must contain at least one item.")

        predictor = self._get_single_predictor_for_batch()
        adapter = settings.adapter or ChatAdapter()
        normalized_examples = _normalize_examples(examples)
        writer = _BatchRequestWriter(
            module=self,
            predictor=predictor,
            adapter=adapter,
            endpoint_override=endpoint,
            custom_id_prefix=custom_id_prefix,
        )
        request_file, metadata_path, metadata = writer.write(normalized_examples, input_file_path)

        if writer.provider_name is None or writer.endpoint is None or writer.model_name is None:
            raise RuntimeError("Unable to infer LM provider/model information for batch creation.")

        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return BatchRequestArtifacts(
            request_file=request_file,
            metadata_file=metadata_path,
            metadata=metadata,
            provider_name=writer.provider_name,
            model_name=writer.model_name,
            endpoint=writer.endpoint,
        )

    async def acreate_batch(
        self,
        examples: Sequence[Example | dict[str, Any]],
        *,
        completion_window: str = "24h",
        endpoint: str | None = None,
        input_file_path: str | Path | None = None,
        custom_id_prefix: str | None = None,
        custom_llm_provider: str | None = None,
        file_kwargs: dict[str, Any] | None = None,
        **batch_kwargs,
    ):
        """
        Convenience wrapper that creates the batch JSONL file and immediately uploads it
        to LiteLLM using ``litellm.acreate_batch``.

        Returns:
            DSPyBatchHandle describing the created batch job and local artifacts.
        """
        artifacts = self.create_batch_file(
            examples,
            endpoint=endpoint,
            input_file_path=input_file_path,
            custom_id_prefix=custom_id_prefix,
        )
        handle = await self.asubmit_batch_file(
            artifacts,
            completion_window=completion_window,
            custom_llm_provider=custom_llm_provider,
            file_kwargs=file_kwargs,
            **batch_kwargs,
        )
        handle.artifacts = artifacts
        return handle

    async def asubmit_batch_file(
        self,
        artifacts: "BatchRequestArtifacts",
        *,
        completion_window: str = "24h",
        custom_llm_provider: str | None = None,
        file_kwargs: dict[str, Any] | None = None,
        **batch_kwargs,
    ) -> "DSPyBatchHandle":
        provider_name = custom_llm_provider or artifacts.provider_name
        file_payload = {"purpose": "batch"}
        if file_kwargs:
            file_payload.update(file_kwargs)

        with open(artifacts.request_file, "rb") as request_buffer:
            file_obj = await litellm.acreate_file(
                file=request_buffer,
                custom_llm_provider=provider_name,
                **file_payload,
            )

        batch_payload = {
            "completion_window": completion_window or "24h",
            "endpoint": artifacts.endpoint,
            "input_file_id": getattr(file_obj, "id", None),
            "custom_llm_provider": provider_name,
        }
        batch_payload.update(batch_kwargs)
        batch_response = await litellm.acreate_batch(**batch_payload)

        metadata = artifacts.load_metadata()
        metadata["litellm"] = {
            "input_file_id": getattr(file_obj, "id", None),
            "batch_id": getattr(batch_response, "id", None),
            "completion_window": completion_window or "24h",
            "endpoint": artifacts.endpoint,
            "custom_llm_provider": provider_name,
        }
        artifacts.metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.metadata = metadata

        return DSPyBatchHandle(
            batch=batch_response,
            batch_id=getattr(batch_response, "id", None),
            input_file_id=getattr(file_obj, "id", None),
            request_file=artifacts.request_file,
            metadata_file=artifacts.metadata_file,
            artifacts=artifacts,
        )

    async def aretrieve_batch(
        self,
        batch_id: str,
        *,
        custom_llm_provider: str | None = None,
        download_output_path: str | Path | None = None,
        **litellm_kwargs,
    ):
        """
        Retrieve a previously created batch job via ``litellm.aretrieve_batch``.

        Args:
            batch_id: Identifier returned by ``module.acreate_batch`` / LiteLLM.
            custom_llm_provider: Overrides the provider name (defaults to ``"openai"``).
            download_output_path: Optional file path where the batch output JSONL should be saved.
                This path must include the filename (for example, ``Path(\"outputs/batch.jsonl\")``).
                The file is only written once the batch exposes ``output_file_id``.
            **litellm_kwargs: Forwarded to ``litellm.aretrieve_batch`` (e.g., ``timeout``).
        """
        if not batch_id:
            raise ValueError("`batch_id` is required.")

        provider_name = custom_llm_provider
        if provider_name is None:
            try:
                predictor = self._get_single_predictor_for_batch()
            except NotImplementedError:
                predictor = None
            lm = getattr(predictor, "lm", None) if predictor else None
            provider_name = _infer_provider_name(getattr(lm, "model", None))

        batch_response = await litellm.aretrieve_batch(
            batch_id=batch_id,
            custom_llm_provider=provider_name,
            **litellm_kwargs,
        )

        if download_output_path:
            output_file_id = getattr(batch_response, "output_file_id", None)
            if output_file_id:
                saved_path = await _download_litellm_file(
                    file_id=output_file_id,
                    destination=download_output_path,
                    provider_name=provider_name,
                )
                setattr(batch_response, "_dspy_local_output_path", str(saved_path))
            else:
                logger.warning(
                    "Batch %s does not have an `output_file_id` yet. Skipping download to %s.",
                    batch_id,
                    download_output_path,
                )

        return batch_response

    async def aretrieve_batch_predictions(
        self,
        batch_id: str,
        artifacts_or_handle: "BatchRequestArtifacts | DSPyBatchHandle",
        *,
        custom_llm_provider: str | None = None,
        download_output_path: str | Path | None = None,
        **litellm_kwargs,
    ) -> list[Prediction]:
        """
        Retrieve the batch, download its output file (if necessary), and parse it back
        into DSPy ``Prediction`` objects aligned with the original examples.
        """
        artifacts = _resolve_artifacts(artifacts_or_handle)
        output_path = download_output_path or artifacts.metadata_file.with_suffix(".output.jsonl")
        batch_response = await self.aretrieve_batch(
            batch_id=batch_id,
            custom_llm_provider=custom_llm_provider,
            download_output_path=output_path,
            **litellm_kwargs,
        )
        local_output = getattr(batch_response, "_dspy_local_output_path", None) or output_path
        metadata = artifacts.load_metadata()
        predictor = self._get_single_predictor_for_batch()
        adapter = settings.adapter or ChatAdapter()
        predictions = _parse_batch_output_file(
            module=self,
            predictor=predictor,
            adapter=adapter,
            metadata=metadata,
            output_path=Path(local_output),
        )
        return predictions
    
    async def abatch(
        self,
        examples: Sequence[Example | dict[str, Any]],
        *,
        sleep_delay: float = 5., 
        completion_window: str = "24h",
        endpoint: str | None = None,
        input_file_path: str | Path | None = None,
        custom_id_prefix: str | None = None,
        custom_llm_provider: str | None = None,
        **batch_kwargs,
    ):
        """Convenience wrapper that creates the batch, sends it and waits for completion in a non-blocking way."""
        batch_handle = await self.acreate_batch(
            examples,
            completion_window=completion_window,
            endpoint=endpoint,
            input_file_path=input_file_path,
            custom_id_prefix=custom_id_prefix,
            custom_llm_provider=custom_llm_provider,
            **batch_kwargs,
        )

        while True:
            info = await self.aretrieve_batch(batch_handle.batch_id)
            if getattr(info, "status", None) == "completed" and getattr(info, "output_file_id", None):
                break
            await asyncio.sleep(sleep_delay)

        predictions = await self.aretrieve_batch_predictions(
            batch_handle.batch_id,
            batch_handle,
            download_output_path=None,
        )
        return predictions



    def _set_lm_usage(self, tokens: dict[str, Any], output: Any):
        # Some optimizers (e.g., GEPA bootstrap tracing) temporarily patch
        # module.forward to return a tuple: (prediction, trace).
        # When usage tracking is enabled, ensure we attach usage to the
        # prediction object if present.
        prediction_in_output = None
        if isinstance(output, Prediction):
            prediction_in_output = output
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], Prediction):
            prediction_in_output = output[0]
        if prediction_in_output:
            prediction_in_output.set_lm_usage(tokens)
        else:
            logger.warning("Failed to set LM usage. Please return `dspy.Prediction` object from dspy.Module to enable usage tracking.")


    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if name == "forward" and callable(attr):
            # Check if forward is called through __call__ or directly
            stack = inspect.stack()
            forward_called_directly = len(stack) <= 1 or stack[1].function != "__call__"

            if forward_called_directly:
                logger.warning(
                    f"Calling module.forward(...) on {self.__class__.__name__} directly is discouraged. "
                    f"Please use module(...) instead."
                )

        return attr


def set_attribute_by_name(obj, name, value):
    magicattr.set(obj, name, value)


@dataclass
class DSPyBatchHandle:
    batch: Any
    batch_id: str | None
    input_file_id: str | None
    request_file: Path
    metadata_file: Path
    artifacts: "BatchRequestArtifacts | None" = None

    def load_metadata(self) -> dict[str, Any]:
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        return json.loads(self.metadata_file.read_text(encoding="utf-8"))


@dataclass
class BatchRequestArtifacts:
    request_file: Path
    metadata_file: Path
    metadata: dict[str, Any]
    provider_name: str
    model_name: str
    endpoint: str

    def load_metadata(self) -> dict[str, Any]:
        if self.metadata_file.exists():
            self.metadata = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        return self.metadata

    @classmethod
    def from_files(cls, request_file: Path, metadata_file: Path) -> "BatchRequestArtifacts":
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        lm_info = metadata.get("lm", {})
        endpoint = metadata.get("endpoint") or metadata.get("litellm", {}).get("endpoint") or "/v1/chat/completions"
        provider = lm_info.get("provider")
        model = lm_info.get("model")
        if not provider or not model:
            raise ValueError(f"Metadata file {metadata_file} is missing LM information.")
        return cls(
            request_file=request_file,
            metadata_file=metadata_file,
            metadata=metadata,
            provider_name=provider,
            model_name=model,
            endpoint=endpoint,
        )


class _BatchRequestWriter:
    def __init__(
        self,
        module: Module,
        predictor,
        adapter,
        endpoint_override: str | None = None,
        custom_id_prefix: str | None = None,
    ):
        self.module = module
        self.predictor = predictor
        self.adapter = adapter
        self.custom_id_prefix = custom_id_prefix or module.__class__.__name__.lower()

        self._endpoint_override = endpoint_override
        self.provider_name: str | None = None
        self.model_name: str | None = None
        self.endpoint: str | None = None
        self._lm_model_type: str | None = None

    def write(
        self,
        examples: list[dict[str, Any]],
        requested_path: str | Path | None = None,
    ):
        file_path = _resolve_request_file_path(self.module, requested_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        metadata_entries = []
        with open(file_path, "w", encoding="utf-8") as buffer:
            for idx, example_inputs in enumerate(examples):
                request_line = self._build_request_line(example_inputs, idx)
                buffer.write(json.dumps(request_line, ensure_ascii=False))
                buffer.write("\n")
                metadata_entries.append(
                    {
                        "custom_id": request_line["custom_id"],
                        "index": idx,
                        "inputs": _json_safe(example_inputs),
                    }
                )

        metadata = self._build_metadata(entries=metadata_entries, request_file=file_path)
        metadata_path = _metadata_path_for(file_path)

        return file_path, metadata_path, metadata

    def _build_request_line(self, example_inputs: dict[str, Any], position: int) -> dict[str, Any]:
        if not hasattr(self.predictor, "_forward_preprocess"):
            raise NotImplementedError(
                "Batch generation is currently supported for DSPy predictors that implement `_forward_preprocess`."
            )

        lm, config, signature, demos, kwargs = self.predictor._forward_preprocess(**example_inputs)
        if self.provider_name is None or self.model_name is None:
            self.provider_name, self.model_name = _infer_provider_and_model(lm)
        if self.endpoint is None:
            self.endpoint = self._endpoint_override or _default_endpoint_for_model(lm)
        self._lm_model_type = getattr(lm, "model_type", "chat")

        processed_signature = self.adapter._call_preprocess(lm, config, signature, kwargs)
        messages = self.adapter.format(processed_signature, demos, kwargs)
        sanitized_kwargs = _sanitize_lm_kwargs(config)
        body = {
            "model": self.model_name,
            "messages": messages,
            **sanitized_kwargs,
        }

        return {
            "custom_id": f"{self.custom_id_prefix}-{position}-{uuid4().hex[:8]}",
            "method": "POST",
            "url": self.endpoint,
            "body": body,
        }

    def _build_metadata(self, entries: list[dict[str, Any]], request_file: Path) -> dict[str, Any]:
        signature_state = None
        if hasattr(self.predictor, "signature") and hasattr(self.predictor.signature, "dump_state"):
            signature_state = self.predictor.signature.dump_state()

        metadata: dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "request_file": str(request_file),
            "endpoint": self.endpoint,
            "dependency_versions": get_dependency_versions(),
            "module": {
                "class_path": f"{self.module.__class__.__module__}.{self.module.__class__.__name__}",
                "repr": repr(self.module),
            },
            "lm": {
                "model": self.model_name,
                "provider": self.provider_name,
                "type": self._lm_model_type or "chat",
            },
            "examples": entries,
        }
        if signature_state:
            metadata["signature"] = signature_state

        return metadata


def _normalize_examples(examples: Sequence[Example | dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, example in enumerate(examples):
        if isinstance(example, Example):
            try:
                normalized.append(example.inputs().toDict())
            except ValueError as exc:
                raise ValueError(
                    "All DSPy Examples passed to `acreate_batch` must define their input fields via `example.with_inputs(...)`."
                ) from exc
        elif isinstance(example, dict):
            normalized.append(dict(example))
        else:
            raise TypeError(
                f"Unsupported example type at position {idx}: {type(example)}. Use `dspy.Example` or plain dictionaries."
            )
    return normalized


def _resolve_request_file_path(module: Module, requested_path: str | Path | None) -> Path:
    if requested_path:
        path = Path(requested_path)
        return path

    default_dir = Path.cwd() / ".dspy_batches"
    default_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    filename = f"{module.__class__.__name__.lower()}_{timestamp}.jsonl"
    return default_dir / filename


def _metadata_path_for(request_file: Path) -> Path:
    return request_file.with_suffix(request_file.suffix + ".metadata.json")


def _sanitize_lm_kwargs(lm_kwargs: dict[str, Any]) -> dict[str, Any]:
    disallowed = {"cache", "headers", "api_key", "api_base", "base_url", "rollout_id"}
    sanitized: dict[str, Any] = {}
    for key, value in lm_kwargs.items():
        if value is None or key in disallowed:
            continue
        if key == "response_format" and hasattr(value, "model_json_schema"):
            schema = value.model_json_schema()
            sanitized[key] = {"type": "json_schema", "json_schema": schema}
            continue
        sanitized[key] = value
    return sanitized


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if hasattr(value, "toDict"):
        return _json_safe(value.toDict())
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return str(value)


def _infer_provider_and_model(lm) -> tuple[str, str]:
    model = getattr(lm, "model", None)
    if not model:
        raise ValueError(
            "Batch creation requires an LM to be configured on the predictor or via `dspy.configure(lm=...)`."
        )
    provider_name = _infer_provider_name(model)
    model_name = model.split("/", 1)[-1] if "/" in model else model
    return provider_name, model_name


def _infer_provider_name(model: str | None) -> str:
    if not model:
        return "openai"
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def _default_endpoint_for_model(lm) -> str:
    model_type = getattr(lm, "model_type", "chat")
    if model_type == "responses":
        return "/v1/responses"
    if model_type == "text":
        return "/v1/completions"
    return "/v1/chat/completions"


def _resolve_artifacts(ref: "BatchRequestArtifacts | DSPyBatchHandle") -> "BatchRequestArtifacts":
    if isinstance(ref, BatchRequestArtifacts):
        return ref
    if isinstance(ref, DSPyBatchHandle):
        if ref.artifacts is not None:
            return ref.artifacts
        artifacts = BatchRequestArtifacts.from_files(ref.request_file, ref.metadata_file)
        ref.artifacts = artifacts
        return artifacts
    raise TypeError(f"Unsupported artifacts reference: {type(ref)}")


def _parse_batch_output_file(
    module: Module,
    predictor,
    adapter,
    metadata: dict[str, Any],
    output_path: Path,
) -> list[Prediction]:
    if not output_path.exists():
        raise FileNotFoundError(f"Batch output file not found: {output_path}")

    entries = {entry["custom_id"]: entry for entry in metadata.get("examples", [])}
    predictions: list[Prediction | None] = [None] * len(entries)

    with open(output_path, "r", encoding="utf-8") as output_file:
        for line in output_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id")
            if custom_id not in entries:
                continue

            entry = entries[custom_id]
            if record.get("error"):
                raise RuntimeError(f"Batch item {custom_id} failed: {record['error']}")

            body = _extract_response_body(record)
            completions = _build_completions_from_body(body)
            parsed_outputs = adapter._call_postprocess(
                predictor.signature,
                predictor.signature,
                completions,
                predictor.lm,
                {},
            )
            prediction = Prediction.from_completions(parsed_outputs, signature=predictor.signature)
            usage = body.get("usage")
            if usage:
                module._set_lm_usage(usage, prediction)

            predictions[entry["index"]] = prediction

    if any(pred is None for pred in predictions):
        raise RuntimeError(
            "Output file did not contain responses for every request. "
            "Wait for completion or verify the batch output file."
        )
    return predictions  # type: ignore[return-value]


def _extract_response_body(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response") or {}
    if isinstance(response, dict):
        if "body" in response and isinstance(response["body"], dict):
            return response["body"]
        if "response" in response and isinstance(response["response"], dict):
            nested = response["response"]
            if "body" in nested and isinstance(nested["body"], dict):
                return nested["body"]
    return response


def _build_completions_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    choices = body.get("choices", [])
    if choices:
        outputs = []
        for choice in choices:
            message = choice.get("message") or {}
            text = _extract_text_from_message(message)
            completion = {"text": text}
            if message.get("tool_calls"):
                completion["tool_calls"] = message["tool_calls"]
            outputs.append(completion)
        return outputs

    # Responses API style
    output_items = body.get("output", [])
    outputs = []
    for item in output_items:
        if isinstance(item, dict) and "content" in item:
            texts = []
            for chunk in item["content"]:
                if isinstance(chunk, dict):
                    texts.append(chunk.get("text") or chunk.get("string") or "")
            outputs.append({"text": "\n".join(filter(None, texts))})
    return outputs or [{"text": ""}]


def _extract_text_from_message(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                if chunk.get("type") == "text":
                    parts.append(chunk.get("text", ""))
                elif "text" in chunk:
                    parts.append(chunk["text"])
        return "\n".join(parts)
    return ""


async def _download_litellm_file(file_id: str, destination: str | Path, provider_name: str) -> Path:
    content = await litellm.afile_content(file_id=file_id, custom_llm_provider=provider_name)
    data = _ensure_binary_content(content)

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(data)
    return destination_path


def _ensure_binary_content(content: Any) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    if hasattr(content, "read"):
        return content.read()
    if hasattr(content, "content"):
        return _ensure_binary_content(content.content)  # type: ignore[attr-defined]
    if isinstance(content, dict) and "content" in content:
        return _ensure_binary_content(content["content"])
    if isinstance(content, list):
        return "\n".join(str(item) for item in content).encode("utf-8")
    return json.dumps(content, ensure_ascii=False, default=str).encode("utf-8")
