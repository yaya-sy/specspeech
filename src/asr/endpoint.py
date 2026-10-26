"""Native vLLM API serving with a bridge to the in-process ASR drafter."""

import argparse
import asyncio
from http import HTTPStatus
import json
import logging
import math
from queue import Empty, Queue
from threading import Thread
from uuid import uuid4

from .cli import add_asr_arguments, create_engines, validate_arguments


class AdmissionMiddleware:
    """Reject excess transcription requests before parsing uploads or preparing audio."""

    def __init__(self, app, client, max_pending):
        self.app, self.client, self.max_pending = app, client, max_pending
        self.active = 0

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not (path.endswith("/audio/transcriptions") or path == "/invocations"):
            return await self.app(scope, receive, send)
        if self.active >= self.max_pending or self.client.stopping or self.client.errored:
            body = json.dumps({"error": {"message": "ASR server unavailable or at capacity; retry later.",
                                         "type": "server_overloaded", "code": "server_overloaded"}}).encode()
            await send({"type": "http.response.start", "status": 503,
                        "headers": [(b"content-type", b"application/json"), (b"retry-after", b"1")]})
            return await send({"type": "http.response.body", "body": body})
        # No await between checking and incrementing: admission is atomic on the HTTP loop.
        self.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.active -= 1


class DraftRenderer:
    """Validate and retain the raw waveform alongside vLLM's prepared audio features."""

    def __init__(self, renderer):
        self.renderer = renderer

    def __getattr__(self, name):
        return getattr(self.renderer, name)

    async def render_cmpl_async(self, prompts, **kwargs):
        import numpy as np
        from vllm.exceptions import VLLMValidationError

        audios = [prompt.get("multi_modal_data", {}).get("audio") for prompt in prompts]
        if any(audio is None or not np.isfinite(audio).all() for audio in audios):
            raise VLLMValidationError("Audio must contain only finite samples")
        inputs = await self.renderer.render_cmpl_async(prompts, **kwargs)
        for prepared, audio in zip(inputs, audios):
            prepared["specspeech_audio"] = audio
        return inputs


class ASREngineClient:
    """Bridge vLLM's async HTTP frontend to our single shared engine loop."""

    def __init__(self, args, engine_options):
        self.args, self.engine_options = args, engine_options
        self.loop = asyncio.get_running_loop()
        self.ready = self.loop.create_future()
        self.requests = Queue()
        self.futures = {}
        self.capacity = asyncio.Semaphore(args.max_pending)
        self.error = self.engine = None
        self.stopping = False
        self.close_deadline = None
        self.thread = Thread(target=self.run, name="specspeech-asr")
        self.thread.start()

    def __getattr__(self, name):
        return getattr(self.engine, name)

    @property
    def errored(self):
        return self.error is not None

    @property
    def is_running(self):
        return self.thread.is_alive() and not self.errored

    @property
    def dead_error(self):
        return self.error or RuntimeError("ASR engine is unavailable")

    def check_admission(self, n=1, request_id=None):
        if self.errored:
            raise self.dead_error
        if self.stopping:
            raise RuntimeError("ASR engine is stopping")

    async def check_health(self):
        if self.stopping or self.errored or not self.thread.is_alive():
            raise self.dead_error

    async def is_tracing_enabled(self):
        return False

    def complete(self, future, result=None, error=None):
        if not future.done():
            future.set_exception(error) if error is not None else future.set_result(result)

    def fail(self, error):
        self.complete(self.ready, error=error)
        for future in tuple(self.futures.values()):
            self.complete(future, error=error)

    def worker_failed(self, error):
        self.error = error
        logging.getLogger(__name__).exception("ASR endpoint engine failed")
        self.loop.call_soon_threadsafe(self.fail, error)

    def batches(self, size):
        while (first := self.requests.get()) is not None:
            batch = [first]
            # Serve a single request immediately; never wait for a full batch.
            while len(batch) < size:
                try:
                    item = self.requests.get_nowait()
                except Empty:
                    break
                if item is None:
                    return
                batch.append(item)
            yield batch

    def run(self):
        verifier = None
        try:
            from .verifiers.vllm.inference import stream_inference

            # Initialize and drive CUDA/vLLM on the same thread.
            verifier, drafter, punctuator = create_engines(self.args, **self.engine_options)
            if self.stopping:
                return
            self.engine = verifier.engine.llm_engine
            self.renderer = DraftRenderer(self.engine.renderer)
            size = drafter.batch_size if drafter else verifier.max_num_seqs
            self.loop.call_soon_threadsafe(self.complete, self.ready)
            results = stream_inference(
                verifier, (), verifier.prepare, max_pending=self.args.max_pending,
                max_tokens=self.args.max_new_tokens, drafter=drafter, punctuator=punctuator,
                input_batches=self.batches(size), stop_inputs=lambda: self.requests.put(None),
                raw_outputs=True,
            )
            try:
                for sample, result in results:
                    self.loop.call_soon_threadsafe(self.complete, sample["future"], result)
            finally:
                results.close()
        except Exception as error:
            self.worker_failed(error)
        finally:
            if verifier is not None:
                try:
                    verifier.close()
                except Exception as error:
                    self.worker_failed(error)

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        from vllm.exceptions import VLLMValidationError
        from vllm.sampling_params import RequestOutputKind

        if kwargs.get("lora_request") is not None:
            raise VLLMValidationError("LoRA serving is not supported by this adapter")
        prepared = dict(prompt)
        audio = prepared.pop("specspeech_audio", None)
        if audio is None:
            raise VLLMValidationError("Prepared transcription input is missing its raw audio")
        params = sampling_params.clone()
        params.output_kind = RequestOutputKind.FINAL_ONLY
        async with self.capacity:
            self.check_admission()
            if request_id in self.futures:
                raise VLLMValidationError(f"Duplicate request ID: {request_id}")
            future = self.loop.create_future()
            self.futures[request_id] = future

            def on_draft(draft):
                # Unpause a request awaiting drafts when native HTTP cancellation fires.
                self.loop.call_soon_threadsafe(
                    future.add_done_callback,
                    lambda done: draft.append([], finished=True) if done.cancelled() else None,
                )

            try:
                self.requests.put(({
                    # A cancelled caller may reuse its ID before the engine processes its abort.
                    "future": future, "request_id": uuid4().hex, "prepared": prepared,
                    "sampling_params": params, "cancelled": future.cancelled, "on_draft": on_draft,
                    "on_error": lambda error: self.loop.call_soon_threadsafe(self.complete, future, None, error),
                }, audio))
                result = await future
                result.request_id = request_id
                yield result
            finally:
                future.cancel()
                self.futures.pop(request_id, None)

    async def abort(self, request_ids, **kwargs):
        for request_id in [request_ids] if isinstance(request_ids, str) else request_ids:
            if future := self.futures.get(request_id):
                future.cancel()

    async def close(self, timeout=None):
        if not self.stopping:
            self.stopping = True
            budget = self.args.shutdown_timeout if timeout is None else timeout
            self.close_deadline = self.loop.time() + max(0, budget)
            self.ready.cancel()
            for future in tuple(self.futures.values()):
                future.cancel()
            self.requests.put(None)
        remaining = max(0, self.close_deadline - self.loop.time())
        await asyncio.to_thread(self.thread.join, remaining)
        if self.thread.is_alive() and self.error is None:
            self.error = TimeoutError("ASR worker did not stop before the shutdown deadline; restart the process")
            logging.getLogger(__name__).error("%s", self.error)

    def shutdown(self, timeout=None):
        # The native launcher calls this from its shutdown executor.
        budget = self.args.shutdown_timeout if timeout is None else max(0, timeout)
        future = asyncio.run_coroutine_threadsafe(self.close(budget), self.loop)
        try:
            future.result(timeout=budget + 1)
        except TimeoutError:
            self.error = TimeoutError("ASR shutdown exceeded its deadline; restart the process")
            logging.getLogger(__name__).error("%s", self.error)


async def serve(args, engine_options):
    from vllm.entrypoints.launchers.api_server.app_state import init_app_state
    from vllm.entrypoints.launchers.app import build_app
    from vllm.entrypoints.launchers.launcher import serve_http
    from vllm.entrypoints.speech_to_text.transcription.serving import OpenAIServingTranscription

    class ASRTranscription(OpenAIServingTranscription):
        async def create_transcription(self, audio_data, request, raw_request=None):
            if request.stream or request.use_beam_search or request.temperature not in (None, 0):
                return self.create_error_response("SpecSpeech currently supports non-streaming greedy transcription")
            if request.max_completion_tokens is None:
                request.max_completion_tokens = args.max_new_tokens
            if request.max_completion_tokens > args.max_new_tokens:
                return self.create_error_response("max_completion_tokens exceeds --max-new-tokens")
            if request.language is None and args.language:
                request.language = next((code for code, name in self.model_cls.supported_languages.items()
                                         if name.lower() == args.language.lower()), args.language)
            try:
                return await super().create_transcription(audio_data, request, raw_request)
            except (RuntimeError, asyncio.CancelledError):
                if not (self.engine_client.stopping or self.engine_client.errored):
                    raise
                return self.create_error_response(
                    "ASR engine is unavailable; retry later", err_type="ServiceUnavailableError",
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )

    client = ASREngineClient(args, engine_options)
    try:
        await client.ready
        await client.check_health()
        tasks = ("transcription",)
        app = build_app(args, tasks, client.model_config)
        app.middleware_stack = AdmissionMiddleware(
            app.middleware_stack or app.build_middleware_stack(), client, args.max_pending,
        )
        await init_app_state(client, app.state, args, tasks)
        app.state.openai_serving_transcription.shutdown()
        handler = ASRTranscription(client, app.state.openai_serving_models, request_logger=None)
        handler.asr_config.max_audio_clip_s = args.verifier_max_audio_seconds
        app.state.openai_serving_transcription = handler
        app.state.openai_serving_translation.shutdown()
        app.state.openai_serving_translation = None
        shutdown = await serve_http(
            app, sock=None, host=args.host, port=args.port, log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log, enable_ssl_refresh=args.enable_ssl_refresh,
            ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile,
        )
        await shutdown
    finally:
        await client.close()


def main(argv=None):
    from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig
    from vllm.engine.arg_utils import AsyncEngineArgs, get_kwargs
    from vllm.entrypoints.launchers.cli_args import FrontendArgs

    from .verifiers.vllm import QwenASRVerifier

    parser = argparse.ArgumentParser(prog="specspeech asr endpoint")
    # Native app defaults first; explicit ASR defaults below take precedence.
    parser.set_defaults(**{**vars(AsyncEngineArgs()), **vars(FrontendArgs())})
    add_asr_arguments(parser)
    native = parser.add_argument_group("vLLM serving and engine options")
    fields = (
        (FrontendArgs, ("host", "port", "api_key", "ssl_keyfile", "ssl_certfile", "uvicorn_log_level")),
        (ModelConfig, ("model", "served_model_name", "dtype", "quantization", "revision", "seed", "enforce_eager")),
        (LoadConfig, ("download_dir",)),
        (CacheConfig, ("cache_dtype", "enable_prefix_caching")),
        (SchedulerConfig, ("max_num_batched_tokens",)),
    )
    for config, names in fields:
        options = get_kwargs(config)
        for name in names:
            flag = "kv_cache_dtype" if name == "cache_dtype" else name
            native.add_argument("--" + flag.replace("_", "-"), dest=flag,
                                **{**options[name], "default": argparse.SUPPRESS})
    parser.set_defaults(model=QwenASRVerifier.MODEL_ID, served_model_name=["specspeech-asr"],
                        dtype="bfloat16", host="127.0.0.1", disable_log_stats=True)
    parser.add_argument("--shutdown-timeout", type=float, default=10,
                        help="Maximum seconds to wait for the ASR worker to stop (default: 10)")
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    names = QwenASRVerifier.ENGINE_OPTIONS | {"enable_prefix_caching"}
    engine_options = {name: getattr(args, name) for name in names if getattr(args, name, None) is not None}
    engine_options["model_id"] = args.model
    asyncio.run(serve(args, engine_options))
