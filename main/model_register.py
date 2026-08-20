"""Register an exact 0.8 model manifest with the local distributor."""

import argparse
import json
import time

import grpc

from main.training_runtime import TrainingRuntime
from proto import training_pb2, training_pb2_grpc
from src.contracts.identity import manifest_message, model_identity_document


def register(manifest_path: str, address: str, timeout: float) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    manifest = manifest_message(TrainingRuntime._manifest_for_wire(document))
    deadline = time.monotonic() + timeout
    last_error = ""
    with grpc.insecure_channel(address) as channel:
        stub = training_pb2_grpc.ModelDistributorServiceStub(channel)
        while time.monotonic() < deadline:
            try:
                response = stub.RegisterModel(
                    training_pb2.RegisterModelReq(manifest=manifest), timeout=2.0
                )
                if response.result in (
                    training_pb2.MODEL_REGISTER_RESULT_REGISTERED,
                    training_pb2.MODEL_REGISTER_RESULT_ALREADY_REGISTERED,
                ):
                    return {
                        "result": training_pb2.ModelRegisterResult.Name(
                            response.result
                        ),
                        "model": model_identity_document(
                            response.manifest.identity
                        ),
                        "distributor_instance_id": (
                            response.distributor.instance_id
                        ),
                    }
                raise RuntimeError(
                    response.message or "model registration rejected"
                )
            except grpc.RpcError as error:
                last_error = error.details() or str(error)
            time.sleep(0.2)
    raise RuntimeError(
        f"model distributor registration timeout at {address}: {last_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--address", default="127.0.0.1:9200")
    parser.add_argument("--timeout", type=float, default=30.0)
    arguments = parser.parse_args()
    print(
        json.dumps(
            register(arguments.manifest, arguments.address, arguments.timeout),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
