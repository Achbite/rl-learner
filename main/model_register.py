"""Register an atomically published model with the local distributor."""

import argparse
import json
import time

import grpc
from google.protobuf.json_format import ParseDict

from proto import maze_pb2, maze_pb2_grpc


def register(manifest_path: str, address: str, timeout: float) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    manifest = ParseDict(document, maze_pb2.ModelArtifactManifest())
    deadline = time.monotonic() + timeout
    last_error = ""
    with grpc.insecure_channel(address) as channel:
        stub = maze_pb2_grpc.ModelDistributorServiceStub(channel)
        while time.monotonic() < deadline:
            try:
                response = stub.RegisterModel(
                    maze_pb2.RegisterModelReq(manifest=manifest), timeout=2.0
                )
                if response.result in (
                    maze_pb2.MODEL_REGISTER_RESULT_REGISTERED,
                    maze_pb2.MODEL_REGISTER_RESULT_ALREADY_REGISTERED,
                ):
                    return {
                        "result": maze_pb2.ModelRegisterResult.Name(response.result),
                        "run_id": response.manifest.run_id,
                        "model_version": response.manifest.model_version,
                        "sha256": response.manifest.sha256,
                        "distributor_instance_id": response.distributor_instance_id,
                    }
                raise RuntimeError(response.message or "model registration rejected")
            except grpc.RpcError as exc:
                last_error = exc.details() or str(exc)
            time.sleep(0.2)
    raise RuntimeError(
        f"ModelDistributor registration timeout at {address}: {last_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--address", default="127.0.0.1:9200")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    result = register(args.manifest, args.address, args.timeout)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
