from fastapi import FastAPI


app = FastAPI(title="Remote Attestation Data Release")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready"}
